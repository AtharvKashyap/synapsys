#include "decoder_task.hpp"

#include <new>

#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "replay.hpp"
#include "sdkconfig.h"

namespace decoder_task {
namespace {

constexpr const char* kTag = "decoder";
constexpr uint32_t kDecoderStackBytes = 12288;
constexpr TickType_t kPostWait = pdMS_TO_TICKS(10);

// Static storage: the Decoder is large and must never live on a task stack or the heap.
alignas(synapsys::Decoder) uint8_t s_decoder_mem[sizeof(synapsys::Decoder)];
alignas(synapsys::Rearm) uint8_t s_rearm_mem[sizeof(synapsys::Rearm)];
synapsys::Decoder* s_decoder = nullptr;
synapsys::Rearm* s_rearm = nullptr;

app::Context s_ctx;
bool s_replay = false;

void post(const app::Event& ev, TickType_t wait = kPostWait) {
  if (xQueueSend(s_ctx.events, &ev, wait) != pdTRUE) {
    s_ctx.decoder->events_dropped.fetch_add(1, std::memory_order_relaxed);
  }
}

// Raw-stream bookkeeping (decoder task only): samples lost since the last queued record.
uint32_t s_raw_lost = 0;
uint8_t s_raw_lost_stream = 0;
bool s_raw_start_lost = false;

// Queues one raw sample for the console stream; never blocks (a full queue drops the
// sample and reports the gap with the next record that fits).
void queue_raw(const app::Sample& s, uint32_t seq, bool stream_start) {
  app::RawSample rs;
  for (int c = 0; c < synapsys::kChannels; ++c) rs.v[c] = s.v[c];
  rs.seq = seq;
  rs.stream = s.stream;
  rs.lost_before = s_raw_lost;
  rs.lost_stream = s_raw_lost_stream;
  rs.start = stream_start || (s_raw_start_lost && s_raw_lost_stream == s.stream);
  if (xQueueSend(s_ctx.raw, &rs, 0) == pdTRUE) {
    s_raw_lost = 0;
    s_raw_start_lost = false;
  } else {
    if (s_raw_lost == 0) s_raw_lost_stream = s.stream;
    // A lost stream start must still produce a header (with a gap) for that stream.
    if (stream_start) {
      s_raw_start_lost = true;
      s_raw_lost_stream = s.stream;
    }
    ++s_raw_lost;
    s_ctx.decoder->raw_dropped.fetch_add(1, std::memory_order_relaxed);
  }
}

void decoder_task(void* /*arg*/) {
  app::DecoderStats* ds = s_ctx.decoder;
  const synapsys::Config& cfg = *s_ctx.cfg;

  // Lifetime and per-period timing (task-local; published as 32-bit atomics).
  uint64_t dec_sum_us = 0, filt_sum_us = 0;
  uint32_t dec_n = 0, filt_n = 0;
  uint32_t per_min = UINT32_MAX, per_max = 0, per_n = 0;
  uint64_t per_sum = 0;

  uint32_t stream_t = 0;  // samples since the last stream start
  app::Sample s;
  synapsys::WindowResult r;

  for (;;) {
    if (xQueueReceive(s_ctx.samples, &s, portMAX_DELAY) != pdTRUE) continue;

    if (s.flags & app::kFlagNoData) {  // host-feed markers: nothing is pushed
      if (s.flags & app::kFlagEval) {
        const int64_t e0 = esp_timer_get_time();
        const synapsys::WindowResult er = s_decoder->evaluate();
        app::Event ee{};
        ee.type = app::EventType::kEval;
        ee.t = stream_t;
        ee.tag = s.tag;
        ee.result = er;
        ee.compute_us = static_cast<uint32_t>(esp_timer_get_time() - e0);
        post(ee, portMAX_DELAY);
      }
      if (s.flags & app::kFlagSync) {
        app::Event ae{};
        ae.type = app::EventType::kAck;
        ae.tag = s.seq;
        ae.t = s.pass;  // bad lines seen by the feed task
        ae.crc = s.tag;
        post(ae, portMAX_DELAY);
      }
      continue;
    }

    if (s.flags & app::kFlagStreamStart) {
      s_decoder->reset();
      s_rearm->reset();
      stream_t = 0;
      if (s_replay) replay::begin_stream(s.stream);
    }

    // push_f64: DC offset removed in double, then rounded to Real (float on the ESP32).
    const int64_t t0 = esp_timer_get_time();
    const bool got = s_decoder->push_f64(s.v, &r);
    const uint32_t dt = static_cast<uint32_t>(esp_timer_get_time() - t0);
    ++stream_t;
    ds->samples.fetch_add(1, std::memory_order_relaxed);
    if (s_ctx.raw != nullptr) queue_raw(s, stream_t - 1, (s.flags & app::kFlagStreamStart) != 0);

    if (!got) {
      filt_sum_us += dt;
      ++filt_n;
      if ((filt_n & 0xFF) == 0) {
        ds->filt_ns_avg.store(static_cast<uint32_t>(filt_sum_us * 1000u / filt_n),
                              std::memory_order_relaxed);
      }
    } else {
      // --- decision point
      dec_sum_us += dt;
      ++dec_n;
      ds->decisions.fetch_add(1, std::memory_order_relaxed);
      if (r.status == synapsys::Status::kDegenerate) {
        ds->degenerate.fetch_add(1, std::memory_order_relaxed);
      }
      app::atomic_min(ds->dec_us_min, dt);
      app::atomic_max(ds->dec_us_max, dt);
      ds->dec_us_avg.store(static_cast<uint32_t>(dec_sum_us / dec_n), std::memory_order_relaxed);
      per_min = dt < per_min ? dt : per_min;
      per_max = dt > per_max ? dt : per_max;
      per_sum += dt;
      ++per_n;

      const int emit = s_rearm->update(r);

      app::Event ev{};
      ev.type = app::EventType::kDecision;
      ev.t = stream_t;
      ev.stream = s.stream;
      ev.pass = s.pass;
      ev.tag = s.tag;
      ev.result = r;
      ev.rearm_state = s_rearm->state_name();
      ev.compute_us = dt;
      ev.check = app::Check::kNone;
      if (s_replay) replay::check_point(s.stream, stream_t, r, cfg, &ev);
      post(ev, s_ctx.host_feed ? portMAX_DELAY : kPostWait);  // host feed: never drop

      if (emit != synapsys::kReject) {
        ds->commands.fetch_add(1, std::memory_order_relaxed);
        app::Event ce{};
        ce.type = app::EventType::kCommand;
        ce.t = stream_t;
        ce.stream = s.stream;
        ce.pass = s.pass;
        ce.tag = s.tag;
        ce.target = static_cast<int8_t>(emit);
        post(ce, pdMS_TO_TICKS(100));  // commands matter more than debug lines
      }

#if CONFIG_SYNAPSYS_TIMING_EVERY > 0
      if (per_n >= static_cast<uint32_t>(CONFIG_SYNAPSYS_TIMING_EVERY)) {
        app::Event te{};
        te.type = app::EventType::kTiming;
        te.t = stream_t;
        te.t_min_us = per_min;
        te.t_max_us = per_max;
        te.t_avg_us = static_cast<uint32_t>(per_sum / per_n);
        te.t_n = per_n;
        post(te);
        per_min = UINT32_MAX;
        per_max = 0;
        per_sum = 0;
        per_n = 0;
      }
#endif
    }

    if (s_replay && (s.flags & app::kFlagStreamEnd)) {
      app::Event se;
      replay::end_stream(s.stream, s.pass, &se);
      se.t = stream_t;
      post(se, portMAX_DELAY);
    }
    if (s_replay && (s.flags & app::kFlagPassEnd)) {
      app::Event pe;
      replay::end_pass(s.pass, &pe);
      post(pe, portMAX_DELAY);
    }
  }
}

}  // namespace

bool start(const app::Context& ctx, const synapsys::RearmConfig& rearm_cfg, bool replay_mode) {
  s_ctx = ctx;
  s_replay = replay_mode;
  const int64_t t0 = esp_timer_get_time();
  s_decoder = new (s_decoder_mem) synapsys::Decoder(*ctx.cfg);  // placement new: no heap
  s_rearm = new (s_rearm_mem) synapsys::Rearm(rearm_cfg);
  ESP_LOGI(kTag, "Decoder constructed in %lld us; sizeof(Decoder) = %u B, sizeof(Rearm) = %u B",
           static_cast<long long>(esp_timer_get_time() - t0),
           static_cast<unsigned>(sizeof(synapsys::Decoder)),
           static_cast<unsigned>(sizeof(synapsys::Rearm)));
  const BaseType_t rc =
      xTaskCreatePinnedToCore(decoder_task, "decoder", kDecoderStackBytes, nullptr,
                              CONFIG_SYNAPSYS_DECODER_TASK_PRIO, &app::g_decoder_task, 0);
  return rc == pdPASS;
}

}  // namespace decoder_task
