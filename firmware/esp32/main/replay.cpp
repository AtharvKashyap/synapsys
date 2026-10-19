#include "replay.hpp"

#include <type_traits>

#include <cmath>
#include <cstring>

#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "generated/fixtures_embedded.hpp"
#include "sdkconfig.h"

namespace replay {
namespace {

constexpr const char* kTag = "replay";
static_assert(fixtures::kChannels == synapsys::kChannels, "fixture channel count mismatch");
static_assert(fixtures::kTargets == synapsys::kTargets, "fixture target count mismatch");
static_assert(std::is_same<std::remove_cv_t<std::remove_pointer_t<decltype(fixtures::Fixture::samples)>>,
                           std::remove_cv_t<std::remove_extent_t<decltype(app::Sample::v)>>>::value,
              "fixture sample type must match app::Sample");
static_assert(fixtures::kWindowSamples == synapsys::kWindow, "fixture window mismatch");

app::Context s_ctx;
TaskHandle_t s_task = nullptr;

// Per-stream / per-pass self-check counters (decoder task only).
struct Counts {
  uint32_t pass = 0, fail = 0, near = 0, missing = 0, unexpected = 0;
};
uint32_t s_seen[fixtures::kNumFixtures] = {};
Counts s_stream[fixtures::kNumFixtures];
Counts s_pass_counts;

// Real-time pacing with the FreeRTOS tick (CONFIG_FREERTOS_HZ = 1000 -> 4 ticks = 250 Hz).
// xTaskDelayUntil keeps an absolute schedule, so the average rate is exact.
static_assert(configTICK_RATE_HZ % 250 == 0, "replay pacing needs a tick rate divisible by 250 Hz");
constexpr TickType_t kSamplePeriodTicks = configTICK_RATE_HZ / 250;

void replay_task(void* /*arg*/) {
  app::SourceStats* st = s_ctx.source;
#if CONFIG_SYNAPSYS_REPLAY_REALTIME
  const bool realtime = true;
#else
  const bool realtime = false;
#endif
#if CONFIG_SYNAPSYS_REPLAY_LOOP
  const bool loop = true;
#else
  const bool loop = false;
#endif
  st->running.store(true);

  uint32_t seq = 0;
  TickType_t last_wake = xTaskGetTickCount();
  for (uint16_t pass = 0;; ++pass) {
    ESP_LOGI(kTag, "pass %u: %d fixtures, %s", static_cast<unsigned>(pass),
             fixtures::kNumFixtures, realtime ? "real time (250 Hz)" : "as fast as possible");
    for (int f = 0; f < fixtures::kNumFixtures; ++f) {
      const fixtures::Fixture& fx = fixtures::kFixtures[f];
      for (uint32_t i = 0; i < fx.n_samples; ++i) {
        app::Sample s;
        std::memcpy(s.v, &fx.samples[static_cast<size_t>(i) * fixtures::kChannels],
                    sizeof(s.v));
        s.seq = seq++;
        s.flags = 0;
        if (i == 0) s.flags |= app::kFlagStreamStart;
        if (i + 1 == fx.n_samples) {
          s.flags |= app::kFlagStreamEnd;
          if (f + 1 == fixtures::kNumFixtures) s.flags |= app::kFlagPassEnd;
        }
        s.stream = static_cast<uint8_t>(f);
        s.pass = pass;
        if (realtime) {
          xTaskDelayUntil(&last_wake, kSamplePeriodTicks);
          // Stream/pass markers must never be lost; plain samples may drop (counted).
          const TickType_t wait = (s.flags != 0) ? portMAX_DELAY : 0;
          if (xQueueSend(s_ctx.samples, &s, wait) == pdTRUE) {
            st->produced.fetch_add(1, std::memory_order_relaxed);
          } else {
            st->dropped.fetch_add(1, std::memory_order_relaxed);
          }
        } else {
          xQueueSend(s_ctx.samples, &s, portMAX_DELAY);
          st->produced.fetch_add(1, std::memory_order_relaxed);
        }
        app::atomic_max(st->queue_high_water,
                        static_cast<uint32_t>(uxQueueMessagesWaiting(s_ctx.samples)));
      }
    }
    if (!loop) break;
    vTaskDelay(pdMS_TO_TICKS(CONFIG_SYNAPSYS_REPLAY_PAUSE_MS));
    last_wake = xTaskGetTickCount();  // restart the schedule after the pause
  }
  st->running.store(false);
  ESP_LOGI(kTag, "replay finished (loop disabled); idling");
  app::g_source_task = nullptr;  // health reporter must not touch a deleted task's stack
  vTaskDelete(nullptr);
}

bool same_label(const char* a, const char* b) {
  return a != nullptr && b != nullptr && std::strcmp(a, b) == 0;
}

}  // namespace

int num_streams() { return fixtures::kNumFixtures; }

const char* stream_name(uint8_t stream) {
  return stream < fixtures::kNumFixtures ? fixtures::kFixtures[stream].id : "?";
}

bool start(const app::Context& ctx) {
  s_ctx = ctx;
  const synapsys::Config& cfg = *ctx.cfg;
  bool ok = std::fabs(cfg.sample_rate_hz - fixtures::kSampleRateHz) < 1e-9;
  for (int k = 0; k < synapsys::kTargets; ++k) {
    ok = ok && same_label(cfg.labels[k], fixtures::kTargetLabels[k]);
  }
  if (!ok) {
    ESP_LOGE(kTag, "embedded fixtures (%s) do not match the decoder config: self-check "
             "results are meaningless", fixtures::kConfigName);
  }
  if (cfg.hop_samples != 125) {
    ESP_LOGW(kTag, "hop_samples = %d != 125: fixture points will be reported missing / "
             "unexpected", cfg.hop_samples);
  }
  for (int f = 0; f < fixtures::kNumFixtures; ++f) {
    ESP_LOGI(kTag, "fixture %d: %-18s %u samples, %u decision points, intended %s", f,
             fixtures::kFixtures[f].id, static_cast<unsigned>(fixtures::kFixtures[f].n_samples),
             static_cast<unsigned>(fixtures::kFixtures[f].n_points),
             fixtures::kFixtures[f].true_label ? fixtures::kFixtures[f].true_label : "none");
  }

  const BaseType_t rc = xTaskCreatePinnedToCore(replay_task, "replay", 4096, nullptr,
                                                CONFIG_SYNAPSYS_ACQ_TASK_PRIO, &s_task, 1);
  app::g_source_task = s_task;
  return rc == pdPASS;
}

void begin_stream(uint8_t stream) {
  if (stream >= fixtures::kNumFixtures) return;
  s_seen[stream] = 0;
  s_stream[stream] = Counts{};
}

void check_point(uint8_t stream, uint32_t t, const synapsys::WindowResult& r,
                 const synapsys::Config& cfg, app::Event* ev) {
  ev->check = app::Check::kUnexpected;
  ev->max_score_err = 0.0f;
  ev->margin_err = 0.0f;
  ev->exp_winner = -1;
  ev->exp_decision = -1;
  ev->exp_degenerate = false;
  if (stream >= fixtures::kNumFixtures) return;
  const fixtures::Fixture& fx = fixtures::kFixtures[stream];
  Counts& sc = s_stream[stream];

  int idx = -1;
  for (uint32_t i = 0; i < fx.n_points; ++i) {
    if (fx.points[i].window_end == t) {
      idx = static_cast<int>(i);
      break;
    }
  }
  if (idx < 0) {
    ++sc.unexpected;
    return;
  }
  s_seen[stream] |= (1u << idx);
  const fixtures::ExpectedPoint& p = fx.points[idx];
  ev->exp_degenerate = p.degenerate;

  bool pass;
  bool near = false;
  if (p.degenerate) {
    pass = (r.status == synapsys::Status::kDegenerate) && (r.decision == synapsys::kReject);
  } else {
    double max_err = 0.0;
    for (int k = 0; k < synapsys::kTargets; ++k) {
      const double e = std::fabs(static_cast<double>(r.scores[k]) - p.scores[k]);
      if (e > max_err) max_err = e;
    }
    const double margin_err = std::fabs(static_cast<double>(r.margin) - p.margin);
    ev->max_score_err = static_cast<float>(max_err);
    ev->margin_err = static_cast<float>(margin_err);
    ev->exp_winner = p.winner;

    // Expected decision under the *configured* gating, applied to the Python scores.
    const double conf = p.scores[p.winner];
    bool accepted = true;
    if (cfg.has_confidence_threshold) {
      accepted = accepted && conf >= cfg.confidence_threshold;
      near = near || std::fabs(conf - cfg.confidence_threshold) < kScoreTol;
    }
    if (cfg.has_margin_threshold) {
      accepted = accepted && p.margin >= cfg.margin_threshold;
      near = near || std::fabs(p.margin - cfg.margin_threshold) < kScoreTol;
    }
    const int exp_decision = accepted ? p.winner : synapsys::kReject;
    ev->exp_decision = static_cast<int8_t>(exp_decision);

    const bool numeric_ok = r.status == synapsys::Status::kOk && r.winner == p.winner &&
                            max_err < kScoreTol && margin_err < kScoreTol;
    pass = numeric_ok && r.decision == exp_decision;
    near = near && numeric_ok && !pass;  // only a gating flip within 1e-3 of a threshold
  }

  if (pass) {
    ev->check = app::Check::kPass;
    ++sc.pass;
  } else if (near) {
    ev->check = app::Check::kNear;
    ++sc.near;
  } else {
    ev->check = app::Check::kFail;
    ++sc.fail;
  }
}

void end_stream(uint8_t stream, uint16_t pass, app::Event* ev) {
  *ev = app::Event{};
  ev->type = app::EventType::kStreamSummary;
  ev->stream = stream;
  ev->pass = pass;
  if (stream >= fixtures::kNumFixtures) return;
  Counts& sc = s_stream[stream];
  const fixtures::Fixture& fx = fixtures::kFixtures[stream];
  sc.missing = 0;
  for (uint32_t i = 0; i < fx.n_points; ++i) {
    if (!(s_seen[stream] & (1u << i))) ++sc.missing;
  }
  ev->n_pass = sc.pass;
  ev->n_fail = sc.fail;
  ev->n_near = sc.near;
  ev->n_missing = sc.missing;
  ev->n_unexpected = sc.unexpected;
  s_pass_counts.pass += sc.pass;
  s_pass_counts.fail += sc.fail;
  s_pass_counts.near += sc.near;
  s_pass_counts.missing += sc.missing;
  s_pass_counts.unexpected += sc.unexpected;
}

void end_pass(uint16_t pass, app::Event* ev) {
  *ev = app::Event{};
  ev->type = app::EventType::kPassSummary;
  ev->pass = pass;
  ev->n_pass = s_pass_counts.pass;
  ev->n_fail = s_pass_counts.fail;
  ev->n_near = s_pass_counts.near;
  ev->n_missing = s_pass_counts.missing;
  ev->n_unexpected = s_pass_counts.unexpected;
  s_pass_counts = Counts{};
}

}  // namespace replay
