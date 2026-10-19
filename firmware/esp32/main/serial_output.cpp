#include "serial_output.hpp"

#include <cstdarg>
#include <cstdio>
#include <cstring>

#include "ble_output.h"
#include "driver/gpio.h"
#include "esp_app_desc.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "replay.hpp"
#include "sdkconfig.h"

namespace output {
namespace {

constexpr const char* kTag = "output";

app::Context s_ctx;
bool s_replay = false;
bool s_ble = false;

// Small bounded line builder (no heap).
struct Line {
  char buf[640];
  size_t n = 0;
  void add(const char* fmt, ...) __attribute__((format(printf, 2, 3))) {
    if (n >= sizeof(buf)) return;
    va_list ap;
    va_start(ap, fmt);
    const int w = vsnprintf(buf + n, sizeof(buf) - n, fmt, ap);
    va_end(ap);
    if (w > 0) n = (n + static_cast<size_t>(w) < sizeof(buf)) ? n + w : sizeof(buf) - 1;
  }
  void emit() {
    fputs(kPrefix, stdout);
    fputs(buf, stdout);
    fputc('\n', stdout);
    fflush(stdout);
  }
};

const char* check_name(app::Check c) {
  switch (c) {
    case app::Check::kPass: return "PASS";
    case app::Check::kFail: return "FAIL";
    case app::Check::kNear: return "NEAR_THRESHOLD";
    case app::Check::kUnexpected: return "UNEXPECTED_POINT";
    default: return "NONE";
  }
}

// ---- status LED (plain GPIO) ----
#if CONFIG_SYNAPSYS_LED_GPIO >= 0
constexpr gpio_num_t kLed = static_cast<gpio_num_t>(CONFIG_SYNAPSYS_LED_GPIO);
bool s_led_state = false;
int64_t s_led_hold_until = 0;
void led_init() {
  gpio_config_t io = {};
  io.pin_bit_mask = 1ULL << CONFIG_SYNAPSYS_LED_GPIO;
  io.mode = GPIO_MODE_OUTPUT;
  gpio_config(&io);
  gpio_set_level(kLed, 0);
}
void led_set(bool on) {
  s_led_state = on;
  gpio_set_level(kLed, on ? 1 : 0);
}
void led_decision() {
  if (esp_timer_get_time() >= s_led_hold_until) led_set(!s_led_state);
}
void led_command() {
  s_led_hold_until = esp_timer_get_time() + 300000;
  led_set(true);
}
void led_tick() {
  if (s_led_hold_until != 0 && esp_timer_get_time() >= s_led_hold_until) {
    s_led_hold_until = 0;
    led_set(false);
  }
}
#else
void led_init() {}
void led_decision() {}
void led_command() {}
void led_tick() {}
#endif

void print_decision(const app::Event& ev) {
  const synapsys::Config& cfg = *s_ctx.cfg;
  const synapsys::WindowResult& r = ev.result;
  const bool degenerate = r.status == synapsys::Status::kDegenerate;

#if CONFIG_SYNAPSYS_PRINT_DECISIONS
  Line l;
  l.add("{\"t\":%u", static_cast<unsigned>(ev.t));
  if (s_replay) {
    l.add(",\"stream\":\"%s\",\"pass\":%u", replay::stream_name(ev.stream),
          static_cast<unsigned>(ev.pass));
  }
  l.add(",\"status\":\"%s\",\"scores\":[", degenerate ? "DEGENERATE" : "OK");
  for (int k = 0; k < synapsys::kTargets; ++k) {
    l.add("%s%.6f", k ? "," : "", static_cast<double>(r.scores[k]));
  }
  l.add("],\"winner\":\"%s\",\"margin\":%.6f,\"decision\":\"%s\",\"state\":\"%s\",\"ms\":%.3f}",
        degenerate ? "NONE" : app::label_of(cfg, r.winner), static_cast<double>(r.margin),
        app::label_of(cfg, r.decision), ev.rearm_state ? ev.rearm_state : "?",
        ev.compute_us / 1000.0);
  l.emit();
#endif

  if (ev.check != app::Check::kNone) {
    Line c;
    c.add("{\"check\":\"%s\",\"stream\":\"%s\",\"pass\":%u,\"t\":%u", check_name(ev.check),
          replay::stream_name(ev.stream), static_cast<unsigned>(ev.pass),
          static_cast<unsigned>(ev.t));
    if (ev.exp_degenerate) {
      c.add(",\"expected\":\"DEGENERATE\",\"got\":\"%s\"}", degenerate ? "DEGENERATE" : "OK");
    } else {
      c.add(",\"max_score_err\":%.3g,\"margin_err\":%.3g,\"exp_winner\":\"%s\",\"winner\":\"%s\","
            "\"exp_decision\":\"%s\",\"decision\":\"%s\"}",
            static_cast<double>(ev.max_score_err), static_cast<double>(ev.margin_err),
            app::label_of(cfg, ev.exp_winner), app::label_of(cfg, r.winner),
            app::label_of(cfg, ev.exp_decision), app::label_of(cfg, r.decision));
    }
    c.emit();
  }

  if (s_ble) {
#if CONFIG_SYNAPSYS_BLE_DEBUG
    // Compact (fits a 23-byte ATT MTU only partially; phones usually negotiate more).
    char dbg[96];
    if (degenerate) {
      snprintf(dbg, sizeof(dbg), "t=%u DEGENERATE", static_cast<unsigned>(ev.t));
    } else {
      snprintf(dbg, sizeof(dbg), "%.2f,%.2f,%.2f m%.2f %s", static_cast<double>(r.scores[0]),
               static_cast<double>(r.scores[1]), static_cast<double>(r.scores[2]),
               static_cast<double>(r.margin), app::label_of(cfg, r.decision));
    }
    ble_output_debug(dbg);
#endif
  }
  led_decision();
}

void print_command(const app::Event& ev) {
  const char* label = app::label_of(*s_ctx.cfg, ev.target);
  Line l;
  l.add("{\"event\":\"command\",\"label\":\"%s\",\"t\":%u", label, static_cast<unsigned>(ev.t));
  if (s_replay) l.add(",\"stream\":\"%s\"", replay::stream_name(ev.stream));
  l.add("}");
  l.emit();
  if (s_ble) ble_output_command(label);
  led_command();
}

void print_timing(const app::Event& ev) {
  Line l;
  l.add("{\"event\":\"timing\",\"n\":%u,\"min_ms\":%.3f,\"avg_ms\":%.3f,\"max_ms\":%.3f}",
        static_cast<unsigned>(ev.t_n), ev.t_min_us / 1000.0, ev.t_avg_us / 1000.0,
        ev.t_max_us / 1000.0);
  l.emit();
}

void print_summary(const app::Event& ev) {
  const bool ok = ev.n_fail == 0 && ev.n_missing == 0 && ev.n_unexpected == 0;
  Line l;
  if (ev.type == app::EventType::kStreamSummary) {
    l.add("{\"event\":\"replay_stream\",\"stream\":\"%s\",\"pass\":%u",
          replay::stream_name(ev.stream), static_cast<unsigned>(ev.pass));
  } else {
    l.add("{\"event\":\"replay_summary\",\"pass\":%u", static_cast<unsigned>(ev.pass));
  }
  l.add(",\"result\":\"%s\",\"passed\":%u,\"failed\":%u,\"near_threshold\":%u,\"missing\":%u,"
        "\"unexpected\":%u}",
        ok ? "PASS" : "FAIL", static_cast<unsigned>(ev.n_pass), static_cast<unsigned>(ev.n_fail),
        static_cast<unsigned>(ev.n_near), static_cast<unsigned>(ev.n_missing),
        static_cast<unsigned>(ev.n_unexpected));
  l.emit();
  if (ev.type == app::EventType::kPassSummary) {
    const unsigned total = ev.n_pass + ev.n_fail + ev.n_near + ev.n_missing;
    if (ok) {
      ESP_LOGI(kTag, "==== REPLAY PASS %u: PASS (%u/%u decision points match Python) ====",
               static_cast<unsigned>(ev.pass), static_cast<unsigned>(ev.n_pass), total);
    } else {
      ESP_LOGE(kTag, "==== REPLAY PASS %u: FAIL (%u failed, %u missing, %u unexpected of %u) ====",
               static_cast<unsigned>(ev.pass), static_cast<unsigned>(ev.n_fail),
               static_cast<unsigned>(ev.n_missing), static_cast<unsigned>(ev.n_unexpected), total);
    }
  }
}

unsigned stack_free(TaskHandle_t h) {
  return h ? static_cast<unsigned>(uxTaskGetStackHighWaterMark(h)) : 0u;  // bytes in ESP-IDF
}

void print_health(float fs_measured, float uptime_s) {
  const app::SourceStats& ss = *s_ctx.source;
  const app::DecoderStats& ds = *s_ctx.decoder;
  const uint32_t dmin = ds.dec_us_min.load();
  Line l;
  l.add("{\"event\":\"health\",\"uptime_s\":%.1f,\"source\":\"%s\",\"source_state\":\"%s\","
        "\"fs\":%.2f",
        static_cast<double>(uptime_s), s_ctx.source_name,
        ss.failed.load() ? "FAILED" : (ss.running.load() ? "running" : "idle"),
        static_cast<double>(fs_measured));
  l.add(",\"produced\":%u,\"dropped\":%u,\"drdy_timeouts\":%u,\"missed_drdy\":%u,"
        "\"spi_errors\":%u,\"bad_status\":%u,\"q_hw\":%u,\"q_len\":%d",
        static_cast<unsigned>(ss.produced.load()), static_cast<unsigned>(ss.dropped.load()),
        static_cast<unsigned>(ss.drdy_timeouts.load()),
        static_cast<unsigned>(ss.missed_drdy.load()), static_cast<unsigned>(ss.spi_errors.load()),
        static_cast<unsigned>(ss.bad_status.load()),
        static_cast<unsigned>(ss.queue_high_water.load()), CONFIG_SYNAPSYS_SAMPLE_QUEUE_LEN);
  l.add(",\"decisions\":%u,\"degenerate\":%u,\"commands\":%u,\"events_dropped\":%u",
        static_cast<unsigned>(ds.decisions.load()), static_cast<unsigned>(ds.degenerate.load()),
        static_cast<unsigned>(ds.commands.load()), static_cast<unsigned>(ds.events_dropped.load()));
  l.add(",\"dec_ms\":{\"min\":%.3f,\"avg\":%.3f,\"max\":%.3f},\"filter_us\":%.2f",
        dmin == UINT32_MAX ? 0.0 : dmin / 1000.0, ds.dec_us_avg.load() / 1000.0,
        ds.dec_us_max.load() / 1000.0, ds.filt_ns_avg.load() / 1000.0);
  l.add(",\"heap_free\":%u,\"heap_min\":%u,\"stack_free\":{\"source\":%u,\"decoder\":%u,"
        "\"output\":%u},\"ble\":\"%s\"}",
        static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_DEFAULT)),
        static_cast<unsigned>(heap_caps_get_minimum_free_size(MALLOC_CAP_DEFAULT)),
        stack_free(app::g_source_task), stack_free(app::g_decoder_task),
        stack_free(app::g_output_task),
        s_ble ? (ble_output_connected() ? "connected" : "advertising") : "off");
  l.emit();

  if (s_ble) {
    char status[192];
    snprintf(status, sizeof(status), "fw=%s src=%s fs=%.2f up=%.0fs drop=%u dec=%u cmd=%u",
             esp_app_get_description()->version, s_ctx.source_name,
             static_cast<double>(fs_measured), static_cast<double>(uptime_s),
             static_cast<unsigned>(ss.dropped.load()), static_cast<unsigned>(ds.decisions.load()),
             static_cast<unsigned>(ds.commands.load()));
    ble_output_set_status(status);
  }
}

void output_task(void* /*arg*/) {
  const int64_t period_us = static_cast<int64_t>(CONFIG_SYNAPSYS_HEALTH_PERIOD_MS) * 1000;
  int64_t last_health = esp_timer_get_time();
  uint32_t last_produced = s_ctx.source->produced.load();
  app::Event ev;

  for (;;) {
    TickType_t wait = pdMS_TO_TICKS(100);  // also drives the LED hold timeout
    if (xQueueReceive(s_ctx.events, &ev, wait) == pdTRUE) {
      switch (ev.type) {
        case app::EventType::kDecision: print_decision(ev); break;
        case app::EventType::kCommand: print_command(ev); break;
        case app::EventType::kTiming: print_timing(ev); break;
        case app::EventType::kStreamSummary:
        case app::EventType::kPassSummary: print_summary(ev); break;
      }
    }
    led_tick();

    const int64_t now = esp_timer_get_time();
    if (period_us > 0 && now - last_health >= period_us) {
      const uint32_t produced = s_ctx.source->produced.load();
      const float dt = static_cast<float>(now - last_health) * 1e-6f;
      const float fs = static_cast<float>(produced - last_produced) / dt;
      last_produced = produced;
      last_health = now;
      print_health(fs, static_cast<float>(now) * 1e-6f);
    }
  }
}

}  // namespace

bool start(const app::Context& ctx, bool replay_mode, bool ble_enabled) {
  s_ctx = ctx;
  s_replay = replay_mode;
  s_ble = ble_enabled;
  led_init();
  if (s_ble) {
    char status[128];
    snprintf(status, sizeof(status), "fw=%s src=%s fs=? (starting)",
             esp_app_get_description()->version, ctx.source_name);
    ble_output_set_status(status);
  }
  const BaseType_t rc = xTaskCreatePinnedToCore(output_task, "output", 6144, nullptr,
                                                CONFIG_SYNAPSYS_OUTPUT_TASK_PRIO,
                                                &app::g_output_task, 0);
  if (rc != pdPASS) ESP_LOGE(kTag, "output task creation failed");
  return rc == pdPASS;
}

}  // namespace output
