// SynapSys SSVEP BCI on the ESP32-S3: samples in -> decoder (filter + CCA) -> decision
// -> re-arm -> command out over BLE and the USB console. See ../README.md.
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "acquisition.hpp"
#include "app_types.hpp"
#include "ble_output.h"
#include "decoder_task.hpp"
#include "host_feed.hpp"
#include "esp_app_desc.h"
#include "esp_chip_info.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "nvs_flash.h"
#include "replay.hpp"
#include "sdkconfig.h"
#include "serial_output.hpp"
#include "synapsys/config_v1.hpp"
#include "synapsys/decoder.hpp"
#include "synapsys/rearm.hpp"

namespace {

constexpr const char* kTag = "app";

// Long-lived objects: static, never freed.
synapsys::Config s_cfg;
app::SourceStats s_source_stats;
app::DecoderStats s_decoder_stats;

// Parses a Kconfig decimal string such as "0.30". Returns false (and logs) when invalid.
[[maybe_unused]] bool parse_threshold(const char* name, const char* text, double lo, double hi,
                                      double* out) {
  errno = 0;
  char* end = nullptr;
  const float v = std::strtof(text, &end);
  if (errno != 0 || end == text || *end != '\0' || !std::isfinite(v) || v < lo || v > hi) {
    ESP_LOGE(kTag, "invalid %s \"%s\" (expected a number in [%g, %g]); threshold DISABLED", name,
             text, lo, hi);
    return false;
  }
  *out = static_cast<double>(v);
  return true;
}

void nvs_init() {
  esp_err_t err = nvs_flash_init();  // needed by the BT controller (PHY calibration data)
  if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    nvs_flash_erase();
    err = nvs_flash_init();
  }
  if (err != ESP_OK) ESP_LOGE(kTag, "nvs_flash_init: %s", esp_err_to_name(err));
}

}  // namespace

extern "C" void app_main(void) {
  const esp_app_desc_t* desc = esp_app_get_description();
  esp_chip_info_t chip;
  esp_chip_info(&chip);
  ESP_LOGI(kTag, "SynapSys BCI %s (%s), built %s %s, ESP-IDF %s", desc->version,
           desc->project_name, desc->date, desc->time, desc->idf_ver);
  ESP_LOGI(kTag, "chip: %d cores, rev %d.%d", chip.cores, chip.revision / 100,
           chip.revision % 100);

  // ---- Decoder configuration: frozen v1 + Kconfig overrides.
#if CONFIG_SYNAPSYS_CONFIG_BENCHMARK
  s_cfg = synapsys::config_benchmark_v1();
  const char* config_name = "benchmark_v1";
#else
  s_cfg = synapsys::config_synapsys_v1();
  const char* config_name = "synapsys_v1";
#endif
  s_cfg.hop_samples = CONFIG_SYNAPSYS_HOP_SAMPLES;
#if CONFIG_SYNAPSYS_CONF_THRESHOLD_EN
  s_cfg.has_confidence_threshold = parse_threshold(
      "confidence threshold", CONFIG_SYNAPSYS_CONF_THRESHOLD, 0.0, 1.0, &s_cfg.confidence_threshold);
#endif
#if CONFIG_SYNAPSYS_MARGIN_THRESHOLD_EN
  s_cfg.has_margin_threshold = parse_threshold("margin threshold", CONFIG_SYNAPSYS_MARGIN_THRESHOLD,
                                               0.0, 1.0, &s_cfg.margin_threshold);
#endif
  synapsys::RearmConfig rearm;
  rearm.confirm_count = CONFIG_SYNAPSYS_REARM_CONFIRM;
  rearm.refractory_samples = CONFIG_SYNAPSYS_REARM_REFRACTORY;
  rearm.release_count = CONFIG_SYNAPSYS_REARM_RELEASE;

  ESP_LOGI(kTag, "decoder config: %s", config_name);
  ESP_LOGI(kTag, "decoder: fs %.1f Hz, window %d, hop %d, %d SOS sections, rank_tol %g",
           s_cfg.sample_rate_hz, synapsys::kWindow, s_cfg.hop_samples, s_cfg.n_sections,
           s_cfg.rank_tol);
  for (int k = 0; k < synapsys::kTargets; ++k) {
    ESP_LOGI(kTag, "  target %d: %-5s %.4f Hz", k, s_cfg.labels[k] ? s_cfg.labels[k] : "?",
             s_cfg.frequencies_hz[k]);
  }
  if (s_cfg.has_confidence_threshold) {
    ESP_LOGW(kTag, "confidence threshold T = %.3f (must come from calibration data)",
             s_cfg.confidence_threshold);
  }
  if (s_cfg.has_margin_threshold) {
    ESP_LOGW(kTag, "margin threshold M = %.3f (must come from calibration data)",
             s_cfg.margin_threshold);
  }
  if (!s_cfg.has_confidence_threshold && !s_cfg.has_margin_threshold) {
    ESP_LOGI(kTag, "gating: none (thresholds uncalibrated) -> every window yields a label");
  }
  ESP_LOGI(kTag, "re-arm: confirm K=%d, refractory R=%u samples, release N=%d",
           rearm.confirm_count, static_cast<unsigned>(rearm.refractory_samples),
           rearm.release_count);

  // ---- Input source.
#if CONFIG_SYNAPSYS_INPUT_REPLAY
  const bool replay_mode = true;
  const char* source_name = "replay";
#elif CONFIG_SYNAPSYS_INPUT_HOST
  const bool replay_mode = false;
  const char* source_name = "host";
#elif CONFIG_SYNAPSYS_INPUT_ADS1299_TEST
  const bool replay_mode = false;
  const char* source_name = "ads1299-test";
#else
  const bool replay_mode = false;
  const char* source_name = "ads1299";
#endif
  ESP_LOGI(kTag, "input source: %s", source_name);

  // ---- Queues (allocated once at start-up; nothing is allocated in the hot paths).
  app::Context ctx;
  ctx.samples = xQueueCreate(CONFIG_SYNAPSYS_SAMPLE_QUEUE_LEN, sizeof(app::Sample));
  ctx.events = xQueueCreate(CONFIG_SYNAPSYS_EVENT_QUEUE_LEN, sizeof(app::Event));
  ctx.source = &s_source_stats;
  ctx.decoder = &s_decoder_stats;
  ctx.cfg = &s_cfg;
  ctx.source_name = source_name;
  ctx.config_name = config_name;
#if CONFIG_SYNAPSYS_INPUT_HOST
  ctx.host_feed = true;
#endif
  if (ctx.samples == nullptr || ctx.events == nullptr) {
    ESP_LOGE(kTag, "queue allocation failed (sample queue %d x %u B)",
             CONFIG_SYNAPSYS_SAMPLE_QUEUE_LEN, static_cast<unsigned>(sizeof(app::Sample)));
    return;
  }
  ESP_LOGI(kTag, "sample queue %d x %u B (%.1f s), event queue %d x %u B",
           CONFIG_SYNAPSYS_SAMPLE_QUEUE_LEN, static_cast<unsigned>(sizeof(app::Sample)),
           CONFIG_SYNAPSYS_SAMPLE_QUEUE_LEN / s_cfg.sample_rate_hz,
           CONFIG_SYNAPSYS_EVENT_QUEUE_LEN, static_cast<unsigned>(sizeof(app::Event)));
#if CONFIG_SYNAPSYS_RAW_STREAM
  ctx.raw = xQueueCreate(CONFIG_SYNAPSYS_RAW_QUEUE_LEN, sizeof(app::RawSample));
  if (ctx.raw == nullptr) {
    ESP_LOGE(kTag, "raw queue allocation failed; raw stream disabled");
  } else {
    ESP_LOGI(kTag, "raw stream ON: \"RAW <seq> <O1> <Oz> <O2> <POz>\" (uV) per sample, "
             "queue %d x %u B", CONFIG_SYNAPSYS_RAW_QUEUE_LEN, static_cast<unsigned>(sizeof(app::RawSample)));
  }
#endif

  // ---- BLE.
  bool ble = false;
  nvs_init();
#if CONFIG_SYNAPSYS_BLE_ENABLE
  ble = ble_output_init(CONFIG_SYNAPSYS_BLE_NAME);
  if (!ble) ESP_LOGE(kTag, "BLE init failed; continuing with serial output only");
#endif

  // ---- Tasks: consumers first, then the producer.
  if (!output::start(ctx, replay_mode, ble)) return;
  if (!decoder_task::start(ctx, rearm, replay_mode)) {
    ESP_LOGE(kTag, "decoder task creation failed");
    return;
  }
  bool src_ok;
#if CONFIG_SYNAPSYS_INPUT_REPLAY
  src_ok = replay::start(ctx);
#elif CONFIG_SYNAPSYS_INPUT_HOST
  src_ok = host_feed::start(ctx);
#elif CONFIG_SYNAPSYS_INPUT_ADS1299_TEST
  src_ok = acquisition::start(ctx, /*test_signal=*/true);
#else
  src_ok = acquisition::start(ctx, /*test_signal=*/false);
#endif
  if (!src_ok) ESP_LOGE(kTag, "failed to start input source %s", source_name);

  ESP_LOGI(kTag, "running; heap free %u B (internal %u B). Console lines prefixed \"%s\"",
           static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_DEFAULT)),
           static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL)), output::kPrefix);
}
