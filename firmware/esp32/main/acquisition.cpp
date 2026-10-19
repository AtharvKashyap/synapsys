#include "acquisition.hpp"

#include "ads1299.hpp"
#include "driver/gpio.h"
#include "esp_attr.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"

namespace acquisition {
namespace {

constexpr const char* kTag = "acq";

struct TaskArgs {
  app::Context ctx;
  bool test_signal;
};

TaskArgs s_args;
TaskHandle_t s_task = nullptr;
ads1299::Device s_dev;  // static: never on a task stack

void IRAM_ATTR drdy_isr(void* /*arg*/) {
  BaseType_t woken = pdFALSE;
  vTaskNotifyGiveFromISR(s_task, &woken);
  portYIELD_FROM_ISR(woken);
}

// ADS1299 channel (0-based) feeding each decoder channel O1, Oz, O2, POz.
constexpr int kChannelMap[synapsys::kChannels] = {
    CONFIG_SYNAPSYS_CH_O1 - 1, CONFIG_SYNAPSYS_CH_OZ - 1, CONFIG_SYNAPSYS_CH_O2 - 1,
    CONFIG_SYNAPSYS_CH_POZ - 1};

[[noreturn]] void fail_forever(app::SourceStats* st, const char* why) {
  st->failed.store(true);
  for (;;) {
    ESP_LOGE(kTag, "acquisition stopped: %s (fix and reset the board)", why);
    vTaskDelay(pdMS_TO_TICKS(5000));
  }
}

void acquisition_task(void* /*arg*/) {
  s_task = xTaskGetCurrentTaskHandle();  // ISR target (also set by xTaskCreate)
  const app::Context& ctx = s_args.ctx;
  app::SourceStats* st = ctx.source;

  // Channel map sanity: distinct channels.
  uint8_t used = 0;
  for (int c = 0; c < synapsys::kChannels; ++c) {
    const uint8_t bit = static_cast<uint8_t>(1u << kChannelMap[c]);
    if (used & bit) fail_forever(st, "channel mapping (Kconfig) uses the same ADS channel twice");
    used |= bit;
  }

  ads1299::Pins pins{};
  pins.sclk = CONFIG_SYNAPSYS_PIN_SCLK;
  pins.mosi = CONFIG_SYNAPSYS_PIN_MOSI;
  pins.miso = CONFIG_SYNAPSYS_PIN_MISO;
  pins.cs = CONFIG_SYNAPSYS_PIN_CS;
  pins.drdy = CONFIG_SYNAPSYS_PIN_DRDY;
  pins.start = CONFIG_SYNAPSYS_PIN_START;
  pins.reset = CONFIG_SYNAPSYS_PIN_RESET;
  pins.pwdn = CONFIG_SYNAPSYS_PIN_PWDN;

  ads1299::Settings set{};
  set.host = SPI2_HOST;
  set.sclk_hz = CONFIG_SYNAPSYS_SPI_SCLK_KHZ * 1000;
  set.gain = CONFIG_SYNAPSYS_ADS_GAIN;
  set.vref_volts = static_cast<float>(CONFIG_SYNAPSYS_ADS_VREF_MV) / 1000.0f;
  set.test_signal = s_args.test_signal;
#if CONFIG_SYNAPSYS_ADS_SRB1
  set.srb1 = true;
#else
  set.srb1 = false;
#endif
#if CONFIG_SYNAPSYS_ADS_SRB2
  set.srb2 = true;
#else
  set.srb2 = false;
#endif
#if CONFIG_SYNAPSYS_ADS_BIAS
  set.bias = true;
#else
  set.bias = false;
#endif
  set.used_mask = used;

  ESP_LOGI(kTag, "pins: SCLK %d MOSI %d MISO %d CS %d DRDY %d START %d RESET %d PWDN %d; "
           "SCLK %d kHz; map O1=CH%d Oz=CH%d O2=CH%d POz=CH%d",
           pins.sclk, pins.mosi, pins.miso, pins.cs, pins.drdy, pins.start, pins.reset,
           pins.pwdn, CONFIG_SYNAPSYS_SPI_SCLK_KHZ, kChannelMap[0] + 1, kChannelMap[1] + 1,
           kChannelMap[2] + 1, kChannelMap[3] + 1);

  if (s_dev.init(pins, set) != ESP_OK) fail_forever(st, "ADS1299 initialisation failed");

  // DRDY interrupt. Installing the ISR service from this task places it on core 1.
  gpio_config_t io = {};
  io.pin_bit_mask = 1ULL << pins.drdy;
  io.mode = GPIO_MODE_INPUT;
  io.pull_up_en = GPIO_PULLUP_ENABLE;
  io.pull_down_en = GPIO_PULLDOWN_DISABLE;
  io.intr_type = GPIO_INTR_NEGEDGE;
  if (gpio_config(&io) != ESP_OK) fail_forever(st, "DRDY gpio_config failed");
  esp_err_t err = gpio_install_isr_service(0);
  if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) fail_forever(st, "gpio_install_isr_service");
  if (gpio_isr_handler_add(static_cast<gpio_num_t>(pins.drdy), drdy_isr, nullptr) != ESP_OK) {
    fail_forever(st, "gpio_isr_handler_add(DRDY) failed");
  }

  ulTaskNotifyTake(pdTRUE, 0);  // discard anything pending
  if (s_dev.start_streaming() != ESP_OK) fail_forever(st, "RDATAC/START failed");
  st->running.store(true);
  ESP_LOGI(kTag, "streaming: %u-byte frames on DRDY, LSB %.4g uV",
           static_cast<unsigned>(s_dev.frame_bytes()),
           static_cast<double>(s_dev.lsb_volts()) * 1e6);

  const TickType_t timeout = pdMS_TO_TICKS(CONFIG_SYNAPSYS_DRDY_TIMEOUT_MS);
  const double lsb = static_cast<double>(s_dev.lsb_volts());
  uint8_t frame[ads1299::kMaxFrameBytes];
  uint32_t seq = 0;
  uint32_t consecutive_timeouts = 0;

  for (;;) {
    const uint32_t n = ulTaskNotifyTake(pdTRUE, timeout);
    if (n == 0) {
      st->drdy_timeouts.fetch_add(1, std::memory_order_relaxed);
      if (++consecutive_timeouts % 50 == 1) {
        ESP_LOGW(kTag, "no DRDY for %d ms (%u timeouts in a row)",
                 CONFIG_SYNAPSYS_DRDY_TIMEOUT_MS, static_cast<unsigned>(consecutive_timeouts));
      }
      continue;
    }
    consecutive_timeouts = 0;
    if (n > 1) st->missed_drdy.fetch_add(n - 1, std::memory_order_relaxed);

    if (s_dev.read_frame(frame) != ESP_OK) {
      st->spi_errors.fetch_add(1, std::memory_order_relaxed);
      continue;
    }
    // Status word must start with 0b1100; a mismatch means a misaligned/garbled frame.
    // The sample is still forwarded (keeps the 250 Hz timeline) but counted.
    if (!ads1299::Device::status_ok(frame)) {
      st->bad_status.fetch_add(1, std::memory_order_relaxed);
    }

    app::Sample s;
    for (int c = 0; c < synapsys::kChannels; ++c) {
      const int32_t code = ads1299::Device::to_int24(&frame[3 + 3 * kChannelMap[c]]);
      s.v[c] = static_cast<double>(code) * lsb;
    }
    s.seq = seq;
    s.flags = (seq == 0) ? app::kFlagStreamStart : 0;
    s.stream = 0;
    s.pass = 0;
    ++seq;

    if (xQueueSend(ctx.samples, &s, 0) == pdTRUE) {
      st->produced.fetch_add(1, std::memory_order_relaxed);
    } else {
      st->dropped.fetch_add(1, std::memory_order_relaxed);
    }
    app::atomic_max(st->queue_high_water,
                    static_cast<uint32_t>(uxQueueMessagesWaiting(ctx.samples)));
  }
}

}  // namespace

bool start(const app::Context& ctx, bool test_signal) {
  s_args.ctx = ctx;
  s_args.test_signal = test_signal;
  const BaseType_t ok = xTaskCreatePinnedToCore(acquisition_task, "acq", 4096, nullptr,
                                                CONFIG_SYNAPSYS_ACQ_TASK_PRIO, &s_task, 1);
  app::g_source_task = s_task;
  return ok == pdPASS;
}

}  // namespace acquisition
