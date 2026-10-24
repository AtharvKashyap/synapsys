#include "host_feed.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"

#if CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG
#include "driver/usb_serial_jtag.h"
#include "driver/usb_serial_jtag_vfs.h"
#else
#include "driver/uart.h"
#include "driver/uart_vfs.h"
#endif

namespace host_feed {
namespace {

constexpr const char* kTag = "host";

app::Context s_ctx;
char s_line[256];  // static: keeps the task stack small

// Makes stdin a blocking, interrupt-driven read of the console port.
bool install_console_rx() {
#if CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG
  usb_serial_jtag_driver_config_t cfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
  cfg.rx_buffer_size = kRxBufferBytes;
  cfg.tx_buffer_size = 4096;
  if (usb_serial_jtag_driver_install(&cfg) != ESP_OK) return false;
  usb_serial_jtag_vfs_use_driver();
#else
  const auto port = static_cast<uart_port_t>(CONFIG_ESP_CONSOLE_UART_NUM);
  if (uart_driver_install(port, kRxBufferBytes, 4096, 0, nullptr, 0) != ESP_OK) return false;
  uart_vfs_dev_use_driver(port);
#endif
  setvbuf(stdin, nullptr, _IONBF, 0);
  return true;
}

void send(app::Sample& s) {
  if (xQueueSend(s_ctx.samples, &s, portMAX_DELAY) == pdTRUE) {
    s_ctx.source->produced.fetch_add(1, std::memory_order_relaxed);
  }
  app::atomic_max(s_ctx.source->queue_high_water,
                  static_cast<uint32_t>(uxQueueMessagesWaiting(s_ctx.samples)));
}

void feed_task(void* /*arg*/) {
  app::SourceStats* st = s_ctx.source;
  st->running.store(true);
  uint32_t stream = 0, seq = 0;
  bool start_pending = false, have_stream = false;
  uint32_t bad_lines = 0;

  for (;;) {
    if (fgets(s_line, sizeof(s_line), stdin) == nullptr) {
      clearerr(stdin);
      vTaskDelay(pdMS_TO_TICKS(10));
      continue;
    }
    char* p = s_line;
    while (*p == ' ' || *p == '\r' || *p == '\n') ++p;
    if (*p == '\0') continue;
    app::Sample s{};
    s.tag = stream;
    const char cmd = *p++;
    switch (cmd) {
      case 'S': {
        if (!have_stream) {
          ++bad_lines;
          break;
        }
        char* end = p;
        bool ok = true;
        for (int c = 0; c < synapsys::kChannels; ++c) {
          char* q = end;
          s.v[c] = std::strtod(q, &end);
          if (end == q) ok = false;
        }
        if (!ok) {
          ++bad_lines;
          ESP_LOGW(kTag, "bad sample line (%u so far): %.60s", static_cast<unsigned>(bad_lines),
                   s_line);
          break;
        }
        s.seq = seq++;
        s.flags = start_pending ? app::kFlagStreamStart : 0;
        start_pending = false;
        send(s);
        break;
      }
      case 'B':
        stream = static_cast<uint32_t>(std::strtoul(p, nullptr, 10));
        seq = 0;
        start_pending = true;
        have_stream = true;
        break;
      case 'E':
        s.flags = app::kFlagNoData | app::kFlagEval;
        send(s);
        break;
      case 'Y':
        s.flags = app::kFlagNoData | app::kFlagSync;
        s.seq = static_cast<uint32_t>(std::strtoul(p, nullptr, 10));
        s.pass = static_cast<uint16_t>(bad_lines > 0xFFFF ? 0xFFFF : bad_lines);
        send(s);
        break;
      default:
        ++bad_lines;
        break;
    }
  }
}

}  // namespace

bool start(const app::Context& ctx) {
  s_ctx = ctx;
  if (!install_console_rx()) {
    ESP_LOGE(kTag, "console RX driver install failed");
    ctx.source->failed.store(true);
    return false;
  }
  TaskHandle_t h = nullptr;
  const BaseType_t ok = xTaskCreatePinnedToCore(feed_task, "host_feed", 4096, nullptr,
                                                CONFIG_SYNAPSYS_ACQ_TASK_PRIO, &h, 1);
  app::g_source_task = h;
  ESP_LOGI(kTag, "host feed ready: send 'Y 0' and wait for the ack (scripts/esp32_benchmark.py)");
  return ok == pdPASS;
}

}  // namespace host_feed
