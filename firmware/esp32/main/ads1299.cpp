#include "ads1299.hpp"

#include <cstring>

#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_rom_sys.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

namespace ads1299 {
namespace {

constexpr const char* kTag = "ads1299";
constexpr uint32_t kDecodeUs = 3;  // >= tSDECODE (4 tCLK = 1.96 us) between command bytes
constexpr uint32_t kCsHoldUs = 3;  // >= 4 tCLK after the last SCLK before CS goes high

struct RegName {
  uint8_t addr;
  const char* name;
};
constexpr RegName kRegNames[] = {
    {kRegConfig1, "CONFIG1"},     {kRegConfig2, "CONFIG2"},     {kRegConfig3, "CONFIG3"},
    {kRegLoff, "LOFF"},           {0x05, "CH1SET"},             {0x06, "CH2SET"},
    {0x07, "CH3SET"},             {0x08, "CH4SET"},             {0x09, "CH5SET"},
    {0x0A, "CH6SET"},             {0x0B, "CH7SET"},             {0x0C, "CH8SET"},
    {kRegBiasSensP, "BIAS_SENSP"}, {kRegBiasSensN, "BIAS_SENSN"}, {kRegLoffSensP, "LOFF_SENSP"},
    {kRegLoffSensN, "LOFF_SENSN"}, {kRegMisc1, "MISC1"},         {kRegConfig4, "CONFIG4"},
};

const char* reg_name(uint8_t addr) {
  for (const auto& r : kRegNames) {
    if (r.addr == addr) return r.name;
  }
  return "?";
}

void set_pin(int pin, int level) {
  if (pin >= 0) gpio_set_level(static_cast<gpio_num_t>(pin), level);
}

esp_err_t config_output(int pin, int initial_level) {
  if (pin < 0) return ESP_OK;
  gpio_config_t io = {};
  io.pin_bit_mask = 1ULL << pin;
  io.mode = GPIO_MODE_OUTPUT;
  io.pull_up_en = GPIO_PULLUP_DISABLE;
  io.pull_down_en = GPIO_PULLDOWN_DISABLE;
  io.intr_type = GPIO_INTR_DISABLE;
  // Set the level before enabling the output to avoid a glitch.
  gpio_set_level(static_cast<gpio_num_t>(pin), initial_level);
  esp_err_t err = gpio_config(&io);
  gpio_set_level(static_cast<gpio_num_t>(pin), initial_level);
  return err;
}

}  // namespace

int Device::gain_code(int gain) {
  switch (gain) {
    case 1: return 0;
    case 2: return 1;
    case 4: return 2;
    case 6: return 3;
    case 8: return 4;
    case 12: return 5;
    case 24: return 6;
    default: return -1;
  }
}

void Device::cs(bool active) {
  if (active) {
    gpio_set_level(static_cast<gpio_num_t>(pins_.cs), 0);
  } else {
    esp_rom_delay_us(kCsHoldUs);
    gpio_set_level(static_cast<gpio_num_t>(pins_.cs), 1);
  }
}

esp_err_t Device::xfer(const uint8_t* tx, uint8_t* rx, size_t n) {
  spi_transaction_t t;
  std::memset(&t, 0, sizeof(t));
  t.length = n * 8;  // bits
  t.tx_buffer = tx;
  t.rx_buffer = rx;
  return spi_device_polling_transmit(dev_, &t);
}

esp_err_t Device::command(uint8_t opcode) {
  uint8_t rx = 0;
  cs(true);
  esp_err_t err = xfer(&opcode, &rx, 1);
  esp_rom_delay_us(kDecodeUs);
  cs(false);
  return err;
}

esp_err_t Device::wreg(uint8_t addr, uint8_t value) {
  const uint8_t bytes[3] = {static_cast<uint8_t>(kWreg | (addr & 0x1F)), 0x00, value};
  uint8_t rx = 0;
  esp_err_t err = ESP_OK;
  cs(true);
  for (int i = 0; i < 3 && err == ESP_OK; ++i) {
    err = xfer(&bytes[i], &rx, 1);
    esp_rom_delay_us(kDecodeUs);
  }
  cs(false);
  return err;
}

esp_err_t Device::rreg(uint8_t addr, uint8_t* value) {
  const uint8_t bytes[3] = {static_cast<uint8_t>(kRreg | (addr & 0x1F)), 0x00, 0x00};
  uint8_t rx[3] = {0, 0, 0};
  esp_err_t err = ESP_OK;
  cs(true);
  for (int i = 0; i < 3 && err == ESP_OK; ++i) {
    err = xfer(&bytes[i], &rx[i], 1);
    esp_rom_delay_us(kDecodeUs);
  }
  cs(false);
  if (err == ESP_OK) *value = rx[2];
  return err;
}

esp_err_t Device::read_frame(uint8_t* buf) {
  static const uint8_t kZeros[kMaxFrameBytes] = {};
  cs(true);
  esp_err_t err = xfer(kZeros, buf, frame_bytes());
  cs(false);
  return err;
}

esp_err_t Device::init(const Pins& pins, const Settings& settings) {
  pins_ = pins;
  set_ = settings;
  const int gcode = gain_code(set_.gain);
  if (gcode < 0) {
    ESP_LOGE(kTag, "invalid PGA gain %d (allowed 1, 2, 4, 6, 8, 12, 24)", set_.gain);
    return ESP_ERR_INVALID_ARG;
  }

  // --- GPIO: CS idle high, START low (no conversions while configuring), RESET/PWDN high.
  esp_err_t err = config_output(pins_.cs, 1);
  if (err == ESP_OK) err = config_output(pins_.start, 0);
  if (err == ESP_OK) err = config_output(pins_.reset, 1);
  if (err == ESP_OK) err = config_output(pins_.pwdn, 1);
  if (err != ESP_OK) {
    ESP_LOGE(kTag, "GPIO setup failed: %s", esp_err_to_name(err));
    return err;
  }

  // --- SPI bus (no DMA: frames are <= 27 bytes) and device in mode 1 with manual CS.
  spi_bus_config_t bus = {};
  bus.mosi_io_num = pins_.mosi;
  bus.miso_io_num = pins_.miso;
  bus.sclk_io_num = pins_.sclk;
  bus.quadwp_io_num = -1;
  bus.quadhd_io_num = -1;
  bus.max_transfer_sz = 64;
  err = spi_bus_initialize(set_.host, &bus, SPI_DMA_DISABLED);
  if (err != ESP_OK) {
    ESP_LOGE(kTag, "spi_bus_initialize failed: %s", esp_err_to_name(err));
    return err;
  }
  spi_device_interface_config_t devcfg = {};
  devcfg.mode = 1;  // CPOL = 0, CPHA = 1
  devcfg.clock_speed_hz = set_.sclk_hz;
  devcfg.spics_io_num = -1;  // CS driven manually (multi-transaction commands)
  devcfg.queue_size = 1;
  err = spi_bus_add_device(set_.host, &devcfg, &dev_);
  if (err != ESP_OK) {
    ESP_LOGE(kTag, "spi_bus_add_device failed: %s", esp_err_to_name(err));
    return err;
  }

  // --- Power-up: wait >= tPOR, then RESET pulse, then SDATAC (device boots in RDATAC).
  vTaskDelay(pdMS_TO_TICKS(200));
  if (pins_.reset >= 0) {
    set_pin(pins_.reset, 0);
    esp_rom_delay_us(10);  // >= 2 tCLK
    set_pin(pins_.reset, 1);
  } else {
    command(kReset);
  }
  esp_rom_delay_us(100);  // >= 18 tCLK before the next command
  command(kSdatac);
  esp_rom_delay_us(10);

  // --- ID: [7:5] REV_ID, [4] = 1, [3:2] DEV_ID = 0b11 (ADS1299), [1:0] NU_CH.
  err = rreg(kRegId, &id_);
  if (err != ESP_OK) {
    ESP_LOGE(kTag, "SPI error reading ID: %s", esp_err_to_name(err));
    return err;
  }
  const uint8_t family = id_ & 0x1F;
  switch (family) {
    case 0x1C: n_channels_ = 4; break;  // ADS1299-4
    case 0x1D: n_channels_ = 6; break;  // ADS1299-6
    case 0x1E: n_channels_ = 8; break;  // ADS1299 (8 ch)
    default:
      ESP_LOGE(kTag,
               "no ADS1299 found: ID register = 0x%02X (expected 0x?C/0x?D/0x?E with bits "
               "[4:2] = 111). 0x00/0xFF usually means no power, wrong pins, wrong SPI mode, "
               "or PWDN/RESET held low.",
               id_);
      n_channels_ = 0;
      return ESP_ERR_NOT_FOUND;
  }
  ESP_LOGI(kTag, "ID = 0x%02X -> ADS1299%s, %d channels, REV_ID %u", id_,
           n_channels_ == 8 ? "" : (n_channels_ == 6 ? "-6" : "-4"), n_channels_,
           static_cast<unsigned>(id_ >> 5));
  const uint8_t chan_bits = static_cast<uint8_t>((1u << n_channels_) - 1u);
  if (set_.used_mask & ~chan_bits) {
    ESP_LOGE(kTag, "channel mapping uses ADS channel(s) 0x%02X beyond this %d-ch device",
             set_.used_mask & ~chan_bits, n_channels_);
    return ESP_ERR_INVALID_ARG;
  }

  // --- Configuration. Internal reference first, then let it settle.
  struct Write {
    uint8_t addr, value, mask;  // mask: bits compared on read-back
  };
  Write writes[4 + kMaxChannels + 6];
  int nw = 0;
  writes[nw++] = {kRegConfig3, static_cast<uint8_t>(set_.bias ? 0xEC : 0xE0), 0xFE};  // bit0 RO
  writes[nw++] = {kRegConfig1, 0x96, 0xFF};  // daisy off, CLK out off, 250 SPS
  writes[nw++] = {kRegConfig2, static_cast<uint8_t>(set_.test_signal ? 0xD0 : 0xC0), 0xFF};
  writes[nw++] = {kRegLoff, 0x00, 0xFF};
  for (int ch = 0; ch < n_channels_; ++ch) {
    uint8_t v;
    if (set_.used_mask & (1u << ch)) {
      v = static_cast<uint8_t>((gcode << 4) | (set_.srb2 ? 0x08 : 0x00) |
                               (set_.test_signal ? 0x05 : 0x00));
    } else {
      v = 0x81;  // powered down, input shorted
    }
    writes[nw++] = {static_cast<uint8_t>(kRegCh1Set + ch), v, 0xFF};
  }
  const uint8_t bias_mask = set_.bias ? static_cast<uint8_t>(set_.used_mask & chan_bits) : 0;
  writes[nw++] = {kRegBiasSensP, bias_mask, chan_bits};
  writes[nw++] = {kRegBiasSensN, bias_mask, chan_bits};
  writes[nw++] = {kRegLoffSensP, 0x00, chan_bits};
  writes[nw++] = {kRegLoffSensN, 0x00, chan_bits};
  writes[nw++] = {kRegMisc1, static_cast<uint8_t>(set_.srb1 ? 0x20 : 0x00), 0xFF};
  writes[nw++] = {kRegConfig4, 0x00, 0xFF};  // continuous conversion, lead-off comp off

  for (int i = 0; i < nw; ++i) {
    err = wreg(writes[i].addr, writes[i].value);
    if (err != ESP_OK) {
      ESP_LOGE(kTag, "SPI error writing %s: %s", reg_name(writes[i].addr), esp_err_to_name(err));
      return err;
    }
    if (i == 0) vTaskDelay(pdMS_TO_TICKS(150));  // internal reference buffer settling
  }

  // --- Read-back verification table.
  bool all_ok = true;
  ESP_LOGI(kTag, "register    wrote  read   ok");
  for (int i = 0; i < nw; ++i) {
    uint8_t rb = 0;
    err = rreg(writes[i].addr, &rb);
    const bool ok = (err == ESP_OK) && ((rb & writes[i].mask) == (writes[i].value & writes[i].mask));
    all_ok = all_ok && ok;
    ESP_LOGI(kTag, "%-10s  0x%02X   0x%02X   %s", reg_name(writes[i].addr), writes[i].value, rb,
             ok ? "yes" : "NO <<<");
  }
  if (!all_ok) {
    ESP_LOGE(kTag, "register read-back mismatch: check SPI wiring / mode / SCLK");
    return ESP_ERR_INVALID_RESPONSE;
  }

  lsb_volts_ = 2.0f * set_.vref_volts / static_cast<float>(set_.gain) / 16777216.0f;
  ESP_LOGI(kTag, "gain %d, VREF %.3f V -> LSB %.4g uV; SRB1 %s, SRB2 %s, bias %s, %s input",
           set_.gain, static_cast<double>(set_.vref_volts),
           static_cast<double>(lsb_volts_) * 1e6, set_.srb1 ? "on" : "off",
           set_.srb2 ? "on" : "off", set_.bias ? "on" : "off",
           set_.test_signal ? "TEST-SIGNAL" : "normal");
  return ESP_OK;
}

esp_err_t Device::start_streaming() {
  esp_err_t err = command(kRdatac);
  if (err != ESP_OK) return err;
  if (pins_.start >= 0) {
    set_pin(pins_.start, 1);
    return ESP_OK;
  }
  return command(kStart);
}

esp_err_t Device::stop_streaming() {
  if (pins_.start >= 0) {
    set_pin(pins_.start, 0);
  } else {
    command(kStop);
  }
  return command(kSdatac);
}

}  // namespace ads1299
