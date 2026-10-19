// Minimal ADS1299 driver (SPI mode 1, manual CS, RDATAC streaming).
//
// Timing rules honoured (ADS1299 datasheet, fCLK = 2.048 MHz internal oscillator):
//   * power-up: PWDN/RESET high, wait >= tPOR (2^18 tCLK = 128 ms) -> wait 200 ms;
//   * RESET pulse >= 2 tCLK, then 18 tCLK before the first command;
//   * multi-byte commands (RREG/WREG): >= 4 tCLK (tSDECODE, 1.96 us) between bytes -> 3 us;
//   * CS held >= 4 tCLK after the last SCLK edge before release -> 3 us.
#pragma once

#include <cstddef>
#include <cstdint>

#include "driver/spi_master.h"
#include "esp_err.h"

namespace ads1299 {

// Opcodes
inline constexpr uint8_t kWakeup = 0x02, kStandby = 0x04, kReset = 0x06, kStart = 0x08,
                         kStop = 0x0A, kRdatac = 0x10, kSdatac = 0x11, kRdata = 0x12,
                         kRreg = 0x20, kWreg = 0x40;
// Registers
inline constexpr uint8_t kRegId = 0x00, kRegConfig1 = 0x01, kRegConfig2 = 0x02,
                         kRegConfig3 = 0x03, kRegLoff = 0x04, kRegCh1Set = 0x05,
                         kRegBiasSensP = 0x0D, kRegBiasSensN = 0x0E, kRegLoffSensP = 0x0F,
                         kRegLoffSensN = 0x10, kRegLoffFlip = 0x11, kRegGpio = 0x14,
                         kRegMisc1 = 0x15, kRegMisc2 = 0x16, kRegConfig4 = 0x17;
inline constexpr int kMaxChannels = 8;
inline constexpr size_t kMaxFrameBytes = 3 + 3 * kMaxChannels;

struct Pins {
  int sclk, mosi, miso, cs, drdy;
  int start;  // -1: not connected (START command used)
  int reset;  // -1: not connected (RESET command used)
  int pwdn;   // -1: tied high on the PCB
};

struct Settings {
  spi_host_device_t host = SPI2_HOST;
  int sclk_hz = 2000000;
  int gain = 24;              // 1, 2, 4, 6, 8, 12, 24
  float vref_volts = 4.5f;
  bool test_signal = false;   // CONFIG2.INT_CAL + CHnSET mux = test
  bool srb1 = true;           // MISC1.SRB1
  bool srb2 = false;          // CHnSET.SRB2 on used channels
  bool bias = false;          // CONFIG3 bias buffer + BIAS_SENSP/N on used channels
  uint8_t used_mask = 0x0F;   // bit k = ADS channel k+1 used; others powered down
};

class Device {
 public:
  // Bus + GPIO setup, power-up sequence, ID check, register configuration with read-back
  // verification (logged as a table). Leaves the device stopped in SDATAC mode.
  esp_err_t init(const Pins& pins, const Settings& settings);

  // RDATAC + START (pin or command). DRDY then falls every 4 ms at 250 SPS.
  esp_err_t start_streaming();
  esp_err_t stop_streaming();

  // Read one RDATAC frame (3 status bytes + 3 bytes per channel) into buf.
  // buf must hold frame_bytes(); call after each DRDY falling edge.
  esp_err_t read_frame(uint8_t* buf);

  int num_channels() const { return n_channels_; }
  size_t frame_bytes() const { return 3 + 3 * static_cast<size_t>(n_channels_); }
  uint8_t id() const { return id_; }
  float lsb_volts() const { return lsb_volts_; }  // 2 * VREF / GAIN / 2^24

  static int32_t to_int24(const uint8_t* p) {
    int32_t v = (static_cast<int32_t>(p[0]) << 16) | (static_cast<int32_t>(p[1]) << 8) | p[2];
    if (v & 0x800000) v -= 0x1000000;  // sign-extend 24-bit two's complement
    return v;
  }
  static bool status_ok(const uint8_t* frame) { return (frame[0] & 0xF0) == 0xC0; }

  // Register access (SDATAC mode only).
  esp_err_t command(uint8_t opcode);
  esp_err_t wreg(uint8_t addr, uint8_t value);
  esp_err_t rreg(uint8_t addr, uint8_t* value);

 private:
  esp_err_t xfer(const uint8_t* tx, uint8_t* rx, size_t n);
  void cs(bool active);
  static int gain_code(int gain);

  spi_device_handle_t dev_ = nullptr;
  Pins pins_{};
  Settings set_{};
  uint8_t id_ = 0;
  int n_channels_ = 0;
  float lsb_volts_ = 0.0f;
};

}  // namespace ads1299
