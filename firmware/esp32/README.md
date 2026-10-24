# SynapSys BCI firmware (ESP32-S3, ESP-IDF v5.3)

The ESP32-S3 runs the whole BCI by itself:

```
ADS1299 (DRDY ISR, core 1) ─┐                                  ┌─► USB console: "SSVEP {json}" lines
                            ├─► sample queue ─► decoder task ──┤
fixture replay (250 Hz)   ─┘    (1024 = 4 s)    (core 0)       └─► BLE GATT: Command / Debug / Status
                                filter + CCA every 125 samples → decision → re-arm → command
```

The decoder is the platform-independent core in `../decoder` (frozen Python reference v1,
float32). This project only links it as the ESP-IDF component `decoder`. The core is never
modified here.

**First flash:** the default input is **fixture replay**, which needs only a bare ESP32-S3
board. It streams the embedded Python fixtures at 250 Hz and checks every decision point
against the Python reference.

## Build, flash, monitor

```bash
. ~/esp/esp-idf/export.sh                 # ESP-IDF v5.3.x
cd firmware/esp32
idf.py set-target esp32s3                 # once
idf.py menuconfig                         # optional: "SynapSys BCI" menu
idf.py build
idf.py -p /dev/cu.usbmodem* flash monitor # native USB port ("USB" on DevKitC-1)
idf.py size                               # flash / RAM usage
```

The console is the **USB-Serial/JTAG** port, so the board's native "USB" connector does
flashing and monitoring on one cable. To use the "UART" connector instead, set
`Component config → ESP System Settings → Channel for console output` to UART0.
`sdkconfig.defaults` only applies when `sdkconfig` does not exist yet. After you edit it,
delete `sdkconfig` and rebuild.

Measured build (IDF v5.3.2, default config): image 815 KB (1.5 MB "single app large"
partition, 4 MB flash setting). Flash data is 367 KB, including 281 KB of embedded fixtures.
Static DRAM is 141 KB of 342 KB, including the ~50 kB `Decoder` in `.bss`.

## Input sources (`SynapSys BCI → Input source`)

| Choice | What it does |
|---|---|
| **Fixture replay** (default) | Streams 6 embedded synthetic fixtures (`main/generated/fixtures_embedded.hpp`) through the real queue → decoder → re-arm → outputs at 250 Hz. Prints a PASS/FAIL line per decision point, a summary per fixture and a summary per pass. Loops forever by default, which doubles as a heap/stack soak test. |
| ADS1299 live | Live EEG from the ADS1299. |
| ADS1299 internal test signal | CONFIG2 = 0xD0 (INT_CAL) and the CHnSET mux of each used channel set to the test signal (gain 24 → 0x65). This proves the SPI, DRDY and scaling path without electrodes: expect a ~1 Hz square wave of about ±1.9 mV. Decoder labels are meaningless in this mode. |

### Verifying with replay mode

Expected console output after about 36 s (6 fixtures × 6 s):

```
SSVEP {"check":"PASS","stream":"syn_yes_snr-15","pass":0,"t":625,"max_score_err":...,...}
...
SSVEP {"event":"replay_summary","pass":0,"result":"PASS","passed":48,"failed":0,"near_threshold":0,"missing":0,"unexpected":0}
I (..) output: ==== REPLAY PASS 0: PASS (48/48 decision points match Python) ====
```

Pass criteria for each point:

- each score |err| < 1e-3;
- margin |err| < 1e-3;
- exact winner;
- exact decision under the *configured* gating, applied to the Python scores;
- `syn_flat` must report `DEGENERATE` with decision REJECT at all 8 points.

A gating flip within 1e-3 of a configured threshold is reported as `NEAR_THRESHOLD`, not
silently passed. The self-check assumes hop = 125 (the default).

```bash
idf.py monitor | tee replay.log
grep '^SSVEP ' replay.log | sed 's/^SSVEP //' > replay.jsonl      # machine-readable
grep -c '"check":"FAIL"' replay.jsonl
```

To change the embedded set:

```bash
python3 firmware/tools/embed_fixtures.py [--fixtures syn_yes_snr-15 ...]
```

Commit the regenerated header. The samples are float64 volts, which is what the
acquisition path delivers.

## Wiring (defaults for ESP32-S3-DevKitC-1, all set in Kconfig)

| ADS1299 pin | ESP32-S3 GPIO | Notes |
|---|---|---|
| SCLK | 12 | SPI2_HOST, mode 1 (CPOL 0, CPHA 1), 2 MHz default |
| DIN | 11 (MOSI) | |
| DOUT | 13 (MISO) | |
| CS | 10 | driven manually as a GPIO (multi-byte command timing) |
| DRDY | 9 | falling-edge interrupt, internal pull-up enabled |
| START | 8 | `-1`: tie START low on the PCB; the START opcode is used instead |
| RESET | 7 | `-1`: tie high; the RESET opcode is used instead |
| PWDN | 6 | `-1`: tied high on the PCB |
| DGND / DVDD | GND / 3V3 | DVDD must match the ESP32 3.3 V logic level |

Channel mapping: `ADS1299 channel carrying O1/Oz/O2/POz` (defaults are CH1–CH4). This feeds
the decoder in the order O1, Oz, O2, POz. Unused channels are powered down (CHnSET = 0x81).

## ADS1299 configuration

Power-up sequence:

1. Bring PWDN and RESET high and wait 200 ms (tPOR is 128 ms).
2. Pulse RESET low for 10 µs.
3. Send SDATAC.
4. Read the ID register.
5. Write the registers below.
6. Read back every register and log the table (`ads1299: register wrote read ok`).
7. Send RDATAC, then raise START.

The ID check accepts `ID & 0x1F` ∈ {0x1C, 0x1D, 0x1E} (ADS1299-4/-6/-8; REV_ID bits are
ignored). Anything else, typically 0x00 or 0xFF, logs a clear error, and acquisition
stops. The console says why every 5 s.

| Register | Value | Kconfig |
|---|---|---|
| CONFIG1 | 0x96 (250 SPS) | – |
| CONFIG2 | 0xC0, or 0xD0 for the test signal | input source |
| CONFIG3 | 0xE0 (internal reference buffer on), or 0xEC with bias | `SYNAPSYS_ADS_BIAS` |
| CHnSET (used) | gain code << 4, plus SRB2 (0x08) and mux (0x05 test) → 0x60 / 0x65 at gain 24 | `SYNAPSYS_ADS_GAIN`, `SYNAPSYS_ADS_SRB2` |
| CHnSET (unused) | 0x81 | – |
| BIAS_SENSP/N | used-channel mask when bias is on, else 0 | `SYNAPSYS_ADS_BIAS` |
| MISC1 | 0x20 (SRB1 on) or 0x00 | `SYNAPSYS_ADS_SRB1` |
| LOFF, LOFF_SENSP/N, CONFIG4 | 0x00 | – |

Data path:

- Each DRDY edge triggers a read of 3 + 3·N bytes.
- The 24-bit two's-complement value is converted to double volts with
  LSB = 2·VREF / GAIN / 2^24 (VREF 4.5 V, gain 24 → 22.35 nV).
- `Decoder::push_f64` removes the first-sample DC offset in double before rounding to
  float32.
- A status word whose top nibble is not `0xC` is counted as `bad_status`. The sample is
  still forwarded, which keeps the 250 Hz timeline intact.

## Kconfig options (`idf.py menuconfig → SynapSys BCI`)

| Option | Default | Meaning |
|---|---|---|
| `SYNAPSYS_INPUT_*` | REPLAY | input source (see above) |
| `SYNAPSYS_REPLAY_REALTIME` | y | 250 Hz, paced by the FreeRTOS tick (`xTaskDelayUntil`). If off, the replay pushes samples as fast as possible (blocking) |
| `SYNAPSYS_REPLAY_LOOP` / `_PAUSE_MS` | y / 3000 | repeat passes |
| `SYNAPSYS_SPI_SCLK_KHZ` | 2000 | SPI clock |
| `SYNAPSYS_PIN_*` | see wiring | GPIOs (`-1` allowed for START/RESET/PWDN) |
| `SYNAPSYS_ADS_GAIN` | 24 | PGA gain (1, 2, 4, 6, 8, 12, 24; validated at runtime) |
| `SYNAPSYS_ADS_VREF_MV` | 4500 | VREF for scaling |
| `SYNAPSYS_CH_O1/OZ/O2/POZ` | 1/2/3/4 | channel mapping (must be distinct and exist on the device) |
| `SYNAPSYS_ADS_SRB1` | y | common reference on SRB1 (MISC1) |
| `SYNAPSYS_ADS_SRB2` | n | SRB2 on the used channels |
| `SYNAPSYS_ADS_BIAS` | n | bias drive |
| `SYNAPSYS_DRDY_TIMEOUT_MS` | 100 | counted as `drdy_timeouts` |
| `SYNAPSYS_SAMPLE_QUEUE_LEN` | 1024 | sample queue length (min 500 = 2 s). Absorbs the CCA bursts |
| `SYNAPSYS_EVENT_QUEUE_LEN` | 32 | decoder → output events |
| `SYNAPSYS_ACQ/DECODER/OUTPUT_TASK_PRIO` | 20 / 10 / 5 | task priorities (acquisition on core 1; decoder and output on core 0) |
| `SYNAPSYS_HOP_SAMPLES` | 125 | overrides `config_synapsys_v1().hop_samples` |
| `SYNAPSYS_CONF_THRESHOLD_EN` + `SYNAPSYS_CONF_THRESHOLD` | n, "0.30" | confidence threshold T, a decimal string parsed with `strtof` (range [0, 1]). An invalid value logs an error and leaves the threshold disabled. **Keep off until calibrated on own-system data.** |
| `SYNAPSYS_MARGIN_THRESHOLD_EN` + `SYNAPSYS_MARGIN_THRESHOLD` | n, "0.05" | margin threshold M, same parsing and caveat |
| `SYNAPSYS_REARM_CONFIRM` / `_REFRACTORY` / `_RELEASE` | 2 / 625 / 2 | re-arm K, R (samples), N |
| `SYNAPSYS_PRINT_DECISIONS` | y | JSON line per decision point |
| `SYNAPSYS_RAW_STREAM` / `_RAW_QUEUE_LEN` | n / 256 | print every raw sample (calibration recordings, see below). ~11 kB/s: native USB console or UART >= 921600 baud |
| `SYNAPSYS_HEALTH_PERIOD_MS` | 5000 | health line period |
| `SYNAPSYS_TIMING_EVERY` | 20 | timing min/avg/max every N decisions |
| `SYNAPSYS_LED_GPIO` | -1 | plain LED. It toggles at each decision point and stays on for 300 ms after a command. (The DevKitC RGB LED on GPIO 48 is addressable and not supported.) |
| `SYNAPSYS_BLE_ENABLE` / `_NAME` / `_DEBUG` | y / "SynapSys" / y | BLE output |

## Serial output (one line per event, prefix `SSVEP `)

Lines without the `SSVEP ` prefix are ordinary ESP-IDF logs.

```
SSVEP {"t":750,"status":"OK","scores":[0.446943,0.111000,0.123340],"winner":"YES","margin":0.323603,"decision":"YES","state":"CANDIDATE","ms":...}  (illustrative values)
SSVEP {"event":"command","label":"YES","t":750}
SSVEP {"event":"timing","n":20,"min_ms":..,"avg_ms":..,"max_ms":..}
SSVEP {"event":"health","uptime_s":..,"source":"ads1299","source_state":"running","fs":250.00,"produced":..,"dropped":0,"drdy_timeouts":0,"missed_drdy":0,"spi_errors":0,"bad_status":0,"q_hw":..,"q_len":1024,"decisions":..,"degenerate":0,"commands":..,"events_dropped":0,"dec_ms":{"min":..,"avg":..,"max":..},"filter_us":..,"heap_free":..,"heap_min":..,"stack_free":{"source":..,"decoder":..,"output":..},"ble":"connected"}
```

Field notes:

- `t` is the number of samples since the stream started, which is also the window end.
  In replay mode, `t` restarts for each fixture, and the line also carries `stream` and
  `pass`.
- `ms` is the decoder compute time for that decision point (filter + 3× CCA), measured
  with `esp_timer_get_time`.
- `stack_free` is the FreeRTOS high-water mark in bytes.
- Degenerate windows print `"status":"DEGENERATE","winner":"NONE","decision":"REJECT"`.

### Raw-sample stream (`SYNAPSYS_RAW_STREAM`, off by default)

For own-system calibration recordings. Decisions and commands are unchanged; the extra
lines are:

```
SSVEP {"event":"raw_info","fs":250,"units":"uV","channels":["O1","Oz","O2","POz"],"source":"ads1299","stream":"live"}
RAW 0 -12.267 -1.656 17.407 17.664
RAW 1 ...
SSVEP {"event":"raw_gap","stream":"live","lost":3,"next_seq":4711}   (only if samples were lost)
```

- `raw_info` is printed once per stream start (live: the first sample; replay: every
  fixture, `stream` = fixture id), before that stream's `RAW` lines.
- `RAW <seq> <O1> <Oz> <O2> <POz>`: `seq` counts samples from the stream start (0-based,
  contiguous unless a `raw_gap` line says otherwise); values are microvolts with 3
  decimals, **before** DC removal and filtering (exactly what the decoder is fed).
- Samples lost because the raw queue was full are reported by `raw_gap` (before the next
  `RAW` line) and counted as `raw_dropped` in the health line. A recording with a gap
  should be discarded, not stitched.
- Verified in QEMU: 6 fixtures x 1500 samples, contiguous `seq`, no gaps, values equal the
  fixture inputs to within the 0.0005 uV print rounding; replay still 48/48 PASS.

## BLE (NimBLE peripheral)

The board advertises as `SynapSys`, with flags and the name in the advertising data and the
128-bit service UUID in the scan response. It accepts one connection and requires no
pairing.

| | UUID | Properties | Content |
|---|---|---|---|
| Service | `a5f30001-2c7e-4b8a-9d61-53594e415053` | | SynapSys BCI |
| Command | `a5f30002-2c7e-4b8a-9d61-53594e415053` | read, notify | UTF-8 `YES` / `NO` / `HELP`, one notify per emitted (re-armed) command. Reads return the last one. |
| Debug | `a5f30003-2c7e-4b8a-9d61-53594e415053` | notify | e.g. `0.45,0.11,0.12 m0.32 YES` per decision point (truncated to 20 bytes at the default 23-byte MTU) |
| Status | `a5f30004-2c7e-4b8a-9d61-53594e415053` | read | `fw=0.1.0 src=ads1299 fs=250.00 up=123s drop=0 dec=240 cmd=3` (refreshed with each health line) |

Quick test: in nRF Connect, connect to `SynapSys`, open the unknown service, and enable
notifications on `…0002`. Each spoken-word command arrives as text.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `no ADS1299 found: ID register = 0x00/0xFF` | No power on AVDD/DVDD, PWDN or RESET low, MISO/MOSI swapped, wrong SPI mode, or CS not connected |
| `register read-back mismatch` | Marginal SPI wiring: lower `SYNAPSYS_SPI_SCLK_KHZ` (e.g. 500) |
| `drdy_timeouts` rising, `fs` 0 | START not high (or START pin tied low with `PIN_START` ≠ -1), DRDY on the wrong GPIO, or the device is not in RDATAC |
| `missed_drdy` > 0 | Acquisition task starved: check task priorities, BLE load and flash writes |
| `dropped` > 0, `q_hw` near `q_len` | Decoder too slow or blocked: check `dec_ms.max` and the console bandwidth |
| `bad_status` > 0 | SPI framing errors: check the clock and the ground |
| Replay `FAIL` | Compare `max_score_err` with the host equivalence report (`firmware/reports/`), and confirm hop = 125 and that the thresholds are off |
| Nothing on the console | Wrong USB connector: use the native "USB" port, or switch the console to UART0 |
| BLE not visible | `SYNAPSYS_BLE_ENABLE` off, or the phone is already connected (only one connection is allowed) |

## Files

`main/app_main.cpp` handles boot info, configuration and wiring. The other files are:

| File | Contents |
|---|---|
| `ads1299.*` | driver |
| `acquisition.*` | DRDY ISR, acquisition task and counters |
| `replay.*` | fixture source and self-check |
| `decoder_task.*` | static Decoder and Rearm, plus timing |
| `serial_output.*` | output task: JSON lines, BLE forwarding, health, LED |
| `ble_output.{h,c}` | NimBLE GATT server. Written in C, like the ESP-IDF examples, because the GATT tables use C compound literals. |
| `app_types.hpp` | shared structs |
| `generated/fixtures_embedded.hpp` | generated by `firmware/tools/embed_fixtures.py` |

## Verify without a board (QEMU)

`firmware/tools/run_qemu.sh` builds the firmware with BLE off and a UART console
(`sdkconfig.qemu.defaults`), runs it in Espressif's QEMU ESP32-S3 machine, and exits 0 only
if the fixture-replay self-check passes. Verified 2026-10-19: real-time replay at 250.7 Hz,
48/48 decision points PASS against the Python reference, commands YES/NO/HELP emitted,
0 dropped samples. QEMU timings are not representative of hardware speed.

Note: until the confidence/margin thresholds are calibrated, every window produces a label,
so pure noise also produces a command (e.g. `syn_noise_only` -> NO). That is expected.
