# firmware/: SynapSys decoder on the ESP32-S3 (C++17 / ESP-IDF v5.3)

The C++ port of the frozen Python reference (`../docs/reference_v1.md`, tag `python-ref-v1`).
It is accepted only when it matches `../fixtures/` decision point by decision point.

| Where | What |
|---|---|
| [`PLAN.md`](PLAN.md) | phased port plan, numeric acceptance gates, open team decisions |
| [`BRINGUP.md`](BRINGUP.md) | step-by-step board bring-up, each stage ending in a PASS/FAIL check |
| [`decoder/`](decoder) | platform-independent C++17 core (filter, CCA, decision, re-arm). No ESP-IDF, no heap in the hot path. Host-built with CMake + ctest against the fixtures |
| [`esp32/`](esp32) | ESP-IDF application: ADS1299 SPI driver, acquisition task, decoder task, serial + BLE output, fixture replay and host-feed modes. See [`esp32/README.md`](esp32/README.md) |
| [`tools/`](tools) | config/SOS → generated C++ header, fixture flattening/embedding, QEMU runner, `board_check.py` |
| [`reports/`](reports) | saved evidence: host equivalence (double/float), QEMU runs, real-board benchmark |

## Status

| Phase | Result |
|---|---|
| 0–4 host C++ core | matches Python on every fixture: double score error 1.3e-15, float 9e-5; re-arm identical on 140 sequences |
| 5 on device | QEMU ESP32-S3: real-time 250 Hz replay 48/48 PASS. Real ESP32-S3-N8R8: Wang S1–S3 decoded on-chip, 96.3 % = Python, 54/54 same winner, 4.5 ms per decision ([report](reports/esp32_benchmark_board.md)) |
| 6 ADS1299 + BLE | code written and builds; needs the ADS1299 board, a phone and a 30 min soak |

## Quick commands (from the repo root)

```bash
# C++ core vs fixtures on your computer (no board)
cmake -S firmware/decoder -B firmware/decoder/build -G Ninja
cmake --build firmware/decoder/build && ctest --test-dir firmware/decoder/build

# Real firmware in Espressif QEMU (no board; exit 0 = PASS)
firmware/tools/run_qemu.sh

# Build + flash a board (ESP-IDF v5.3.2)
. ~/esp/esp-idf/export.sh && cd firmware/esp32 && idf.py build flash monitor
```
