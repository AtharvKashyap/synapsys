# Wang benchmark decoded on the ESP32-S3 (QEMU) vs Python (2026-10-23)

Firmware `firmware/esp32` in host-feed mode (`sdkconfig.host.defaults` + `sdkconfig.qemu.defaults`:
UART console, BLE off), decoder config `benchmark_v1`, running in Espressif QEMU (ESP32-S3).
`scripts/esp32_benchmark.py --port socket://127.0.0.1:5555 --subjects 1 2 3` streamed every
trial's raw samples (exact float64) to the emulated chip, which filtered and decoded them with
the unchanged float32 C++ core; every result was compared with the Python reference.

| Quantity | Result |
|---|---|
| Trials (S1-S3, 8/12/15 Hz, 6 blocks) | 54 |
| Accuracy, Python reference | 96.3% |
| Accuracy, ESP32-S3 | 96.3% |
| Same winner as Python | 54/54 |
| Max score error, benchmark window [160, 785) | 4.4e-05 (gate 1e-3) |
| Max score error, all hop points | 5.3e-05 over 432 points, 0 failed |
| Decision compute (QEMU, not real timing) | avg 3.33 ms, max 5.22 ms |

Misclassified trials (identical on both): S2 block 0 true F15_0 -> F8_0, S2 block 4 true F15_0 -> F12_0

Generic CCA on the public benchmark only; not SynapSys YES/NO/HELP accuracy. QEMU
timings are not hardware timings: repeat on a real board (firmware/BRINGUP.md stage 1b).

## COM-port build (2026-10-24)

`sdkconfig.defaults;sdkconfig.host.defaults;sdkconfig.uart.defaults` (console on the UART bridge
port, 115200 baud; the build for boards plugged in via "COM"/"UART") also run in QEMU:
S1 first 6 trials, same winner 6/6, max score error 3.1e-05 (window), 2.8e-05 (48 hop points,
0 failed), PASS.
