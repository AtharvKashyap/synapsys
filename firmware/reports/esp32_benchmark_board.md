# Wang benchmark decoded on a real ESP32-S3 vs Python (2026-10-24)

Board: ESP32-S3-N8R8 dev board, connected via its COM connector (WCH CH343 USB-serial,
`/dev/cu.usbmodem5B8F0676581`). Firmware `firmware/esp32`, build
`sdkconfig.defaults;sdkconfig.host.defaults;sdkconfig.uart.defaults` (host feed, decoder
config `benchmark_v1`, console on UART0 at 115200, CRC-checked blocks).
Command: `scripts/esp32_benchmark.py --port /dev/cu.usbmodem5B8F0676581 --subjects 1 2 3`.

| Quantity | Result |
|---|---|
| Trials (S1-S3, 8/12/15 Hz) | 54 |
| Accuracy, Python reference | 96.3% |
| Accuracy, ESP32-S3 (real chip) | 96.3% |
| Same winner as Python | 54/54 |
| Max score error, benchmark window | 4.4e-05 (tolerance 1e-3) |
| Max score error, all 432 hop points | 5.3e-05, 0 failed |
| **On-chip decision compute (filter + 3 CCA, 240 MHz)** | **avg 4.46 ms, max 4.72 ms** |
| Link resends (CRC) | 0 |
| Wall time (115200 baud transfer) | 684 s |

Result: **PASS**. Same two misclassified trials as Python (S2). Evidence:
`firmware/reports/board/stage1b_esp32_benchmark_20261024/` (run.json, trials.csv, console.txt).

An earlier run on the same board (before the CRC check and the lowered UART RX FIFO
threshold) stopped after 17 matching trials with "board reported 2 unparsable line(s)":
bytes lost on the UART link, not a decoder error. With the fix: 0 resends over 54 trials.

Generic CCA on the public benchmark only; not SynapSys YES/NO/HELP accuracy.
