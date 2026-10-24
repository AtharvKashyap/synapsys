# Board bring-up: from a new ESP32-S3 to a calibrated BCI

Work through the stages in order. Each stage ends with a `board_check.py` command that
prints PASS or FAIL; only move on after a PASS. Save the log of every PASS (put it under
`firmware/reports/board/` and commit it). That log is the Phase 6 evidence.

| Stage | Needs | Proves |
|---|---|---|
| 1 Replay | bare ESP32-S3 + USB cable | the chip runs the decoder correctly at 250 Hz, and its real speed |
| 1b Real brain data | bare board + USB cable | the chip decodes the Wang benchmark EEG exactly like Python (the 96 % test, on the chip) |
| 2 BLE | + phone | commands reach a phone |
| 3 Soak (replay) | bare board, 30 min | no memory leak, no drops, stable timing |
| 4 ADS1299 test signal | + ADS1299 board, wired | SPI, DRDY, 250 Hz and the scaling to volts |
| 5 Live, electrodes on a person | + electrodes | real EEG: alpha, then SSVEP |
| 6 Calibration + held-out test | + recording sessions | T/M and the accuracy number (docs/calibration.md) |

## 0. Hardware to get

- **An ESP32-S3 board** with at least 4 MB of flash, e.g. **ESP32-S3-DevKitC-1** (N8R8 or
  N16R8 is fine).
  - It must be an **S3**. The plain ESP32, the C3 and the S2 will not run this firmware.
  - The chip or module name printed on the board should say "ESP32-S3".
- **A USB data cable.** Many cables are charge-only, which means no serial port appears.
  If nothing shows up when you plug in, try another cable.
- Later: the ADS1299 board, jumper wires, and electrodes with gel or paste.

The DevKitC-1 has **two USB connectors**:
- **"USB"**: the S3's native USB. **Use this one.** Flashing and the console both go
  through it.
- **"UART"** (on many N8R8 clone boards: **"COM"**): a USB-serial bridge chip (CP210x or
  CH34x). Also works: add `sdkconfig.uart.defaults` to the build (see "Using the COM port"
  below). On the Mac it shows up as `/dev/cu.usbserial-*`, `/dev/cu.wchusbserial*` or
  `/dev/cu.SLAB_USBtoUART` instead of `/dev/cu.usbmodem*`.

**Using the COM port.** The default firmware prints its console on the native USB port, so
over COM you would see only the boot ROM. Build with the extra overlay so the console (and
host feed) use the COM port, at 115200 baud:
```bash
idf.py -B build_uart -D SDKCONFIG=build_uart/sdkconfig \
    -D SDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.host.defaults;sdkconfig.uart.defaults" build
idf.py -B build_uart -p /dev/cu.usbserial-XXXX flash monitor
```
(Drop `sdkconfig.host.defaults` and use another build dir, e.g. `build_uart_replay`, for the
stage 1 replay self-test over COM.) Everything else in this guide is the same, with your
`cu.usbserial-*` port in place of `cu.usbmodem*`. Over COM, the 54-trial benchmark takes
about 12-15 min (115200 baud) instead of a few minutes. This is the console path the QEMU
runs use.

**Power:** there is no on switch. Plugging in the USB cable powers the board and starts it.
- **RST** (also labelled EN) reboots it.
- **BOOT** forces download mode: hold BOOT, press and release RST, then release BOOT.
  You only need this if flashing fails to connect.

## 1. Mac setup (once per terminal)

```bash
cd "~/Desktop/Claude projects/Synapsys/synapsys-decoder"    # the repo
. ~/esp/esp-idf/export.sh        # puts idf.py, esptool.py and pyserial on PATH
idf.py --version                 # expect v5.3.2
```

Nothing has to be copied or sent to the board by hand. `idf.py` builds the firmware on
the Mac and writes it to the board over the USB cable.

To flash the board from **another computer** without ESP-IDF: build here, create one image
with
`cd firmware/esp32/build && esptool.py --chip esp32s3 merge_bin -o synapsys.bin @flash_args`,
then copy `synapsys.bin` over. On that computer, run `pip install esptool` and then
`esptool.py --chip esp32s3 write_flash 0x0 synapsys.bin`.

## Stage 1: replay self-test (bare board, nothing else connected)

1. Plug the board into the **USB** connector and find its port:
   ```bash
   ls /dev/cu.usbmodem*          # e.g. /dev/cu.usbmodem1101; appears when plugged in
   ```
   If nothing appears, try another cable, the other connector, or the BOOT/RST sequence.
2. Build, flash and watch:
   ```bash
   cd firmware/esp32
   idf.py build                                   # ~1-3 min the first time
   idf.py -p /dev/cu.usbmodem1101 flash monitor   # flashes (~20 s), then shows the console
   ```
   Exit the monitor with **Ctrl+]**.
3. What you should see:
   - Boot lines `SynapSys BCI 0.1.0 ...`, `input source: replay` and
     `gating: none (thresholds uncalibrated)`.
   - A stream of `SSVEP {"check":"PASS",...}` lines.
   - After about 40 s, `SSVEP {"event":"replay_summary","pass":0,"result":"PASS","passed":48,...}`
     and `==== REPLAY PASS 0: PASS (48/48 decision points match Python) ====`.
   - Every 5 s, a `health` line with `"fs":250.0` and `"dropped":0`.
   - Every 20 decisions, a `timing` line. This is the **first real measurement of the
     decoder's speed on the chip**.
4. Record the evidence (close the monitor first, because only one program can hold the port):
   ```bash
   cd ../..                                     # repo root
   python firmware/tools/board_check.py --port /dev/cu.usbmodem1101 --stage replay \
       --seconds 90 --log firmware/reports/board/stage1_replay.log
   # press RST on the board right after starting it
   ```
   **PASS** = 48/48 on every pass, fs about 250, no drops, and decision compute under 50 ms.

## Stage 1b: real brain data through the chip (no ADS1299 needed)

This repeats the Python benchmark test (Wang 2017, S1-S3, 54 trials), with the processing on
the ESP32-S3:
- The Mac reads each recorded trial and streams the raw samples over the USB cable.
- The board filters and decodes them on its own CPU and prints its scores.
- The script compares every result with Python on the same samples.

Only the source of the samples differs from live use (USB instead of the ADS1299); the
decoder, the queues and the tasks are the production code.

1. Make sure the benchmark data is on the Mac: `ls data/benchmark/S1.mat`. If it is
   missing, run `.venv/bin/python scripts/fetch_benchmark.py 1 2 3`.
2. Build and flash the host-feed firmware. It uses its own build folder, so the normal
   build is untouched:
   ```bash
   cd firmware/esp32
   idf.py -B build_host -D SDKCONFIG=build_host/sdkconfig \
       -D SDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.host.defaults" build
   idf.py -B build_host -p /dev/cu.usbmodem1101 flash
   ```
   The equivalent in menuconfig is `Input source → Host feed over USB` with
   `Decoder config: Wang benchmark` on.
3. Run the comparison (no monitor open; it needs the port):
   ```bash
   cd ../.. && .venv/bin/pip install -e ".[record]"     # once (pyserial)
   .venv/bin/python scripts/esp32_benchmark.py --port /dev/cu.usbmodem1101 --subjects 1 2 3
   ```
   It prints one line per trial (true target, Python's answer, the board's answer, score
   difference), then a summary like:
   ```
   accuracy  python 96.3%  esp32 96.3%  (54 trials, chance 33.3%)
   same winner on 54/54 trials; max score error ~1e-5 (benchmark window), ... (all 432 hop points, 0 failed)
   on-chip decision time: avg .. ms, max .. ms (filter + 3 CCA)
   PASS; results results/<run_id>_esp32_benchmark
   ```
4. **PASS** =
   - the same winner as Python on every trial;
   - every score within 1e-3 of Python's;
   - all hop points present.

   The ESP32 accuracy must then equal Python's 96.3 %. The `decision time` line is the
   real on-chip speed. Commit `results/<run_id>_esp32_benchmark/` (or copy it into
   `firmware/reports/board/`) as evidence.

Verified in QEMU (2026-10-23, emulated ESP32-S3, same firmware and script): see
`firmware/reports/esp32_benchmark_qemu.md`.

## Stage 2: BLE to a phone

1. Install **nRF Connect** (Nordic, free; iOS or Android).
2. Scan and connect to **SynapSys**. Only one connection is allowed at a time.
3. Open the service `a5f30001-…`.
4. On `a5f30002-…` (Command), tap the subscribe / notify icon.
5. While replay runs you should receive `YES`, `NO`, `HELP` and `NO` texts, about one per
   replay fixture.
6. Read `a5f30004-…` (Status): `fw=0.1.0 src=replay fs=250.00 ...`.
7. Take a screenshot as evidence. The console health line now says `"ble":"connected"`.

**PASS** = the commands arrive on the phone while the console shows the matching
`"event":"command"` lines.

## Stage 3: 30-minute soak (still bare)

Replay loops forever by default.

```bash
python firmware/tools/board_check.py --port /dev/cu.usbmodem1101 --stage soak \
    --seconds 1800 --log firmware/reports/board/stage3_soak.log
```

Run it once with the phone connected (stage 2) and once without.

**PASS** =
- every pass 48/48;
- `dropped`, `events_dropped` = 0;
- `heap_min` unchanged;
- timing stable.

## Stage 4: ADS1299 internal test signal

**Before wiring, confirm with the PCB designer** (these are the defaults; any of them can
be wrong):
- the ADS1299 variant;
- which ADS channels carry O1, Oz, O2 and POz (default CH1-4);
- the reference: SRB1 on (default) or SRB2;
- bias on or off;
- gain 24 and VREF 4.5 V;
- which pins connect START, RESET and PWDN (use `-1` for "tied on the PCB").

**Wiring.** **Unplug USB while wiring.** Defaults (change them in menuconfig if your
board differs):

| ADS1299 | ESP32-S3 GPIO |
|---|---|
| SCLK | 12 |
| DIN (MOSI) | 11 |
| DOUT (MISO) | 13 |
| CS | 10 |
| DRDY | 9 |
| START | 8 |
| RESET | 7 |
| PWDN | 6 |
| DGND | GND |
| DVDD | 3V3 |

The ADS1299's analog supply (AVDD/AVSS) comes from its own board. Check that the grounds
are shared.

**Configure:**

```bash
cd firmware/esp32 && idf.py menuconfig
```

In menuconfig, use the arrow keys, Enter to open a menu, Space to toggle, then **S** to
save and **Q** to quit.
- `SynapSys BCI → Input source → ADS1299 internal test signal`
- `SynapSys BCI → ADS1299`: set the pins, gain, VREF, channel map, SRB1/SRB2 and bias to
  match the PCB.
- `SynapSys BCI → Output → Raw-sample stream`: **on**. The checker uses it to measure the
  square wave.

**Flash and check:**

```bash
idf.py -p /dev/cu.usbmodem1101 flash monitor      # look, then Ctrl+]
cd ../.. && python firmware/tools/board_check.py --port /dev/cu.usbmodem1101 \
    --stage ads_test --seconds 60 --log firmware/reports/board/stage4_ads_test.log
```

**What to look for in the boot log:**
- `ID = 0x3E -> ADS1299-8` (or -4/-6);
- a register table with every row `ok`;
- `streaming: ... LSB 0.02235 uV`.

**PASS** = the ID is read, the read-back is ok, fs is 245-255, there are no DRDY
timeouts, SPI errors or bad status, and every channel shows a ~1 Hz square wave of a few mV.
Decoder labels are meaningless in this mode; ignore them.

**If it fails:** the troubleshooting table in `firmware/esp32/README.md` covers these:
- ID 0x00/0xFF: power, MISO/MOSI swapped, CS, or PWDN/RESET low;
- read-back mismatch: lower the SPI clock to 500 kHz;
- `drdy_timeouts`: START or the DRDY pin.

## Stage 5: live EEG on a person

> **Safety:** while electrodes are on a person, run the laptop **on battery with the
> charger unplugged** (or use a USB isolator). Never connect a mains-powered device to
> someone through the electrodes.

1. In menuconfig, set `Input source → ADS1299 live`. Keep the raw stream on and the
   thresholds **off**.
2. Flash, then run the live check:
   ```bash
   python firmware/tools/board_check.py --port ... --stage live --seconds 60 --log .../stage5_live.log
   ```
3. **Alpha test** (proves it is real EEG):
   - record 30 s eyes closed and 30 s eyes open;
   - analyse both;
   - closed should show a clear 8-12 Hz peak at O1/Oz/O2 that open does not.
   ```bash
   .venv/bin/pip install -e ".[record]"
   .venv/bin/python scripts/record_serial.py --port ... --seconds 30 --label REST \
       --split pilot --note "eyes closed" --out data/synapsys/pilot/alpha_closed.csv
   .venv/bin/python scripts/record_serial.py --port ... --seconds 30 --label REST \
       --split pilot --note "eyes open"   --out data/synapsys/pilot/alpha_open.csv
   .venv/bin/python scripts/analyze_recording.py data/synapsys/pilot/alpha_closed.csv
   .venv/bin/python scripts/analyze_recording.py data/synapsys/pilot/alpha_open.csv
   ```
4. **One target at a time** (the stimulus app shows only one flicker):
   - record 20 s per target with `--label YES/NO/HELP --split pilot`;
   - in `analyze_recording.py`, the fixated target's score should lead.
5. **Device == Python on real EEG** (Phase 6 gate evidence):
   - start the recording;
   - press RST so the capture contains the stream start;
   - then verify:
   ```bash
   .venv/bin/python scripts/verify_device_log.py data/synapsys/pilot/<name>.serial.log
   ```
6. **Latency:** this needs the stimulus sync signal, so plan it with the stimulus repo.

## Stage 6: calibration and the held-out test

Follow `docs/calibration.md`. The team first decides the tolerated false commands per
minute. Then:
1. record the calibration session;
2. run `calibrate_thresholds.py`;
3. freeze T/M into the config and the firmware;
4. record the test session;
5. run `evaluate_heldout.py` **once**.

## Quick reference

| Task | Command |
|---|---|
| set up a terminal | `. ~/esp/esp-idf/export.sh` |
| flash and watch | `cd firmware/esp32 && idf.py -p PORT flash monitor` (Ctrl+] exits) |
| watch only | `idf.py -p PORT monitor` |
| change settings | `idf.py menuconfig`, then build and flash |
| start from the defaults again | `rm sdkconfig && idf.py build` |
| check a stage | `python firmware/tools/board_check.py --port PORT --stage replay\|ads_test\|live\|soak` |
| re-check a saved log | `python firmware/tools/board_check.py --from-log LOG --stage ...` |
| board won't flash | hold BOOT, tap RST, release BOOT, flash again |
