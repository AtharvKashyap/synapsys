# SynapSys live demo

Recorded brain data on the laptop → ESP32-S3 decodes it → iPhone says the answer.

```
Laptop (pygame window)                ESP32-S3 (demo firmware)                 iPhone (SynapSys app)
5 flickering tiles, one highlighted   filter + CCA + gate + re-arm, on-chip    green YES + "Yes"
recorded EEG of a volunteer    ──COM──►  every 0.5 s                   ──BLE──►  red NO + "No"
looking at that tile, real time       (same C++ as the frozen reference)       orange HELP + alert + "Help"
"Expected: YES / quiet" + board echo                                           quiet tiles: nothing
```

Five scenarios. The keys 1–5 match the tiles from left to right; Space plays them in story order.

| Tile (key) | Frequency | Recording played | Phone |
|---|---|---|---|
| YES (1) | 8 Hz | volunteer looked at 8 Hz | green, says "Yes" |
| — (2) | 10 Hz | volunteer looked at 10 Hz | stays quiet |
| NO (3) | 12 Hz | volunteer looked at 12 Hz | red, says "No" |
| — (4) | 14 Hz | volunteer looked at 14 Hz | stays quiet |
| HELP (5) | 15 Hz | volunteer looked at 15 Hz | orange flash, alert, says "Help" |

The data is the public Wang et al. 2017 SSVEP benchmark (S1–S3), played exactly as it was
recorded: 0.5 s before the flicker, 5 s of flicker, 0.5 s after. The board answers about
**3 s** after a scenario starts.

Everything for the demo lives in this folder. The real decoder, firmware and configs are
only read, never changed: the firmware is the normal `firmware/esp32` code built with the
settings file `firmware/sdkconfig.demo.defaults`.

## Folder

| Path | What |
|---|---|
| `feeder/app.py` | the laptop stage window |
| `feeder/{board_link,runner,scenarios}.py` | COM link (CRC-checked), real-time pacing, the 5 trials |
| `feeder/fake_board.py` | **development only**: a pretend board in Python (the window shows a red banner) |
| `feeder/tests/` | 22 tests (`.venv/bin/pytest -q live_demo` from the repo root, ~50 s) |
| `firmware/sdkconfig.demo.defaults` | demo firmware settings: COM at 921600, BLE on, Wang 8/12/15 Hz config, gate T = 0.42 |
| `firmware/build_flash.sh` | build (and flash) the demo firmware into `build/esp32` |
| `firmware/qemu_check.sh` | runs the real demo firmware in an emulator and plays all scenarios (no board) |
| `board_up.py` | one-command bring-up: flash + COM + real-time + BLE checks |
| `relay/` | laptop Bluetooth relay (Swift) for a board whose radio can't run; `feeder/relay.py` drives it |
| `analysis/prove_on_board.py` | evidence the board decodes: scores vs Python, timing, made-up signals |
| `firmware/ble_probe.py` | checks BLE from the Mac: plays the 5 scenarios, confirms each arrives over Bluetooth |
| `analysis/quiet_threshold.py` | how the demo gate T = 0.42 was chosen (and its held-out numbers) |
| `ios/` | the iPhone app (Xcode project, SwiftUI + CoreBluetooth); `ios/Tests/run_logic_tests.sh`; `ios/install_on_phone.sh`; `ios/make_icon.py` (app icon) |
| `build/` | generated files only (git-ignored) |

## One-time setup

All commands from `live_demo/` unless stated otherwise.

**1. Laptop Python.** It uses the repo's venv (Python 3.11–3.13). On a fresh machine, from
the repo root:
```bash
python3 -m venv ~/.venvs/synapsys-decoder && ln -s ~/.venvs/synapsys-decoder .venv   # see the main README (iCloud note)
.venv/bin/pip install -e ".[dev,data]" pygame pyserial bleak
.venv/bin/python scripts/fetch_benchmark.py 1 2 3     # Wang S1-S3, ~640 MB -> data/benchmark/
cd live_demo
../.venv/bin/python -m feeder.scenarios               # picks + caches the 5 trials in build/
```

**2. Flash the board.** You need ESP-IDF **v5.3.2** installed at `~/esp/esp-idf` (Espressif's
"get started" guide; the build script sources `~/esp/esp-idf/export.sh`). Use the board's
**COM** connector (the USB-serial chip), not the one
labelled USB. On the Mac it shows up as `/dev/cu.usbmodem…` (CH343), `/dev/cu.wchusbserial…`
or `/dev/cu.usbserial…`.
```bash
ls /dev/cu.*                                        # find the port
firmware/build_flash.sh /dev/cu.usbmodemXXXX        # build + flash (~2 min the first time)
firmware/build_flash.sh /dev/cu.usbmodemXXXX --monitor   # optional: watch it boot, Ctrl-] to quit
```
The boot log should show `decoder config: benchmark_v1`, `confidence threshold T = 0.420` and
`advertising as "SynapSys"`. Close the monitor before starting the stage app: only one
program can use the port at a time.

**3. Install the iPhone app** (free Apple ID is enough; the iPhone needs iOS 17 or later):
1. Open `ios/SynapSysDemo.xcodeproj` in Xcode.
2. Select the SynapSysDemo target, open **Signing & Capabilities**, then set **Team** to your
   Apple ID (Personal Team).
3. If Xcode says the bundle identifier is taken, change it to something of your own, e.g.
   `com.<yourname>.synapsysdemo`. Each teammate needs their own.
4. Plug the iPhone in, pick it as the run destination and press **Run**.
5. First time only, on the iPhone:
   - Settings → Privacy & Security → **Developer Mode** on (the phone restarts).
   - Settings → General → VPN & Device Management → trust your developer certificate.
   - When the app asks for Bluetooth, tap **Allow**.
6. A free-account install **expires after 7 days**. Re-run it from Xcode within a week of the show.
7. Reinstalling later doesn't need the Xcode window: run `ios/install_on_phone.sh` with
   the phone plugged in. It builds outside the iCloud-synced folder, because builds inside
   it fail code signing with "resource fork, Finder information, or similar detritus not
   allowed". Building from Xcode is fine, since Xcode builds outside the folder too.

## Board bring-up in one command

Plug the board's COM connector in (close the iPhone app first: the board accepts one
Bluetooth connection at a time), then:
```bash
../.venv/bin/python board_up.py          # waits for the board, flashes, tests COM, real time, BLE
```
It prints PASS/FAIL per step with the fix for any failure, and ends with the exact command
to start the stage app. `--check` runs only the laptop-side checks; `--no-flash` skips
flashing. The log goes to `build/board_up_<time>.log`.

## Board whose radio can't run: Bluetooth-off build + laptop relay

Some boards brown out (reboot loop) the moment the radio starts. Our YD-ESP32-S3 does this,
and so does Espressif's stock `bleprph` example on it. For those boards:
```bash
NO_BT=1 firmware/build_flash.sh /dev/cu.usbmodemXXXX     # Bluetooth-off build: board stays up
relay/build_relay.sh                                     # once: the Mac Bluetooth relay
../.venv/bin/python -m feeder.app --port /dev/cu.usbmodemXXXX --relay
```
The path becomes ESP32-S3 (decodes) → COM → laptop → Bluetooth → iPhone. The Mac advertises
as "SynapSys" with the board's own Bluetooth service, so the iPhone app works unchanged. The
laptop decides nothing: it forwards the label from the board's own `command` line. The window
only says "iPhone connected" and "sent to iPhone", so **say out loud** that the laptop relays it.
`analysis/prove_on_board.py --port …` produces the evidence that the board does the decoding:
- every decision's CCA scores match the Python reference;
- each answer arrives only after its data was sent;
- made-up signals are decoded correctly.

## Before the show (T−30 min)

1. Plug the board's COM connector into the laptop.
2. **BLE check** (the Mac acts as the phone; close the iPhone app first, since the board
   accepts one Bluetooth connection at a time):
   ```bash
   ../.venv/bin/python firmware/ble_probe.py --feed /dev/cu.usbmodemXXXX
   ```
   Expect five `ok` lines and `PASS`. macOS may ask once to allow Bluetooth for Terminal.
3. **Start the stage app first** (opening the COM port can reset the board, which would drop
   a phone that is already connected):
   ```bash
   ../.venv/bin/python -m feeder.app --port /dev/cu.usbmodemXXXX
   ```
   Top right should show a green dot and "ESP32-S3 on …".
4. **Then the phone:** open the app and wait for "Connected to SynapSys". Use the **Test**
   menu (bottom right) to play YES, NO and HELP locally and set the volume. The app speaks
   even with the silent switch on. Disconnect AirPods: speech goes to whatever audio output
   is connected. After pressing **R** in the stage app, check the phone still says Connected
   (if not, it reconnects within a few seconds).
5. **Show the phone to the room:**
   - Easiest: point a document camera at it.
   - Or mirror it: connect it to the Mac by cable, then open QuickTime Player → File → New
     Movie Recording → click the arrow next to record → choose the iPhone as camera. If you
     also choose it as microphone, the phone's sound plays through the Mac instead of the
     phone, so turn QuickTime's volume slider up (good for a big room). Check this in
     rehearsal.
   - With the mirror on the same screen, keep the stage app windowed (don't press F) and put
     the two windows side by side. Use **F** (full screen) only when the phone is shown some
     other way or on a second display.
6. Do one full run (Space ×5) and check that every scenario shows the tick mark on the
   laptop and the right result on the phone.

## During the show

- **Say this before the first scenario:** "The next screen has flickering squares. If
  flashing lights affect you, please look away." The idle screen shows the same warning.
  Press **X** at any time to stop the flicker. The EEG is a recording, so turning the
  flicker off changes nothing about the result. Start with `--no-flicker` if in doubt.
- A presenter clicker works: its next-slide key (PageDown / →) plays the next scenario.
- Press **Space** (or 1–5). What happens:
  1. The tiles flicker and a white frame marks the tile the volunteer looked at.
  2. The EEG traces scroll while the data streams to the ESP32-S3.
  3. Bottom left shows what to expect; bottom right shows what the board decided (dots fill
     in as decisions arrive every 0.5 s, then "ESP32-S3 sent YES" with a tick mark).
  4. The phone turns green/red/orange and speaks about 3 s after the start.
  5. For the 10 Hz and 14 Hz tiles the phone stays silent and the laptop shows "no command
     (quiet)".
- The phone goes back to "Ready" by itself after 4 s (HELP after 6 s), or when you tap it.

## If something goes wrong

| Problem | Fix |
|---|---|
| Header says "board not answering" | Wrong port, probably the native USB one: plug in the COM connector and press **R**. If the firmware isn't flashed, run step 2 of setup |
| "board lost" / "Board lost — plug it in, press R" | Cable came out: plug it back in, press **R**, check the phone reconnected |
| "Link error — press the key again" | Bytes were lost on the cable. The app resyncs by itself, so just press the key again |
| Red text "link slower than real time" | Close other serial monitors. If it persists, the demo still works, just slower |
| A scenario gives an unexpected result | **Shift+key** switches that tile to its next backup recording (2 backups each, all verified) |
| Phone says "Searching…" | Give it a few seconds. Toggle Bluetooth on the phone, or force-quit and reopen the app. Close the Mac's `ble_probe.py` if it's running |
| Phone shows the result but no sound | Volume up; disconnect Bluetooth headphones. If QuickTime mirrors with the iPhone as microphone, the sound comes out of the Mac: turn QuickTime's volume up |
| No phone at all | The laptop's "ESP32-S3 sent YES" line is the board's own answer; the free LightBlue app on any phone can also show the Command value |

**Never use `--fake` on stage.** It runs the decoder in Python on the laptop, not on the
ESP32, and the window says so in a red banner.

## What has been verified, and what still needs the board

Verified without hardware:
- **Real demo firmware in the QEMU emulator** (`firmware/qemu_check.sh`). All 5 scenarios
  plus both backups each (15 recordings) gave exactly the expected result: YES/NO/HELP
  commands, and quiet for 10 and 14 Hz. BLE is off in the emulator because it has no radio.
- **Laptop app and link:** 22 tests, including:
  - CRC errors and recovery;
  - the same scenario twice;
  - running slower than real time;
  - an unplugged board (shown as an error, never as "quiet");
  - pressing a new key mid-scenario (never freezes the window);
  - a headless run of the real window.
- **iPhone app:** builds for device and Simulator; 32 checks of its display logic;
  screenshots reviewed.
- **Gate T = 0.42:** chosen on Wang blocks 0–2, measured on blocks 3–5:
  - targets correct: 27/27;
  - off-target trials that produce a command: 5.2 % (99.7 % without the gate).

  The demo's recordings are **pre-screened** from blocks 3–5 to give the expected result,
  and to keep at least 0.01 away from the gate.

Still to do on the real board:
- [ ] Flash; check the boot log, the handshake at 921600, and that the app's header says
      250 samples/s (real time; the emulator could only manage about 100).
- [ ] `ble_probe.py --feed` passes (BLE + COM together on the board).
- [ ] iPhone: connects; YES/NO/HELP shown and spoken; silent switch on; lock/unlock, then
      the next command still arrives.
- [ ] Full rehearsal: 5 scenarios × 3, phone right every time, about 3 s each.
