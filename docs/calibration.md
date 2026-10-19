# Own-system recording, calibration and held-out evaluation (M11)

This is the path from a working board to an honest accuracy number. Every step is tooled
and tested; none of it can be run until the board records real EEG.

```
board (RAW stream on) --record_serial.py--> CSV + sidecar (label, split)
   calibration split --calibrate_thresholds.py--> T, M (+ apply.txt for YAML and sdkconfig)
   test split        --evaluate_heldout.py------> the one reported number
   any capture       --verify_device_log.py-----> device decisions == Python on those samples
```

## 1. Flash for recording

`idf.py menuconfig` → SynapSys:
- Input source: **ADS1299 live**.
- Output: **Raw-sample stream** on (`CONFIG_SYNAPSYS_RAW_STREAM`). It sends about 11 kB/s,
  so use the ESP32-S3's native USB port; a UART bridge needs at least 921600 baud.
- Thresholds: leave them **off** while recording calibration data.

Each sample becomes a line `RAW <seq> <O1> <Oz> <O2> <POz>` in µV, before any filtering.
The device keeps decoding as usual, so its decision lines are captured as well.

## 2. Record

```bash
.venv/bin/pip install -e ".[record]"        # pyserial
.venv/bin/python scripts/record_serial.py --port /dev/cu.usbmodem* --seconds 20 \
    --label YES --subject S01 --split calibration --out data/synapsys/S01/cal/yes_01.csv
```

- **Labels:** `YES`, `NO`, `HELP` (the participant fixates that target for the whole
  recording) or `REST` (no target is fixated: blank screen, looking around, blinking,
  talking).
- **Length:** make each recording at least 20 s. The first 3 s are discarded while the
  causal filter settles, then each 2.5 s window gives one decision point every 0.5 s.
- **Gaps:** the recording is sample-count based and must be gap-free. A missing `seq` or a
  `raw_gap` event aborts it; record that trial again.
- **What is saved:** the CSV and JSON sidecar, plus `<name>.serial.log` with the raw device
  lines.
- **Split:** decide it **before** recording, by session or block, never by window:
  - `calibration`: used to choose T and M;
  - `test`: used once, at the end;
  - `pilot`: anything exploratory.

Suggested amount, per participant (adjust to the team's protocol in validation.md §6):

| Split | Recordings |
|---|---|
| Calibration | ~10 per target + 5 minutes of REST spread over several recordings |
| Test (another session) | ~10 per target + 5 minutes of REST |

## 3. Choose T and M (calibration split only)

```bash
.venv/bin/python scripts/calibrate_thresholds.py data/synapsys/S01/cal \
    --max-rest-commands-per-min 0.5 [--max-wrong-fraction 0.1]
```

**The team sets the limit.** `--max-rest-commands-per-min` is how many false commands per
minute of not looking are tolerable. It is a product decision, and the script refuses to
run without it.

**What the script does:**
1. Decodes every recording on the device schedule. Its scores are what the firmware would
   print for the same samples; `tests/test_streaming.py` checks that against all fixtures.
2. Sweeps T over 0–1 and M over 0–0.3, both in steps of 0.01.
3. Runs the re-arm state machine (K=2, R=625, N=2 unless `--rearm` says otherwise).
4. Applies a rule fixed in advance:
   - among the (T, M) pairs within the REST limit (and within the wrong-command limit, if
     one is given), take the highest fraction of target recordings whose **first command is
     correct**;
   - break ties by fewer wrong commands, then fewer REST commands, then the larger T, then
     the larger M.

**What it refuses:**
- any recording marked `split: test`;
- a set without REST recordings.

**Outputs:** `results/<run_id>/`, containing:
- `sweep.csv`;
- `recordings.csv` (gate off vs the chosen pair, per recording);
- `threshold_map.png`;
- `selection.json`;
- `apply.txt`, with the YAML `decision:` block and the `CONFIG_SYNAPSYS_CONF_THRESHOLD*` /
  `CONFIG_SYNAPSYS_MARGIN_THRESHOLD*` lines.

**Then freeze the result:**
- copy the values into `configs/synapsys.yaml` and the firmware config;
- commit them, with the run directory kept as evidence;
- record the new version in `docs/reference_v1.md` (this is a v1.1 settings change; the
  algorithm stays the same).

## 4. Held-out evaluation (once)

```bash
.venv/bin/python scripts/evaluate_heldout.py data/synapsys/S01/test
```

**Inputs it accepts:** only recordings marked `split: test`. It reads T and M from the
config and has no command-line threshold options, so nothing can be tuned at this stage.

**What it reports:**
- **Accuracy over all target trials.** A trial with no command counts as wrong. This is the
  strict denominator for the 85 % requirement.
- Accuracy over trials that produced a command.
- The no-command rate.
- REST false commands per minute.
- The config hash, saved in `held_out.json`.

If the result is not good enough, the next attempt needs **new** test recordings.

## 5. Device vs Python on real EEG (Phase 6 evidence)

```bash
.venv/bin/python scripts/verify_device_log.py capture.serial.log
```

The log must start at a device stream start, i.e. contain the `raw_info` line: start
capturing, then reset the board. The script then decodes the RAW samples in Python and
checks every device decision:
- scores and margin agree within 1e-3;
- the winner is the same;
- the decision is the same (pass `--confidence-threshold` / `--margin-threshold` if the
  board has thresholds enabled).

In QEMU replay (2026-10-19): 6 streams, 48/48 points, max score error 6.1e-5
(`firmware/reports/raw_stream_qemu.md`).

## Not covered here

- **Latency.** Stimulus change to BLE notification is measured in Phase 6 with the stimulus
  sync signal. These recordings have no cue timestamps.
- **Choosing K, R and N.** The script reports results for the given values; change them
  only with a team decision (PLAN.md §8.1).
