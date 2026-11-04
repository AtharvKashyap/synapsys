# synapsys-decoder

Signal processing and classification for **SynapSys**, a 3-command SSVEP brain–computer
interface. The user looks at one of three flickering targets; EEG over the visual cortex
picks up the flicker frequency; the decoder turns that into **YES**, **NO** or **HELP**.

```
ADS1299 (4 EEG channels, 250 Hz) ──SPI──► ESP32-S3 (filter + CCA + decision) ──BLE──► phone
```

This repo contains:

1. **The Python reference decoder** (`src/synapsys_decoder/`). It is the source of truth for
   the algorithm, frozen as v1 (tag `python-ref-v1`, [docs/reference_v1.md](docs/reference_v1.md)).
2. **The C++ port and ESP32-S3 firmware** (`firmware/`). They must reproduce the Python
   outputs on shared fixtures (`fixtures/`), and they do (see Status).
3. **The tools for the hardware phase**: recording own EEG from the board, calibrating the
   gating thresholds, and the one held-out accuracy evaluation.

The stimulus renderer (PsychoPy) is a separate project, `synapsys-stimulus`.

## How the decoder works

```
raw EEG (O1, Oz, O2, POz; 250 Hz; volts)
  -> remove DC
  -> Butterworth band-pass 0.5-45 Hz (4 biquads) + 60 Hz notch, Q = 30 (1 biquad)
  -> keep the most recent 2.5 s (625 samples), re-evaluated every 0.5 s
  -> CCA against sin/cos references of each target (2 harmonics)
  -> 3 scores (largest canonical correlation per target)
  -> winner = argmax; confidence = top score; margin = top - runner-up
  -> optional gating on confidence / margin (thresholds not calibrated yet = off)
  -> YES / NO / HELP / REJECT -> re-arm state machine -> one command per gaze
```

| Command | Frequency | Display (60 Hz) |
|---|---|---|
| YES | 60/7 ≈ 8.571 Hz | 7 frames per cycle |
| NO | 12 Hz | 5 frames per cycle |
| HELP | 15 Hz | 4 frames per cycle |

Standard CCA is **not trained**: nothing is fit to data. The only data-derived values will be
the two gating thresholds, once calibrated on our own recordings. Full maths, filter design
and porting notes: [docs/algorithm.md](docs/algorithm.md).

**Requirements:** ≥ 85 % held-out accuracy on our own system; ≤ 5 s end-to-end latency.

## Status

| # | Milestone | Status |
|---|---|---|
| M1–M6 | Python package, references, CCA, classifier, DSP, synthetic end-to-end | Done |
| M7 | Real EEG (public Wang 2017 benchmark, S1–S3, 8/12/15 Hz) | **96.3 %** causal filter / 98.1 % zero-phase (54 trials, chance 33 %) |
| M8 | Window / harmonic / channel sweeps | Run on S1–S3; baseline choice still to be written up |
| M9 | C++ port of the decoder | Done: matches Python on all fixtures (float score error ≤ 9e-5), re-arm identical on 140 sequences |
| M10 | Python vs ESP32 regression | Done in QEMU (48/48) **and on a real ESP32-S3-N8R8**: Wang S1–S3 decoded on-chip, 96.3 % = Python, 54/54 same winner, **4.5 ms per decision** ([report](firmware/reports/esp32_benchmark_board.md)) |
| M11 | Own hardware: ADS1299, BLE, recordings, calibration, held-out test | Tooling done and tested; waiting on the ADS1299 board |

**Next steps:** wire the ADS1299 ([firmware/BRINGUP.md](firmware/BRINGUP.md) stages 2–5), then
record → calibrate → held-out test ([docs/calibration.md](docs/calibration.md)).

**Team decisions still open** ([firmware/PLAN.md §8](firmware/PLAN.md)): re-arm parameters
(K, R, N); decision hop; whether REJECTs count against the 85 %; the false-activation limit
used for calibration.

## Quick start

Python 3.11–3.13.

```bash
git clone <this repo> && cd synapsys-decoder
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q                     # 272 tests, ~15 s
```

Optional extras: `.[data]` (download the Wang benchmark), `.[record]` (pyserial, for
recording from the board).

> **macOS + iCloud Desktop/Documents:** iCloud re-applies the `hidden` flag to files inside
> dot-folders, and Python 3.13 then skips the editable-install `.pth` file
> (`ModuleNotFoundError: synapsys_decoder`). Keep the venv outside iCloud and symlink it:
> `python3 -m venv ~/.venvs/synapsys-decoder && ln -s ~/.venvs/synapsys-decoder .venv`.

## Common tasks

All commands from the repo root.

| Goal | Command |
|---|---|
| Unit tests / lint | `.venv/bin/pytest -q` · `.venv/bin/ruff format . && .venv/bin/ruff check .` |
| Synthetic validation (M6) | `.venv/bin/python scripts/run_synthetic_validation.py` |
| Download Wang benchmark S1–S3 (~105 MB each) | `.venv/bin/python scripts/fetch_benchmark.py 1 2 3` |
| Real-EEG walkthrough (7 checked stages + viewer page) | `.venv/bin/python scripts/benchmark_walkthrough.py` |
| Benchmark validation + sweeps (M7/M8) | `.venv/bin/python scripts/run_benchmark_validation.py --config configs/benchmark.yaml` |
| Regenerate Python→C++ fixtures | `.venv/bin/python scripts/make_fixtures.py` |
| Prove the tests catch bugs (plants 11 bugs) | `.venv/bin/python scripts/mutation_check.py` |
| All weekly-report evidence | `.venv/bin/python scripts/make_report_artifacts.py --out docs/report` |
| C++ core vs fixtures (no board) | `cmake -S firmware/decoder -B firmware/decoder/build -G Ninja && cmake --build firmware/decoder/build && ctest --test-dir firmware/decoder/build` |
| Real firmware in QEMU (no board) | `firmware/tools/run_qemu.sh` |
| Build / flash the ESP32-S3 (ESP-IDF v5.3.2) | `. ~/esp/esp-idf/export.sh && cd firmware/esp32 && idf.py build flash monitor` |
| Board bring-up checks | `python firmware/tools/board_check.py --port PORT --stage replay\|ads_test\|live\|soak` |
| Wang EEG decoded on the chip vs Python | `.venv/bin/python scripts/esp32_benchmark.py --port PORT --subjects 1 2 3` |
| Record own EEG from the board | `.venv/bin/python scripts/record_serial.py --port PORT --seconds 20 --label YES --split calibration --out data/synapsys/S01/cal/yes_01.csv` |
| Choose gating thresholds (calibration split only) | `.venv/bin/python scripts/calibrate_thresholds.py data/synapsys/S01/cal --max-rest-commands-per-min <team limit>` |
| The one held-out evaluation (test split only) | `.venv/bin/python scripts/evaluate_heldout.py data/synapsys/S01/test` |
| Device decisions == Python on captured samples | `.venv/bin/python scripts/verify_device_log.py capture.serial.log` |
| Inspect one recording (PSD, scores over time) | `.venv/bin/python scripts/analyze_recording.py data/synapsys/rec.csv` |

Every script run writes `results/<run_id>/` (git-ignored) with the config used, `run.json`
(metrics, seed, package versions, git revision), per-trial predictions and figures.

## Repository map

```
configs/
  synapsys.yaml            own-hardware config: targets, filters, gating (null = off)
  benchmark.yaml           Wang 2017 config: 8/12/15 Hz, no 60 Hz notch (50 Hz mains)
src/synapsys_decoder/
  config.py                YAML -> typed config, validation
  types.py                 Trial, ClassificationResult, REJECT
  references.py            sin/cos harmonic reference matrices
  cca.py                   CCA via QR + SVD (Björck–Golub principal angles)
  classifier.py            score targets, winner, confidence/margin gating
  preprocessing.py         DC removal, Butterworth SOS, notch, streaming filter
  pipeline.py              decode_trial: select channels -> filter -> crop -> classify
  streaming.py             the device's decision schedule, offline
  rearm.py                 command re-arming state machine (spec for the C++ port)
  calibration.py           threshold sweep + selection rule, split guards
  synthetic.py             synthetic SSVEP / noise / interference generators
  metrics.py               accuracy, rejection, confusion, ITR, false activations
  plotting.py              figures (never imported by algorithm modules)
  runlog.py                results/<run_id>/ artifacts
  io/                      Wang .mat loader, CSV+JSON trial format, firmware serial logs
scripts/                   runnable entry points (table above)
tests/                     pytest suite, one file per module + script smoke tests
fixtures/                  Python -> C++ regression set (synthetic, committed; see fixtures/README.md)
firmware/
  PLAN.md, BRINGUP.md      port plan with gates; board bring-up guide
  decoder/                 platform-independent C++17 core + host tests
  esp32/                   ESP-IDF app: ADS1299, acquisition, decoder task, BLE, serial
  tools/                   header generators, QEMU runner, board_check.py
  reports/                 saved equivalence / QEMU / board evidence
docs/
  algorithm.md             equations, filter design, gating, porting notes
  validation.md            what has been validated, how, and what remains
  calibration.md           record -> calibrate -> held-out test workflow (M11)
  reference_v1.md          the frozen Python reference and what "changing it" means
  figures/                 synthetic and benchmark figures from committed runs
  report/                  generated weekly-report evidence (see docs/report/INDEX.md)
data/                      raw EEG, git-ignored (see data/README.md)
results/                   run outputs, git-ignored
```

## Rules for contributors

These keep results honest and the firmware in sync. Please don't change them silently.

**Data and code conventions**
- EEG arrays are `(n_channels, n_samples)`, float64, in **volts**. Select channels by name.
- Class order = the order of `targets` in the config.
- Quote YAML target labels (`"YES"`, `"NO"`): YAML 1.1 parses bare YES/NO as booleans.
- Project constants live in `configs/`; never hard-code them in algorithm modules.
- Algorithm modules are pure functions with explicit `fs`/window arguments; no plotting.
- Loaders (`io/`) never filter or resample.
- Synthetic code uses fixed seeds (`numpy.random.default_rng(seed)`).

**The frozen reference**
- `tests/test_frozen_reference.py` pins the settings, filter coefficients and every fixture
  output. Changing decoder behaviour means a deliberate v2: bump the version, regenerate
  fixtures, re-run the C++ equivalence and record the evidence. Never change the algorithm
  and the language at the same time.
- The serial line formats (`SSVEP {json}`, `RAW ...`, host-feed commands) are shared by the
  firmware and `io/serial_log.py` / `scripts/esp32_benchmark.py`: change both together.

**Evaluation**
- Gating thresholds stay `null` until calibrated on our own **calibration** recordings. Never
  tune on the test split, never pick thresholds from synthetic or benchmark data.
- `calibrate_thresholds.py` refuses `test` data; `evaluate_heldout.py` accepts only `test` data.
- The false-activation limit is a team decision, passed on the command line.
- `mode: zero_phase` filtering is offline-only; the ESP32 runs `mode: causal`.

**Data**
- Raw EEG never goes in git (`data/` is ignored). Wang-derived fixtures (`fixtures/wang/`)
  are generated locally because the dataset's redistribution terms are not stated.

## What this repo does NOT claim yet

- **No SynapSys accuracy exists yet.** Nothing has been recorded on SynapSys hardware from a
  person. The 85 % requirement is untested.
- The Wang 2017 benchmark validates the generic CCA pipeline only: different hardware,
  montage, stimulus and mains frequency (50 Hz), and experienced subjects. It has no
  8.571 Hz target, so its trials are never relabelled as YES/NO/HELP.
- Synthetic results show pipeline behaviour (frequency selectivity, noise trends), not human
  performance.

## Headline synthetic results

Run `20261003-231518_synthetic_validation` (seed 20261003, 120 trials per condition,
`configs/synapsys.yaml`, no gating). Chance is 33.3 %. SNR is broadband; see
[docs/validation.md](docs/validation.md).

| Condition | Accuracy |
|---|---|
| Clean, causal | 100 % |
| SNR 0 to −15 dB, causal | 100 % |
| SNR −20 dB, causal / zero-phase / no preprocessing | 95.8 % / 95.0 % / 85.8 % |
| SNR −25 dB | 67.5 % / 65.0 % / 64.2 % |
| SNR −30 dB | 46.7 % / 49.2 % / 40.8 % |

![Accuracy vs SNR](docs/figures/synthetic/accuracy_vs_snr.png)

## Data source

Wang Y., Chen X., Gao X., Gao S., "A Benchmark Dataset for SSVEP-Based Brain–Computer
Interfaces," *IEEE TNSRE* 25(10), 2017. Download instructions: [data/README.md](data/README.md).
