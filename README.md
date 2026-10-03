# synapsys-decoder

Reference Python implementation of the SynapSys SSVEP decoder (preprocessing + standard CCA + optional confidence/margin gating). It is the algorithmic source of truth for the later ESP32-S3 C/C++ port: the port must reproduce this code's outputs on shared fixtures.

Hardware chain: ADS1299 -> ESP32-S3 -> BLE -> phone. The stimulus renderer lives in the sibling repo `../synapsys-stimulus` and is kept separate.

Standard CCA is **not trained**. Nothing in this decoder is fit to data; the only data-derived quantities are the (not yet calibrated) gating thresholds.

## Pipeline

```
raw EEG (4 ch, 250 Hz, volts)
  -> remove DC            (causal: subtract first sample; zero_phase: subtract epoch mean)
  -> Butterworth band-pass 0.5-45 Hz, N=4 prototype (4 biquads)
  -> 60 Hz IIR notch, Q=30 (1 biquad; 5 biquads total)
  -> fixed window         (most recent 2.0-3.0 s; default 2.5 s = 625 samples)
  -> CCA vs sin/cos references of each target (2 harmonics -> 4 reference rows)
  -> 3 scores (largest canonical correlation per target)
  -> argmax, confidence = top score, margin = top - runner-up
  -> optional gating (thresholds currently null = no gating)
  -> YES / NO / HELP / REJECT
```

Details: [docs/algorithm.md](docs/algorithm.md). Evidence and plans: [docs/validation.md](docs/validation.md).

## Project constants

Defined in `configs/synapsys.yaml`; never hard-coded in algorithm modules.

| Item | Value |
|---|---|
| YES | 60/7 = 8.5714... Hz (display 7 frames/cycle at 60 Hz) |
| NO | 12 Hz (5 frames/cycle) |
| HELP | 15 Hz (4 frames/cycle) |
| Sample rate | 250 Hz |
| Channels | O1, Oz, O2, POz (selected by name) |
| Window | 2.0-3.0 s, default 2.5 s |
| Harmonics | 2 |
| Mains notch | 60 Hz, Q = 30 |
| Requirements | >= 85 % held-out own-system accuracy; <= 5 s end-to-end latency |

Data conventions: EEG arrays are `(n_channels, n_samples)`, float64, in volts. Class order is the insertion order of `targets` in the config. YAML target labels must be quoted (`"YES"`, `"NO"`) because YAML 1.1 parses bare YES/NO as booleans; `config.py` rejects non-string labels.

## Setup

Python 3.11-3.13.

```bash
python3 -m venv ~/.venvs/synapsys-decoder
ln -s ~/.venvs/synapsys-decoder .venv
.venv/bin/pip install -e ".[dev]"
```

The venv lives at `~/.venvs/synapsys-decoder` with `.venv` as a symlink to it. This folder is under iCloud-synced Desktop, which re-applies the `hidden` flag to files in dot-directories; Python 3.13 then skips the editable-install `.pth` file and `import synapsys_decoder` fails with `ModuleNotFoundError`. (pytest additionally sets `pythonpath = ["src"]`.) See `CLAUDE.md`.

## Reproducing every result

All commands from the repo root.

| Goal | Command |
|---|---|
| Unit tests | `.venv/bin/pytest -q` (190 passed, 1 skipped: the real-data benchmark test, skipped when `data/benchmark/` is absent) |
| Format / lint | `.venv/bin/ruff format . && .venv/bin/ruff check .` |
| Synthetic validation (M6) | `.venv/bin/python scripts/run_synthetic_validation.py` |
| Inspect one own-hardware CSV recording | `.venv/bin/python scripts/analyze_recording.py data/synapsys/rec.csv [--config configs/synapsys.yaml] [--step-seconds 0.5]` |
| Public benchmark (milestones 7-8, requires `data/benchmark/`, see [data/README.md](data/README.md)) | `.venv/bin/python scripts/run_benchmark_validation.py --config configs/benchmark.yaml` |

Every script run writes `results/<run_id>/` containing the config used, `run.json` (metrics, seed, package versions, git revision), per-trial predictions (`trials.csv`) and figures. `results/` is a scratch/archive area; the figures embedded below are a copy of one synthetic run in `docs/figures/synthetic/` (including its `run.json`).

`analyze_recording.py` is a debugging / biological-validation tool (PSD before/after filtering, sliding-window CCA scores over time), not an accuracy evaluation.

## Repository layout

```
configs/
  synapsys.yaml          own-hardware decoder config (targets, preprocessing, gating = null)
  benchmark.yaml         Wang 2017 benchmark config (8/12/15 Hz, no 60 Hz notch, epoching section)
data/                    raw EEG, git-ignored (see data/README.md)
docs/
  algorithm.md           equations, filter design, gating semantics, ESP32 porting notes
  validation.md          validation strategy, synthetic results, benchmark and hardware plans
  figures/synthetic/     figures + run.json from the synthetic run
scripts/
  run_synthetic_validation.py
  analyze_recording.py
  run_benchmark_validation.py   (M7-M8)
src/synapsys_decoder/
  config.py              YAML -> DecoderConfig dataclasses, validation
  types.py               Trial, ClassificationResult, REJECT
  references.py          sin/cos harmonic reference matrices / bank
  cca.py                 CCA via principal angles (Bjorck-Golub)
  classifier.py          score_targets, decide (gating), classify_window
  preprocessing.py       DC removal, Butterworth SOS, notch, StreamingFilter, windowing, response analysis
  pipeline.py            decode_trial: select channels -> filter whole epoch -> crop window -> classify
  synthetic.py           synthetic SSVEP / noise / off-target / mixture generators
  metrics.py             accuracy, accepted accuracy, rejection rate, confusion, ITR, false-activation rate
  plotting.py            figures (never imported by algorithm modules)
  runlog.py              run directories and reproducibility artifacts
  io/benchmark.py        Wang 2017 .mat loader (no filtering/resampling)
  io/csv_eeg.py          CSV + JSON-sidecar trial format (also the planned Python<->ESP32 fixture format)
tests/                   pytest suite, one file per module plus script smoke tests
```

## Status

| # | Milestone (handoff spec §16) | Status |
|---|---|---|
| M1 | Repository: package, config model, pytest, Ruff | Done |
| M2 | Reference generation (sin/cos harmonics) | Done |
| M3 | CCA core + independent cross-checks + degenerate handling | Done |
| M4 | Classifier: score all targets, structured result, optional gating | Done |
| M5 | DSP: configurable causal bandpass/notch, zero-phase offline mode, response tests | Done |
| M6 | Synthetic end-to-end: noise/interference, SNR sweep, plots | Done |
| M7 | Public dataset: Wang Benchmark loader + 3-frequency smoke test | Loader + script done (tested on fake data); awaiting dataset download |
| M8 | Window / harmonic / channel experiments | Script done; awaiting dataset download |
| M9 | Embedded design: causal DSP + CCA in C/C++ | Not started (deferred until Python reference is validated) |
| M10 | Cross-platform regression: Python vs ESP32 fixtures | Not started |
| M11 | Hardware integration: acquisition, alpha, SSVEP, calibration, held-out test | Not started |

## Headline synthetic results

Source: `docs/figures/synthetic/run.json` (run `20261003-231156_synthetic_validation`, seed 20261003, 40 trials per class = 120 trials per condition, 6 s epochs, decoder uses the last 2.5 s, `configs/synapsys.yaml`, no gating).

**These are pipeline-behaviour checks on synthetic signals. They are NOT human EEG performance and say nothing about whether SynapSys meets 85 %.**

| Condition | Accuracy |
|---|---|
| Clean synthetic, causal | 120/120 = 100 % |
| SNR 0 to -15 dB (broadband), causal | 100 % |
| SNR -20 dB, causal / zero_phase / no preprocessing | 95.8 % / 95.0 % / 85.8 % |
| SNR -25 dB, causal / zero_phase / no preprocessing | 67.5 % / 65.0 % / 64.2 % |
| SNR -30 dB, causal / zero_phase / no preprocessing | 46.7 % / 49.2 % / 40.8 % |

Chance is 33.3 %. SNR is broadband (white-noise variance); the narrowband SNR around a target is roughly 21 dB higher. See [docs/validation.md](docs/validation.md) for the definition and interpretation.

![Accuracy vs SNR](docs/figures/synthetic/accuracy_vs_snr.png)

## What this repo does NOT claim

- No SynapSys accuracy exists yet. Nothing here has been run on SynapSys hardware or human recordings.
- No gating thresholds exist yet. `confidence_threshold` and `margin_threshold` are `null` in both configs and must be calibrated on own-system development data, never on synthetic data and never on the final test set.
- The public benchmark (Wang et al. 2017) was recorded on different hardware, with a different montage, stimulus and country (50 Hz mains). Results on it would validate the generic CCA implementation only; they do not show that SynapSys reaches 85 %.
- The benchmark has no 8.571 Hz target (its grid is 8.0-15.8 Hz in 0.2 Hz steps). Benchmark trials are never relabelled as YES/NO/HELP; `configs/benchmark.yaml` uses the genuine 8, 12 and 15 Hz targets.
- Synthetic results show pipeline behaviour (shapes, units, frequency selectivity, SNR trends), not human performance.
- The ESP32 port does not exist yet.
