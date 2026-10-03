# CLAUDE.md — synapsys-decoder

Reference Python DSP + CCA decoder for the SynapSys SSVEP BCI
(ADS1299 → ESP32-S3 → BLE → phone). Python here is the **source of truth**;
the ESP32 C/C++ port must reproduce it (fixture equivalence, milestone 10).
Sibling repo `../synapsys-stimulus` is the PsychoPy stimulus renderer — keep them separate.

## Commands
```bash
.venv/bin/pip install -e ".[dev]"        # setup (python3 -m venv .venv first)
.venv/bin/pytest -q                       # all tests
.venv/bin/ruff format . && .venv/bin/ruff check .
.venv/bin/python scripts/run_synthetic_validation.py      # M6 artifacts -> results/
.venv/bin/python scripts/run_benchmark_validation.py      # M7/M8, needs data/benchmark/
```

## Project constants (live in configs/, never hard-code in algorithm modules)
- Targets: YES 60/7≈8.571 Hz, NO 12 Hz, HELP 15 Hz (60 Hz display; 7/5/4 frames/cycle)
- fs 250 Hz; channels O1, Oz, O2, POz; window 2.0–3.0 s (default 2.5); harmonics 2
- Requirements: ≥85 % held-out own-system accuracy, ≤5 s end-to-end latency

## Conventions (do not change silently)
- EEG arrays: `(n_channels, n_samples)`, float64, **volts**. Select channels by name.
- Class order = insertion order of `targets` in the config.
- YAML target labels must be **quoted** (`"YES"`, `"NO"`) — YAML 1.1 parses bare YES/NO as booleans. `config.py` rejects non-string labels.
- Pure functions, explicit `fs`/window args, no global mutable state, no plotting in algorithm modules.
- Loaders (`io/`) never filter or resample; they only reshape, convert to volts, attach provenance.
- Deterministic seeds (`numpy.random.default_rng(seed)`) in all synthetic code/tests.
- Every result run saves its config + run ID + per-trial predictions under `results/<run_id>/`.

## Guardrails (from the handoff spec — enforce these)
- Standard CCA is **not trained**. Never describe it as a model fit to data.
- `confidence_threshold` / `margin_threshold` stay `null` until calibrated on own-system dev data. Never invent them; never tune on the final test set; never pick human thresholds from synthetic data.
- `mode: zero_phase` (sosfiltfilt) is offline-only. The ESP32 path is `mode: causal`; on-device the filter runs continuously on the stream, so offline evaluation should filter the whole epoch causally and **then** crop the window.
- Wang Benchmark has no 8.571 Hz target (0.2 Hz grid, 8.0–15.8 Hz). Never relabel benchmark trials as YES/NO/HELP. Benchmark results validate the generic CCA only — they do not prove SynapSys meets 85 %.
- Benchmark was recorded on 50 Hz mains; `configs/benchmark.yaml` disables the 60 Hz notch.
- Raw EEG stays out of git (`data/` ignored).
- Commits: no Claude / Co-Authored-By attribution.

## Layout
`src/synapsys_decoder/`: `config.py`, `types.py` (Trial, ClassificationResult, REJECT),
`references.py`, `cca.py`, `classifier.py`, `preprocessing.py`, `synthetic.py`,
`metrics.py`, `plotting.py`, `io/{benchmark,csv_eeg}.py`. Docs in `docs/algorithm.md`,
`docs/validation.md`.

## Milestone status
See README "Status". Scope agreed so far: Python milestones M1–M8; ESP32 port (M9+) deferred.
