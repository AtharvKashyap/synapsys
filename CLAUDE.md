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
.venv/bin/python scripts/fetch_benchmark.py 1 2 3            # Wang data -> data/benchmark/ (needs .[data])
.venv/bin/python scripts/benchmark_walkthrough.py         # 7-stage real-EEG check + viewer page
.venv/bin/python scripts/make_fixtures.py               # regenerate fixtures/ (Python->C++ regression set)
.venv/bin/python scripts/mutation_check.py              # plant 11 bugs, confirm tests catch each (~1 min)
cmake -S firmware/decoder -B firmware/decoder/build -G Ninja && cmake --build firmware/decoder/build && ctest --test-dir firmware/decoder/build   # C++ core vs fixtures
. ~/esp/esp-idf/export.sh && cd firmware/esp32 && idf.py build flash monitor   # ESP-IDF v5.3.2
firmware/tools/run_qemu.sh                                 # real firmware in QEMU, no board (exit 0 = PASS)
.venv/bin/python scripts/record_serial.py --port /dev/cu.usbmodem* --seconds 20 --label YES --split calibration --out data/synapsys/S01/cal/yes_01.csv   # needs .[record], RAW_STREAM=y
.venv/bin/python scripts/calibrate_thresholds.py data/synapsys/S01/cal --max-rest-commands-per-min <team limit>
.venv/bin/python scripts/evaluate_heldout.py data/synapsys/S01/test      # once, frozen T/M
.venv/bin/python scripts/verify_device_log.py capture.serial.log         # device == Python on captured samples
.venv/bin/python scripts/make_report_artifacts.py --out docs/report   # all weekly-report evidence
```
macOS/iCloud gotcha: this folder is under iCloud-synced Desktop, which re-applies the `hidden` flag to files in dot-dirs; Python 3.13 then skips the editable-install `.pth` (`ModuleNotFoundError: synapsys_decoder`). So the venv lives at `~/.venvs/synapsys-decoder` and `.venv` is a symlink to it (recreate: `python3 -m venv ~/.venvs/synapsys-decoder && ln -s ~/.venvs/synapsys-decoder .venv`). pytest also sets `pythonpath = ["src"]`.

## Project constants (live in configs/, never hard-code in algorithm modules)
- Targets: YES 60/7≈8.571 Hz, NO 12 Hz, HELP 15 Hz (60 Hz display; 7/5/4 frames/cycle)
- fs 250 Hz; channels O1, Oz, O2, POz; window 2.0–3.0 s (default 2.5); harmonics 2
- Requirements: ≥85 % held-out own-system accuracy, ≤5 s end-to-end latency

## Frozen reference v1 (tag `python-ref-v1`, see docs/reference_v1.md)
- The Python decoder is FROZEN as the reference for the ESP32 port. `tests/test_frozen_reference.py` pins the settings, SOS coefficients and every fixture output. Changing behaviour = deliberate v2 bump + regenerate fixtures + record evidence. Never change algorithm and language at the same time.
- `fixtures/`: synthetic fixtures committed; `fixtures/wang/` generated locally (dataset redistribution terms unstated) and verified via `MANIFEST.json` hashes.
- The C++ must port the Björck–Golub SVD/QR CCA (no covariance inversion, no regularisation). References restart at t=0 each window (constant tables).
- `firmware/`: ESP32/C++ work. Follow `firmware/PLAN.md` phase gates.
- M11 workflow (docs/calibration.md): recordings carry `true_label` (target or `REST`) and `metadata.split` (`calibration`|`test`|`pilot`). `calibrate_thresholds.py` refuses `test`; `evaluate_heldout.py` accepts only `test` and takes T/M from the config only. The false-activation limit (`--max-rest-commands-per-min`) is a team decision: never pick it yourself.
- Serial line formats (`SSVEP {json}`, `RAW <seq> <uV x4>`) are shared by `serial_output.cpp` and `io/serial_log.py`: change both together.

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
- Raw EEG stays out of git (`data/` ignored). Note `data/` sits in the iCloud-synced folder (~640 MB for S1–S3).
- Walkthrough data-quality checks (SSVEP SNR > 3 dB, injected-drift ≥ 20 dB) are dataset diagnostics, not classifier thresholds. A spectral "power below 0.5 Hz" check on a 4.9 s segment is ill-posed (0.2 Hz bins, 0.4 Hz in transition band) — use the injected-drift linearity test.
- Commits: no Claude / Co-Authored-By attribution.

## Layout
`src/synapsys_decoder/`: `config.py`, `types.py` (Trial, ClassificationResult, REJECT),
`references.py`, `cca.py`, `classifier.py`, `preprocessing.py`, `synthetic.py`,
`metrics.py`, `plotting.py`, `pipeline.py` (decode_trial: select→filter epoch→crop→classify), `streaming.py` (device schedule offline), `calibration.py` (T/M sweep + selection rule), `rearm.py`, `runlog.py` (results/<run_id>/ artifacts), `io/{benchmark,csv_eeg,serial_log}.py`.
Scripts: `run_synthetic_validation.py` (M6), `run_benchmark_validation.py` (M7–M8), `analyze_recording.py` (own CSV recordings).
`docs/figures/synthetic/` = committed copy of a synthetic run made from a clean tree; regenerate + recopy if algorithm code changes. Docs in `docs/algorithm.md`,
`docs/validation.md`.

## Milestone status
See README "Status" (M1–M7 done; M8 sweeps run on S1–S3 (2026-10-18), baseline choice to document; S1–S3 downloaded). Python reference frozen 2026-10-18. ESP32 port: phases 0-5 done (host equivalence + QEMU replay 48/48); phase 6 hardware (ADS1299/BLE on a real board) pending. M11 tooling ready: optional firmware raw stream (QEMU: 9000 samples, no gaps; device == Python 48/48 via verify_device_log.py). Disk on this Mac is nearly full (~3 GB free): keep one ESP-IDF build dir.
- Feed the decoder with `push_f64` (DC removal before float rounding); on ESP32-S3 the 16 KB 'IRAM 99.99%' line in idf.py size is normal (overflow goes to DIRAM).
