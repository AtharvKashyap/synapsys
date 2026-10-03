**pytest: 193 passed, 0 failed, 1 skipped (of 194)**

| Test file | Verifies | Tests | Passed |
|---|---|---|---|
| test_config.py | YAML config loading, validation, YES/NO label guard | 8 | 8 |
| test_references.py | sin/cos reference shapes, values, FFT frequency (incl. 60/7 Hz) | 17 | 17 |
| test_cca.py | CCA bounds, matched > unmatched, invariances, 2 independent cross-checks | 24 | 24 |
| test_classifier.py | 3-class decisions, margin, gating/REJECT, tie-break | 33 | 33 |
| test_preprocessing.py | filter dB response, 12 Hz kept / drift-60-80 Hz removed, causality | 45 | 45 |
| test_synthetic.py | synthetic generator: SNR accuracy, frequencies, determinism | 12 | 12 |
| test_metrics.py | accuracy, confusion matrix, REJECT handling, ITR | 10 | 10 |
| test_plotting.py | every figure renders and saves | 7 | 7 |
| test_pipeline.py | end-to-end decode, channel order, gating, run artifacts, scripts | 13 | 13 |
| test_io_benchmark.py | public-dataset loader (fake files): channels, units, frequencies | 13 | 12 (1 skipped: needs real data) |
| test_io_csv.py | CSV recording/fixture format round-trip | 10 | 10 |
| test_benchmark_script.py | benchmark experiment script end-to-end (fake data) | 2 | 2 |
