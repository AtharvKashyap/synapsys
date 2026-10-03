# Mutation check: does the test suite catch planted bugs?

Each row: one deliberate bug injected into a temporary copy of the code, then the full pytest suite run against it. The real working tree is untouched.

**Baseline (unmodified code): 193 passed, 0 failed.**

| # | Planted bug | Simulates | Tests failed | Caught | Example failing test |
|---|---|---|---|---|---|
| 1 | `ref_sin_becomes_cos` (references.py) | reference generator emits cos instead of sin (CCA loses the quadrature pair) | 8 | yes | `test_finite_and_in_unit_interval` |
| 2 | `ref_no_harmonics` (references.py) | every 'harmonic' row is generated at the fundamental | 10 | yes | `test_finite_and_in_unit_interval` |
| 3 | `ref_wrong_time_base` (references.py) | sample-rate bug: references at twice the intended frequency | 5 | yes | `test_benchmark_script_runs_end_to_end` |
| 4 | `cca_no_centering` (cca.py) | EEG not mean-centred before CCA (DC offsets corrupt correlations) | 9 | yes | `test_dc_offset_invariance` |
| 5 | `classifier_picks_lowest` (classifier.py) | argmax replaced by argmin: the least-correlated command wins | 33 | yes | `test_benchmark_script_runs_end_to_end` |
| 6 | `classifier_margin_wrong` (classifier.py) | margin computed against the worst class instead of the runner-up | 3 | yes | `test_margin_and_confidence` |
| 7 | `gating_inverted` (classifier.py) | confidence gate rejects confident windows and accepts weak ones | 7 | yes | `test_gating[0.5-None-True]` |
| 8 | `causal_mode_uses_future` (preprocessing.py) | 'causal' path silently uses future samples (would not port to ESP32) | 7 | yes | `test_signal_components[causal]` |
| 9 | `notch_mistuned` (preprocessing.py) | 60 Hz notch designed at 65 Hz | 4 | yes | `test_documented_response_table` |
| 10 | `pipeline_skips_filtering` (pipeline.py) | classifier fed the unfiltered signal (preprocessing silently bypassed) | 1 | yes | `test_default_window_is_last_window_and_explicit_start_works` |
| 11 | `yaml_yes_no_guard_removed` (config.py) | YAML YES/NO-as-boolean labels accepted silently | 1 | yes | `test_unquoted_yes_no_labels_rejected` |

**11/11 planted bugs caught.**
