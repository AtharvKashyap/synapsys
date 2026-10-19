# Python reference v1 (frozen 2026-10-18)

This is the decoder the ESP32-S3 port must reproduce. Git tag: `python-ref-v1`.
Guarded by `tests/test_frozen_reference.py` and the fixtures in `fixtures/`.

| Parameter | Frozen value |
|---|---|
| Sample rate | 250 Hz |
| Channels | O1, Oz, O2, POz (this order) |
| DC handling | subtract each channel's first sample of the stream |
| Bandpass | 0.5–45 Hz Butterworth, prototype order 4 (8 poles, 4 biquads), causal |
| Notch | 60 Hz, Q = 30 (one biquad), configurable / can be disabled |
| Window | 2.5 s = 625 samples, last 625 filtered samples |
| References | sin/cos at f and 2f (2 harmonics), restarting at t = 0 each window |
| Targets | YES 60/7 ≈ 8.5714 Hz, NO 12 Hz, HELP 15 Hz (class order YES, NO, HELP) |
| CCA | Björck–Golub principal angles, rank_tol 1e-10 (float64), no regularisation |
| Decision | max score; ties → first in class order; margin = winner − runner-up |
| Thresholds T, M | **unset** until calibrated on own-system data |

## Evidence behind the freeze

- 196+ unit/integration tests; mutation check catches 11/11 planted bugs.
- Synthetic: 100 % clean; graceful degradation with SNR (docs/validation.md §2).
- Wang 2017 S1–S3, 8/12/15 Hz, this exact pipeline: 52/54 = 96.3 % causal (§3).
  2.5 s ≈ 3.0 s; 2 harmonics > 1.

## What v1 does NOT establish

- Accuracy at 8.571 Hz jointly with 12/15 Hz (benchmark has no 8.571 Hz target).
- 60 Hz notch on human recordings (benchmark was recorded on 50 Hz mains, pre-notched).
- Rejection thresholds, false-activation rate, naive-user accuracy, end-to-end latency.

## Change policy

Do not change the algorithm and the language at the same time. Any change to v1 behaviour
(filter band, order, notch, window, harmonics, CCA numerics, tie-break) is a new reference
version: bump to v2 in this file, regenerate fixtures, re-run the C++ equivalence suite,
and record the evidence for the change. Comparing alternative filters (e.g. 5–35 Hz) is
done as an experiment on the side, not by editing v1.
