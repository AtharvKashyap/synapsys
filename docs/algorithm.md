# Algorithm

Reference for what `src/synapsys_decoder/` computes. Numbers below are taken from the module docstrings and `configs/synapsys.yaml`; the filter table is asserted (with tolerances) by `tests/test_preprocessing.py`.

Contents: [1. Pipeline](#1-pipeline) - [2. References](#2-reference-signals) - [3. CCA](#3-cca-via-principal-angles) - [4. Classifier and gating](#4-classifier-and-gating) - [5. Preprocessing](#5-preprocessing) - [6. Harmonic overlap](#6-harmonic-overlap) - [7. ESP32 porting notes](#7-esp32-porting-notes)

## 1. Pipeline

```
X_raw (p x N_epoch, volts)
  -> DC removal -> SOS cascade (Butterworth band-pass, optional notch)   [preprocess / StreamingFilter]
  -> crop window X (p x n), n = round(window_seconds * fs)               [extract_window / last_window]
  -> for each target k: s_k = rho_1(X, Y_k)                              [score_targets]
  -> winner, confidence, margin, gating                                  [decide]
```

`pipeline.decode_trial` does this for one `Trial`: select channels by name (config order), **filter the whole epoch, then crop**, classify. The default window is the last `window_seconds` of the epoch (the real-time convention).

Standard CCA has no trained parameters.

## 2. Reference signals

For target frequency f, H harmonics and sample times t_i = i / fs (i = 0..n-1), the reference matrix Y_f is 2H x n:

```
rows:  sin(2 pi 1 f t), cos(2 pi 1 f t), sin(2 pi 2 f t), cos(2 pi 2 f t), ..., cos(2 pi H f t)
```

Row 2(h-1) is the sine and row 2(h-1)+1 the cosine of harmonic h. The layout is fixed so the table can be precomputed per target on-device. A set whose top harmonic H*f >= fs/2 is refused with `ValueError` (no silent aliasing/truncation). `reference_bank` stacks one set per target in class order.

## 3. CCA via principal angles

### Definition

For X in R^(p x n) (EEG channels) and Y in R^(q x n) (references), CCA finds weight vectors u, v maximising corr(u^T X, v^T Y). The maximal value is the first canonical correlation rho_1; all of them are rho_1 >= rho_2 >= ... >= 0. The decoder's per-target score is s_k = rho_1(X, Y_k), in [0, 1].

### Computation (Bjorck & Golub, 1973)

Reference: A. Bjorck and G. H. Golub, "Numerical methods for computing angles between linear subspaces," Math. Comp. 27(123), 1973. Implemented in `cca.canonical_correlations`.

1. Validate: 2-D inputs, equal n, all finite, **n > p + q**.
2. Center each row: X_c = X - mean(X), Y_c = Y - mean(Y) (mean over samples).
3. Orthonormal bases of the row spaces via thin SVD: X_c^T = U S V^T. Keep the columns of U whose singular value satisfies s_i > tol * s_max. This gives Q_x (n x r_x); likewise Q_y (n x r_y).
4. The canonical correlations are the singular values of the small r_x x r_y matrix Q_x^T Q_y, i.e. the cosines of the principal angles between the two subspaces. They are sorted descending and clipped to [0, 1].

`cca_score` returns the first one.

### Why not covariance inversion

The textbook route computes C_xx^-1 C_xy C_yy^-1 C_yx and takes the square root of its largest eigenvalue. That squares the condition number of the data, and a dead, constant or duplicated channel makes C_xx singular (NaN or garbage). The SVD-basis route never forms or inverts a covariance; a rank-deficient direction simply contributes no basis vector. A cross-check against the textbook covariance formulation and against scikit-learn's CCA is part of the test suite.

Consequences (all tested): the result is invariant to channel permutation, per-channel scaling and DC offset (any invertible linear transform of the rows leaves the column space unchanged).

### Minimum samples rule: n > p + q

After centering, the data live in an (n-1)-dimensional subspace of R^n. If p + q > n - 1 and both sides have full rank, the two subspaces must intersect, which forces rho_1 = 1 regardless of the signal. Requiring n >= p + q + 1 excludes this guaranteed spurious result. Real windows (hundreds of samples versus p + q = 8 for 4 channels and 2 harmonics) are far above the bound.

### Rank tolerance and degenerate input

- Directions with s_i <= tol * s_max are dropped. Default `rank_tol = 1e-10` (float64). For a float32 C port use roughly **1e-5** (a few hundred float32 ulps); the tolerance must track arithmetic precision.
- A side is declared zero-rank when s_max <= tol * sqrt(n) * max|raw|. This catches constant rows whose centering leaves only floating-point residue (~1e-16 relative) that a purely relative test would mistake for signal.
- Degenerate input raises `DegenerateSignalError` (a `ValueError` subclass) instead of returning NaN: non-finite values, n <= p + q, no variables, or a zero-rank side (e.g. all channels constant). A single dead channel next to real ones is fine: it is simply dropped.

## 4. Classifier and gating

`classifier.decide` operates on scores from an already-preprocessed window (the classifier never filters).

- **Winner**: highest score. Ties are broken deterministically by class order (strict `>` while iterating in config order, so the first label wins).
- **Confidence** = winner score. **Margin** = winner score - runner-up score (equals the winner score with a single class).
- **Gating**: thresholds are `confidence_threshold` in [0, 1] and `margin_threshold` >= 0, each optional.
  - A window is accepted only if every configured threshold passes. Comparisons are **inclusive**: `confidence >= confidence_threshold` and `margin >= margin_threshold`.
  - A `None` threshold is skipped.
  - If both are `None` the result is **ungated**: `gated = False`, `accepted = True`, `decision = winner`.
  - If any threshold fails, `accepted = False` and `decision = REJECT`. The result still reports the winner, scores, confidence and margin, so rejected windows can be analysed.
- `ClassificationResult` fields: `scores` (ordered as the config), `winner`, `winner_frequency`, `confidence`, `margin`, `gated`, `accepted`, `decision`.
- Validation: score labels must equal the target labels; scores must be finite; thresholds out of range raise `ValueError`.

Both thresholds are `null` in `configs/synapsys.yaml` and `configs/benchmark.yaml`. They are to be calibrated on own-system development data (see [validation.md](validation.md#threshold-calibration-plan)).

## 5. Preprocessing

### Structure

Stage order: remove DC -> Butterworth band-pass -> IIR notch -> window. Each stage can be disabled in `PreprocessingConfig` (`bandpass_*_hz: null`, `notch_hz: null`, `remove_dc: false`).

- Band-pass: `scipy.signal.butter(N, [lo, hi], "bandpass")` with N = `bandpass_order` = the **prototype order**. A band-pass has 2N poles. Default N = 4 gives 8 poles = **4 biquads** (4th-order 0.5 Hz high-pass skirt and 4th-order 45 Hz low-pass skirt). A single high- or low-pass edge of order N has N poles (ceil(N/2) biquads).
- Notch: `scipy.signal.iirnotch(60, Q=30)`: 1 biquad, -3 dB bandwidth = 60/30 = 2 Hz.
- Default cascade: **5 biquads** (4 band-pass + 1 notch).

### SOS layout

`build_filter_sos` returns an `(n_sections, 6)` array. Rows: band-pass sections first (scipy order), then the notch. Each row is `[b0, b1, b2, a0, a1, a2]` with `a0 = 1`; section k's output is section k+1's input. Each section has two state variables per channel (direct form II transposed, as in `sosfilt`). `remove_dc` and `mode` are not part of the SOS. The identical array is what the C port implements.

### Frequency response (fs = 250 Hz, N = 4, 0.5-45 Hz, 60 Hz notch Q = 30)

Measured with `frequency_response_db`, rounded to 0.01 dB:

| Freq (Hz) | Band-pass only (dB) | Band-pass + notch (dB) | Note |
|---|---|---|---|
| 0.2 | -32.13 | -32.13 | drift, attenuated |
| 0.5 | -3.01 | -3.01 | low -3 dB edge |
| 8.571 | -0.00 | -0.00 | YES |
| 12 | -0.00 | -0.00 | NO |
| 15 | -0.00 | -0.00 | HELP |
| 17.14 | -0.00 | -0.00 | YES 2nd harmonic |
| 24 | -0.01 | -0.01 | NO 2nd harmonic |
| 30 | -0.09 | -0.09 | HELP 2nd harmonic |
| 45 | -3.01 | -3.03 | high -3 dB edge |
| 50 | -6.03 | -6.07 | |
| 60 | -13.98 | < -100 (-292 numeric) | mains |
| 80 | -31.89 | -31.90 | |

Every SSVEP fundamental and 2nd harmonic passes with < 0.1 dB loss. The 45 Hz low-pass alone gives only ~14 dB at 60 Hz; the notch supplies the depth. The notch is -3 dB at 59/61 Hz, -0.97 dB at 58/62 Hz and -0.17 dB at 55/65 Hz, so it does not touch the SSVEP band.

### Group delay (causal cascade)

11.5 ms at 8.571 Hz, 10.3 ms at 12 Hz, 10.0 ms at 15 Hz (< 3 samples; nearly equal across targets, so it only shifts effective stimulus latency by ~10 ms).

### Settling of the causal filter

The 0.5 Hz high-pass dominates (`settling_time_s`, impulse-response energy tail):

| Criterion | Time |
|---|---|
| Remaining energy <= 1e-2 | 0.23 s |
| <= 1e-3 | 1.05 s |
| <= 1e-4 | 2.10 s |
| Step response above 10 % of a residual DC step | 1.57 s |
| Step response above 1 % | 3.71 s |

When an epoch is filtered causally from its first sample, the first ~1-2 s carry a mostly sub-1 Hz start-up transient (not zero). On the real-time device the filter runs continuously, so this matters only after boot or reset: discard ~2-4 s of output.

### Causal vs zero-phase

| | `causal` | `zero_phase` |
|---|---|---|
| Implementation | `sosfilt`, past samples only | `sosfiltfilt` (forward + backward) |
| Needs future samples | No | Yes |
| Portable to ESP32 | Yes | No (offline only) |
| Use | the real path | comparison / ablation |

`zero_phase` raises if the epoch is shorter than scipy's padding length.

### DC removal policy (`remove_dc`)

- `causal`: subtract each channel's **first sample** from the whole stream, then filter from zero state. For a cascade containing a high-pass this is mathematically identical to initialising the filter state with `sosfilt_zi(sos) * x[:, 0]`. Residual DC is then removed by the 0.5 Hz high-pass; with the high-pass disabled the residual (true baseline minus first sample) remains. Uses only the past, so it is portable (one constant per channel).
- `zero_phase`: subtract each channel's **epoch mean** (non-causal), then `sosfiltfilt`.
- `remove_dc = false`: no subtraction; causal filtering starts from zero state, so a large electrode offset produces a large start-up transient.

### Filter the whole epoch, then crop

The device cannot see the future but filters its stream continuously and then takes the latest window. Offline evaluation must reproduce this: filter the **whole epoch** causally, then crop the analysis window. Cropping first would give the window a cold-start filter and measure a transient the device does not have after warm-up. `pipeline.decode_trial` follows this rule.

Window sample counts use `round(seconds * fs)` (nearest integer, ties to even), the same rule as `DecoderConfig.window_samples`. Example: fs = 250, start 0.64 s, 2.5 s -> samples 160..784 (625 samples).

### StreamingFilter

`StreamingFilter(sos, n_channels, remove_dc)` is the reference for the device loop: per-channel state `zi` of shape `(n_sections, n_channels, 2)`, chunked input of any size, same DC policy (first sample after construction or `reset()` is stored and subtracted). It is tested to equal `preprocess(..., mode="causal")` on the concatenated stream. `StreamingFilter.from_config` rejects `zero_phase`.

## 6. Harmonic overlap

With the configured 2 harmonics the reference frequencies are:

| Target | f (Hz) | 2f (Hz) |
|---|---|---|
| YES | 8.571 | 17.143 |
| NO | 12 | 24 |
| HELP | 15 | 30 |

None coincide, so the three reference subspaces are distinguishable by frequency content. This does not hold for many harmonics: HELP's 4th harmonic (15 x 4 = 60 Hz) and NO's 5th (12 x 5 = 60 Hz) both land on 60 Hz, colliding with each other and with the mains frequency that the notch removes. With >= 4-5 harmonics the references for NO and HELP share a component, so keep `harmonics` at 2 (or at most 3) unless this is re-examined. The Nyquist check (H*f < fs/2 = 125 Hz) does not catch this.

## 7. ESP32 porting notes

Python is the source of truth; the port must match it.

- **Filter**: one SOS array, 5 biquads x 6 coefficients, applied per channel with 2 state variables per section. Export the coefficients from `build_filter_sos`. Use the first-sample DC policy and the boot/reset discard period above. Float32 biquad cascades with a 0.5 Hz high-pass (poles close to z = 1 at fs = 250 Hz) are the numerically sensitive part: check float32 against the float64 output before trusting it.
- **References**: precompute one 2H x n sin/cos table per target (class order) as a constant. Per window, only the EEG basis changes.
- **Precomputed Q_y**: the reference-side centering and orthonormal basis Q_y (n x r_y) depend only on target, fs, n and H, so they can also be precomputed per target; the device then only computes Q_x and the SVD of Q_x^T Q_y.
- **Small SVD**: the decomposition of Q_x^T Q_y is at most 8 x 8 (p <= 8, q = 2H <= 8). A one-sided Jacobi SVD is adequate. Q_x itself can come from a one-sided Jacobi SVD or column-pivoted Householder / modified Gram-Schmidt QR on the n x p matrix; any rank-revealing method spanning the same space is valid.
- **Precision**: start with float32 and `rank_tol` ~ 1e-5 (float64 default is 1e-10). Make degenerate input an explicit error/flag as in Python, never NaN.
- **Rules to preserve**: n > p + q; tie-break by class order; inclusive thresholds; REJECT semantics (Section 4).
- **Fixture equivalence (M10)**: exchange test epochs using the CSV + JSON sidecar format of `io/csv_eeg.py` (CSV with one column per channel and optional `time_s`; sidecar with `fs`, `units`, optional `true_label`, `true_frequency`, `metadata`; values written to 10 significant digits). Run the same epoch through Python and the device and compare, in this order: (1) filtered samples, (2) per-target scores, (3) winner, (4) margin, (5) decision. Numeric tolerances are **not set yet**: they are to be chosen after observing the actual float32-vs-float64 differences; decision-level outputs should match exactly except on windows whose margin is within that numeric noise.
