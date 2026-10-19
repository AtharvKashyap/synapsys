# ESP32-S3 decoder port: plan of action

Status (2026-10-19): **phases 0-5 done**, phase 6 code written, needs a physical board.

| Phase | Status | Evidence |
|---|---|---|
| 0 Scaffolding | done | CMake host build, generated config header + byte-exact drift check |
| 1 Filter | done | double bit-exact; float 8.6e-4 x RMS (gate 1e-3) via `push_f64` |
| 2 CCA | done | score err double 1.3e-15, float 9.0e-5; degenerate points exact |
| 3 Decision | done | winner/decisions 100 % on 340 points; 36 near-threshold points listed, none differ |
| 4 Re-arm | done | Python reference + C++ port identical on 140 sequences / 5800 steps |
| 5 ESP-IDF + playback | done in QEMU | real firmware, real-time 250 Hz replay, 48/48 PASS; physical board pending |
| 6 ADS1299 + BLE | code written, builds | needs PCB/board: register read-back, DRDY 250 Hz, BLE notify, 30 min soak |
| 7 prep: raw stream | done in QEMU | optional `SYNAPSYS_RAW_STREAM` prints every raw sample (uV) for calibration recordings; 9000/9000 samples, no gaps, 48/48 PASS |


## 0. Goal and scope

Reproduce the frozen Python reference (`docs/reference_v1.md`, tag `python-ref-v1`) in C++ so
that it runs on the ESP32-S3 in real time and matches `fixtures/` **decision point by decision
point**. Then connect it to the ADS1299 stream and BLE.

| In scope | Out of scope (later milestones) |
|---|---|
| Platform-independent C++ decoder core, host-tested on macOS | Choosing rejection thresholds T, M (needs own calibration data) |
| Fixture-equivalence harness (Python vs C++) | Final accuracy / false-activation testing (M11) |
| Command re-arming state machine (Python first, then C++) | Any algorithm change (filter band, FBCCA, ...) |
| ESP-IDF build, on-device fixture playback, timing and memory | Phone app / TTS |
| ADS1299 acquisition task and BLE output | PCB bring-up itself |

## 1. Ground rules

1. **One variable at a time.** The algorithm is frozen at v1 while the language changes.
   Behaviour changes require a v2 reference (regenerate fixtures, record evidence).
2. **Host first.** Every phase passes on the Mac before anything ESP32-specific is touched.
3. **Gates, not vibes.** Each phase ends with a numeric acceptance test against `fixtures/`.
   A phase is not "mostly done": its gate passes or the phase is open.
4. **Same contract as Python.** First-sample DC removal; SOS coefficients copied from
   `fixtures/config/synapsys_v1.json` (never re-designed in C++); references restart at
   t = 0 every window; Björck–Golub CCA with a rank tolerance; ties go to the first target;
   degenerate input returns an explicit status, never NaN, never a command.
5. **No hidden state.** No globals; no heap allocation after initialisation; fixed sizes
   (4 channels × 625 samples × 3 targets × 2 harmonics) as compile-time constants.

## 2. Correction to the porting advice we received

The advice said to reproduce "the Python algorithm's centering, covariance calculations,
regularization". The Python reference uses **no covariance matrices and no regularisation**.
It centres each row, builds orthonormal bases with a rank-revealing SVD (dropping directions
below `rank_tol × s_max`), and takes the singular values of `Qx^T Qy`. The C++ must port
*that* method:

- Householder QR of the centred 625×4 EEG window → R (4×4); one-sided Jacobi SVD of R
  gives the singular values for the rank test and the basis `Qx = Q·U_R` (same subspace as
  Python's SVD, therefore the same correlations).
- `Qy` for each target is constant (references restart at t = 0 every window), so it is
  computed **once at start-up** and kept in RAM.
- Canonical correlations = singular values of the 4×4 matrix `Qx^T Qy` (Jacobi again).
- Degenerate test identical to Python: zero rank if `s_max ≤ rank_tol·√n·max|raw|`;
  minimum samples `n > p + q`.

## 3. Numerics: measured before planning

Float32 simulation of the frozen pipeline on all 20 fixtures (160 decision points,
synapsys_v1 config), compared with the float64 Python outputs:

| Configuration | Worst error | Winner changes |
|---|---|---|
| float32 biquad cascade (DF2T) | 6.5e-4 × signal RMS | – |
| float32 CCA on float64-filtered data | 4.8e-7 (score) | 0 / 160 |
| everything float32 | 5.4e-5 (score) | 0 / 160 |

Conclusion: float32 throughout is expected to meet the score tolerance (1e-3) with ~20×
headroom, which matters because the ESP32-S3 FPU is single precision only (double is
emulated in software). The core is written as a template on the scalar type so a
`double` build exists for the host: the double build must match Python to ~1e-9 and
proves the logic; the float build proves the precision. Float32 rank tolerance starts at
1e-5 and is validated on `syn_dead_channel` and `syn_flat`.

## 4. Architecture

```
            ┌──────────────────────── decoder core (C++17, platform-independent) ───────────────────────┐
 sample ──► │ DcRemover ─► BiquadCascade×4 ─► RingBuffer 4×625 ─► every 125 samples:                       │
 (4 ch, V)  │                                                     window ─► Cca(Qy table ×3) ─► Decision ─► Rearm │──► Event
            └────────────────────────────────────────────────────────────────────────────────────────────┘
   host: fixture player + ctest                         ESP32: ADS1299 task (core 1) ─► queue ─► decoder task ─► BLE
```

Proposed public API (sketch):

```cpp
namespace synapsys {
enum class Status { kOk, kNotReady, kDegenerate };
struct WindowResult {                // one decision point; mirrors expected.json
  Status status;
  std::array<float, kTargets> scores;
  int winner; float confidence, margin;
  int decision;                      // target index, or kReject
};
class Decoder {
 public:
  explicit Decoder(const Config& cfg);          // builds Qy tables; no allocation after this
  std::optional<WindowResult> push(const float sample[kChannels]);  // result every hop
  void reset();                                 // next sample re-captures DC offsets
};
}
```

Memory: ring buffer 4×625 floats = 10 kB; Qy tables 3×625×4 floats = 30 kB; scratch for
QR ≈ 10 kB. Well inside the ESP32-S3's 512 kB SRAM.

## 5. Phases and gates

### Phase 0: scaffolding (host)
- `firmware/decoder/` CMake project, C++17, `-Wall -Wextra -Werror`, no exceptions/RTTI
  needed by the core; ctest.
- `firmware/tools/gen_config.py`: `fixtures/config/synapsys_v1.json` → generated
  `config_v1.hpp` (SOS coefficients, frequencies, sizes). A test asserts the header and the
  JSON agree, so coefficients can't drift by hand-editing.
- `firmware/tools/flatten_fixtures.py`: `expected.json` → simple CSV the C++ test runner can
  read without a JSON library.
- **Gate 0:** `cmake --build && ctest` runs on the Mac; generated header matches the JSON.

### Phase 1: streaming filter
- `DcRemover` (first sample per channel) + `BiquadCascade` (DF2T, per-channel state).
- **Gate 1:** for every fixture and both configs, per-sample error vs `filtered_<cfg>.csv`:
  double build ≤ 1e-12 V; float build ≤ 1e-3 × signal RMS (measured expectation 6.5e-4).
  Sample-by-sample and block processing give identical output.

### Phase 2: references and CCA
- Reference tables, Householder QR, one-sided Jacobi SVD, rank test, degenerate status.
- **Gate 2:** at every decision point: each score |err| < 1e-3 (float) / 1e-9 (double);
  `all_correlations` match; `syn_flat` reports kDegenerate at exactly the expected points;
  `syn_dead_channel` classifies.

### Phase 3: decision and streaming decoder
- Ring buffer, hop of 125 samples, winner/margin/gating, test-only thresholds.
- **Gate 3:** decision points at the same sample indices; winner 100 %; margin |err| < 1e-3;
  ungated and test-threshold decisions 100 %, except points within 1e-3 of a threshold,
  which are listed in the report rather than silently passed.

### Phase 4: command re-arming (Python first)
The decoder emits a decision every 0.5 s; staring at a target must not repeat the command.
Proposed state machine (parameters are configurable, defaults need team agreement):

| State | Rule |
|---|---|
| ARMED | waiting; an accepted label L starts CANDIDATE(L, 1) |
| CANDIDATE(L, k) | same L accepted again → k+1; anything else → ARMED. At k = K, **emit L** → REFRACTORY |
| REFRACTORY | no output until both: ≥ R s since emit (window flush), and N consecutive decision points that are REJECT or a label ≠ L ("gaze released") → ARMED |

Proposed defaults: K = 2 (adds 0.5 s), R = 2.5 s (one full window of fresh data),
N = 2. Important limitation: until T and M are calibrated, the decoder never outputs
REJECT, so "looking away" cannot be detected; the interim rule is R plus a change of label.
Implement as `src/synapsys_decoder/rearm.py` with tests and a decision-sequence fixture
set, freeze it as reference v1.1, then port.
- **Gate 4:** C++ re-arm output identical to Python on the decision-sequence fixtures.

### Phase 5: ESP-IDF build and on-device playback
- `firmware/esp32/` ESP-IDF (v5.x) project that links the unchanged core.
- Embed a subset of synthetic fixtures in flash; replay at 250 Hz; print results over USB
  serial; host script compares against `expected.json` with the same gates.
- Measure: µs per sample (filter), ms per decision (3× CCA), peak stack, heap after init.
- **Gate 5:** same decisions as Python on device; decision compute < 50 ms (budget: well
  inside the 500 ms hop); zero heap growth over a 1 h replay loop.

### Phase 6: real-time integration
- ADS1299 driver: DRDY interrupt → SPI read (24-bit two's complement → volts using the
  configured PGA gain and VREF), acquisition task pinned to one core, lock-free queue to the
  decoder task on the other core; counters for missed DRDY, queue overruns, SPI errors.
- First tests with the ADS1299 internal test signal, then a function generator.
- BLE GATT: notify emitted commands; a debug characteristic streams scores + margin.
- **Gate 6:** 30 min soak at 250 S/s with zero missed samples / overruns; latency from a
  stimulus change to BLE notification measured with the stimulus sync signal.

### Phase 7: hand-off to hardware validation (M11)
Tooling (done, no board needed): optional raw-sample stream in the firmware, `record_serial.py`,
`calibrate_thresholds.py`, `evaluate_heldout.py`, `verify_device_log.py`; workflow in
`docs/calibration.md`, QEMU evidence in `reports/raw_stream_qemu.md`.
Alpha blocking → single-frequency SSVEP (incl. 8.571 Hz and the 60 Hz notch on human EEG)
→ three-target calibration set with rest trials → choose T, M (and re-arm parameters) on
calibration data → one held-out evaluation.

## 6. Latency budget (to be measured, not assumed)

| Component | Estimate |
|---|---|
| Visual latency | ~0.14 s |
| Fill a 2.5 s window with gaze-on-target data | 2.5 s |
| Wait for next decision point (hop) | 0–0.5 s |
| Re-arm confirmation (K = 2) | +0.5 s |
| Decoder compute on ESP32 | < 0.05 s (to measure, Gate 5) |
| BLE notification | ~0.05–0.1 s |
| Phone speech start | ~0.3–1 s |
| **Total** | **~3.6–4.8 s vs 5 s requirement** |

Tight but feasible. If measurement says otherwise, the levers are the hop (0.25 s), K = 1
with thresholds, or a 2.0 s window (91 % vs 96 % on the benchmark: an evidence-backed
trade-off, not a free one).

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Coefficients copied wrong | Generated header + test vs JSON (Phase 0) |
| Off-by-one window indices / reference phase | Fixtures pin exact window sample indices; references restart at t = 0 |
| Float32 precision in the 0.5 Hz high-pass | Measured 6.5e-4 × RMS; double-state fallback for filter only if Gate 1 fails |
| Rank decisions differ in float32 | Dead-channel and flat fixtures; tolerance validated explicitly |
| Near-threshold flips in gating | Reported explicitly at Gate 3 |
| Missed ADS1299 samples / timing jitter | DRDY interrupt, dedicated core, counters, soak test |
| BLE/radio contention with acquisition | Separate cores and priorities; soak test with BLE active |
| 60 Hz notch untested on human EEG | Phase 7 single-frequency recordings; notch stays configurable |
| Algorithm drift on Python side | `tests/test_frozen_reference.py` + MANIFEST hashes |

## 8. Decisions needed from the team

1. **Re-arming defaults** (K, R, N above) and whether repeating the same command must
   require looking away.
2. **Decision hop**: 0.5 s proposed.
3. **Wang-derived fixtures**: keep generated locally (default) or commit (needs a
   decision on redistribution terms).
4. **85 % denominator** (do REJECTs count as errors?) and the false-activation target.
5. **Board and toolchain**: exact ESP32-S3 module, ESP-IDF version, who owns the ADS1299
   driver.
6. **Language subset**: C++17 without exceptions/RTTI (recommended) vs plain C.

## 9. Definition of done for the port

- Gates 0–6 pass; reports are saved under `firmware/reports/` with the commit hash.
- Every fixture decision point matches Python (winner 100 %, scores/margins < 1e-3).
- Re-arm behaviour identical to the Python reference.
- On device: no missed samples in a 30 min soak; compute time and memory recorded;
  end-to-end latency measured against the 5 s requirement.
- Nothing claims SynapSys accuracy; that remains M11.
