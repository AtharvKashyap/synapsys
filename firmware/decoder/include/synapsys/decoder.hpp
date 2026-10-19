// SynapSys SSVEP decoder core: public API (platform-independent C++17).
//
// Reproduces the frozen Python reference v1 (docs/reference_v1.md, tag python-ref-v1):
//
//   sample (4 ch, volts) -> subtract first sample per channel -> SOS biquad cascade (DF2T)
//   -> ring buffer of the last kWindow samples -> every hop_samples: CCA against sin/cos
//   references (restarting at t = 0 each window) -> winner / margin / gating -> decision.
//
// Rules: no heap allocation after construction, no exceptions, no RTTI, no globals.
// A Decoder is large (~50 kB with float); allocate it statically, never on a task stack.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace synapsys {

#if defined(SYNAPSYS_REAL_DOUBLE)
using Real = double;  // host reference build: must match Python to ~1e-9
#else
using Real = float;   // ESP32-S3 build: single-precision FPU
#endif

inline constexpr int kChannels = 4;   // O1, Oz, O2, POz (config order)
inline constexpr int kTargets = 3;    // YES, NO, HELP (config order)
inline constexpr int kHarmonics = 2;
inline constexpr int kRefRows = 2 * kHarmonics;  // sin/cos per harmonic
inline constexpr int kWindow = 625;   // 2.5 s at 250 Hz
inline constexpr int kMaxSections = 8;
inline constexpr int kMaxCorr = (kChannels < kRefRows) ? kChannels : kRefRows;
inline constexpr int kReject = -1;    // decision value for REJECT

// Default rank tolerance per precision (Python float64 reference uses 1e-10).
inline constexpr double kDefaultRankTol = sizeof(Real) == sizeof(double) ? 1e-10 : 1e-5;

struct Config {
  double sample_rate_hz = 250.0;
  std::array<double, kTargets> frequencies_hz{};
  std::array<const char*, kTargets> labels{};
  int n_sections = 0;  // number of valid rows in sos
  // Each row: b0, b1, b2, a0 (== 1), a1, a2 -- exactly scipy.signal SOS layout.
  std::array<std::array<double, 6>, kMaxSections> sos{};
  bool remove_dc = true;  // subtract each channel's first sample of the stream
  int hop_samples = 125;  // decision every 0.5 s once kWindow samples exist
  double rank_tol = kDefaultRankTol;
  // Gating. Unset (false) until calibrated on own-system data -- never invent values.
  bool has_confidence_threshold = false;
  double confidence_threshold = 0.0;
  bool has_margin_threshold = false;
  double margin_threshold = 0.0;
};

enum class Status : uint8_t {
  kOk = 0,
  kDegenerate = 1,  // zero-rank side after centring (e.g. flat input): no command, never NaN
};

struct WindowResult {
  Status status = Status::kOk;
  uint32_t window_end_sample = 0;  // samples pushed so far (window = [end - kWindow, end))
  std::array<Real, kTargets> scores{};  // largest canonical correlation per target
  std::array<std::array<Real, kMaxCorr>, kTargets> all_correlations{};  // sorted descending
  std::array<int, kTargets> n_correlations{};
  int winner = kReject;  // target index with max score; ties -> lowest index
  Real confidence = 0;   // winner score
  Real margin = 0;       // winner - runner-up
  bool gated = false;    // at least one threshold configured
  bool accepted = false; // false only if a configured threshold failed (or degenerate)
  int decision = kReject;  // winner if accepted, else kReject
};

// Pure decision rule (same as Python classifier.decide). Fills winner..decision fields.
void decide(const std::array<Real, kTargets>& scores, const Config& cfg, WindowResult* out);

class Decoder {
 public:
  explicit Decoder(const Config& cfg);  // precomputes reference bases; may be slow-ish

  // Push one sample (volts, config channel order). Returns true when this sample completes a
  // scheduled decision point (samples_seen >= kWindow and (samples_seen - kWindow) % hop == 0);
  // the result is written to *out (if non-null).
  bool push(const Real sample[kChannels], WindowResult* out);

  // Same as push(), but takes float64 volts: the first-sample DC offset is stored and
  // subtracted in double before the result is rounded to Real. Use this (or convert ADC
  // counts relative to the first count) when the input carries a large electrode offset:
  // rounding e.g. 0.3 V to float32 (ulp 3e-8 V) before DC removal adds error comparable to
  // a few-uV EEG signal. In the double build it is identical to push().
  bool push_f64(const double sample[kChannels], WindowResult* out);

  // Evaluate the window ending at the current sample, regardless of the hop schedule.
  // Requires samples_seen() >= kWindow; otherwise returns status kDegenerate with winner kReject.
  WindowResult evaluate();

  // Clear filter state, ring buffer and DC offsets; the next sample re-captures DC offsets.
  void reset();

  uint32_t samples_seen() const { return samples_seen_; }
  const Config& config() const { return cfg_; }

  // Latest filtered sample (for streaming the filtered signal / equivalence tests).
  const std::array<Real, kChannels>& last_filtered() const { return last_filtered_; }

  // Copy the current window of filtered samples in chronological order (oldest first).
  // Only meaningful once samples_seen() >= kWindow (earlier slots are zero). Test/debug aid.
  void copy_window(std::array<std::array<Real, kWindow>, kChannels>* out) const;

 private:
  // Builds an orthonormal basis of the centred rows held in cols[0..p-1] (in place).
  // Returns its rank (0 = degenerate side). raw_max_abs = max |uncentred value|.
  static int orthonormal_basis(Real* const* cols, int p, double raw_max_abs, double rank_tol);

  Config cfg_;
  // SOS coefficients in the working precision: b0, b1, b2, a1, a2 (a0 == 1).
  std::array<std::array<Real, 5>, kMaxSections> coef_{};
  uint32_t samples_seen_ = 0;
  bool have_offset_ = false;
  std::array<double, kChannels> offset_{};  // first sample per channel (double, see push_f64)
  std::array<std::array<std::array<Real, 2>, kChannels>, kMaxSections> state_{};  // DF2T z1,z2
  std::array<Real, kChannels> last_filtered_{};
  // Ring buffer, channel-major: ring_[ch][i]; head_ = next write index.
  std::array<std::array<Real, kWindow>, kChannels> ring_{};
  int head_ = 0;
  // Orthonormal bases of the centred references, per target: qy_[t][col][n], col < qy_rank_[t].
  std::array<std::array<std::array<Real, kWindow>, kRefRows>, kTargets> qy_{};
  std::array<int, kTargets> qy_rank_{};
  // Scratch for the EEG window basis (no allocation in the hot path).
  std::array<std::array<Real, kWindow>, kChannels> scratch_{};
};

}  // namespace synapsys
