// SynapSys SSVEP decoder core (see include/synapsys/decoder.hpp for the contract).
//
// Numerical notes
//  * Everything runs in Real (float on the ESP32, double in the host reference build),
//    except a few cheap O(n) reductions that are always accumulated in double:
//    the per-row means used for centring and max|raw| for the rank test.
//  * Reference tables are evaluated in double (sin/cos of 2*pi*h*f*n/fs) and centred in
//    double before being rounded to Real; their bases are then built once in Real.
//  * CCA = Bjorck-Golub principal angles: Householder QR of the centred window, one-sided
//    Jacobi SVD of R for the rank test and U_R, Qx = Q U_R[:, keep], correlations =
//    singular values of Qx^T Qy (Jacobi), sorted descending, clipped to [0, 1].
#include "synapsys/decoder.hpp"

#include <algorithm>
#include <cmath>

#include "linalg.hpp"

namespace synapsys {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

void mark_reject(WindowResult* r) {
  r->winner = kReject;
  r->confidence = 0;
  r->margin = 0;
  r->accepted = false;
  r->decision = kReject;
}

}  // namespace

void decide(const std::array<Real, kTargets>& scores, const Config& cfg, WindowResult* out) {
  int winner = 0;
  Real best = scores[0];
  for (int t = 1; t < kTargets; ++t) {
    if (scores[t] > best) {  // strict '>' => first target wins ties
      best = scores[t];
      winner = t;
    }
  }
  Real runner_up = 0;
  bool have_other = false;
  for (int t = 0; t < kTargets; ++t) {
    if (t == winner) continue;
    if (!have_other || scores[t] > runner_up) runner_up = scores[t];
    have_other = true;
  }
  const Real margin = have_other ? best - runner_up : best;

  const bool gated = cfg.has_confidence_threshold || cfg.has_margin_threshold;
  bool accepted = true;
  if (cfg.has_confidence_threshold && static_cast<double>(best) < cfg.confidence_threshold) {
    accepted = false;
  }
  if (cfg.has_margin_threshold && static_cast<double>(margin) < cfg.margin_threshold) {
    accepted = false;
  }
  out->scores = scores;
  out->winner = winner;
  out->confidence = best;
  out->margin = margin;
  out->gated = gated;
  out->accepted = accepted;
  out->decision = accepted ? winner : kReject;
}

int Decoder::orthonormal_basis(Real* const* cols, int p, double raw_max_abs, double rank_tol) {
  constexpr int n = kWindow;
  constexpr int N = (kChannels > kRefRows) ? kChannels : kRefRows;
  std::array<Real, N> tau{};
  detail::householder_qr(cols, n, p, tau.data());

  detail::Small<Real, N> r{};
  for (int j = 0; j < p; ++j) {
    for (int i = 0; i <= j; ++i) r[i][j] = cols[j][i];
  }
  std::array<Real, N> s{};
  detail::Small<Real, N> u_r{};
  detail::jacobi_svd<Real, N>(r, p, p, &s, &u_r);

  const double s_max = static_cast<double>(s[0]);
  if (!(s_max > 0.0) || s_max <= rank_tol * std::sqrt(static_cast<double>(n)) * raw_max_abs) {
    return 0;
  }
  int rank = 0;
  while (rank < p && static_cast<double>(s[rank]) > rank_tol * s_max) ++rank;

  detail::form_q(cols, n, p, tau.data());
  // Qx = Q * U_R[:, :rank], in place row by row (each row only depends on itself).
  for (int i = 0; i < n; ++i) {
    std::array<Real, N> row{};
    for (int k = 0; k < p; ++k) row[k] = cols[k][i];
    for (int j = 0; j < rank; ++j) {
      Real acc = 0;
      for (int k = 0; k < p; ++k) acc += row[k] * u_r[k][j];
      cols[j][i] = acc;
    }
  }
  return rank;
}

Decoder::Decoder(const Config& cfg) : cfg_(cfg) {
  for (int s = 0; s < cfg_.n_sections && s < kMaxSections; ++s) {
    const auto& row = cfg_.sos[s];
    const double a0 = row[3];  // == 1 for scipy SOS; normalise defensively
    coef_[s] = {static_cast<Real>(row[0] / a0), static_cast<Real>(row[1] / a0),
                static_cast<Real>(row[2] / a0), static_cast<Real>(row[4] / a0),
                static_cast<Real>(row[5] / a0)};
  }
  if (cfg_.n_sections > kMaxSections) cfg_.n_sections = kMaxSections;
  if (cfg_.n_sections < 0) cfg_.n_sections = 0;
  if (cfg_.hop_samples < 1) cfg_.hop_samples = 1;

  // Reference bases: rows sin(2 pi h f n / fs), cos(...) for h = 1..H, n = 0..kWindow-1,
  // restarting at n = 0 in every window, so they are constant and computed once here.
  for (int t = 0; t < kTargets; ++t) {
    double raw_max = 0.0;
    std::array<Real*, kRefRows> cols{};
    for (int h = 1; h <= kHarmonics; ++h) {
      for (int trig = 0; trig < 2; ++trig) {
        const int row = 2 * (h - 1) + trig;
        cols[row] = qy_[t][row].data();
        double sum = 0.0;
        for (int i = 0; i < kWindow; ++i) {
          const double phase = 2.0 * kPi * h * cfg_.frequencies_hz[t] *
                               (static_cast<double>(i) / cfg_.sample_rate_hz);
          const double v = trig == 0 ? std::sin(phase) : std::cos(phase);
          sum += v;
          raw_max = std::max(raw_max, std::fabs(v));
        }
        const double mean = sum / kWindow;
        for (int i = 0; i < kWindow; ++i) {
          const double phase = 2.0 * kPi * h * cfg_.frequencies_hz[t] *
                               (static_cast<double>(i) / cfg_.sample_rate_hz);
          const double v = trig == 0 ? std::sin(phase) : std::cos(phase);
          qy_[t][row][i] = static_cast<Real>(v - mean);
        }
      }
    }
    qy_rank_[t] = orthonormal_basis(cols.data(), kRefRows, raw_max, cfg_.rank_tol);
  }
  reset();
}

void Decoder::reset() {
  samples_seen_ = 0;
  have_offset_ = false;
  offset_ = {};
  state_ = {};
  last_filtered_ = {};
  for (auto& ch : ring_) ch.fill(0);
  head_ = 0;
}

bool Decoder::push(const Real sample[kChannels], WindowResult* out) {
  // Real -> double is exact; for float input the double difference is then rounded once to
  // float, i.e. the same as a float subtraction whenever the difference is exact in double
  // (always, for samples within 2^29 of each other in magnitude).
  double s[kChannels];
  for (int c = 0; c < kChannels; ++c) s[c] = static_cast<double>(sample[c]);
  return push_f64(s, out);
}

bool Decoder::push_f64(const double sample[kChannels], WindowResult* out) {
  if (cfg_.remove_dc && !have_offset_) {
    for (int c = 0; c < kChannels; ++c) offset_[c] = sample[c];
    have_offset_ = true;
  }
  for (int c = 0; c < kChannels; ++c) {
    Real x = static_cast<Real>(cfg_.remove_dc ? sample[c] - offset_[c] : sample[c]);
    for (int s = 0; s < cfg_.n_sections; ++s) {
      const auto& k = coef_[s];
      auto& z = state_[s][c];
      // Direct form II transposed, exactly scipy.signal.sosfilt.
      const Real y = k[0] * x + z[0];
      z[0] = k[1] * x - k[3] * y + z[1];
      z[1] = k[2] * x - k[4] * y;
      x = y;
    }
    last_filtered_[c] = x;
    ring_[c][head_] = x;
  }
  head_ = (head_ + 1 == kWindow) ? 0 : head_ + 1;
  ++samples_seen_;

  if (samples_seen_ < static_cast<uint32_t>(kWindow)) return false;
  if ((samples_seen_ - kWindow) % static_cast<uint32_t>(cfg_.hop_samples) != 0) return false;
  if (out != nullptr) *out = evaluate();
  return true;
}

void Decoder::copy_window(std::array<std::array<Real, kWindow>, kChannels>* out) const {
  // Oldest sample sits at head_ once the buffer is full (and before that, slots are zero).
  const int first = kWindow - head_;
  for (int c = 0; c < kChannels; ++c) {
    std::copy(ring_[c].begin() + head_, ring_[c].end(), (*out)[c].begin());
    std::copy(ring_[c].begin(), ring_[c].begin() + head_, (*out)[c].begin() + first);
  }
}

WindowResult Decoder::evaluate() {
  WindowResult r;
  r.window_end_sample = samples_seen_;
  r.gated = cfg_.has_confidence_threshold || cfg_.has_margin_threshold;
  if (samples_seen_ < static_cast<uint32_t>(kWindow)) {
    r.status = Status::kDegenerate;
    mark_reject(&r);
    return r;
  }

  // Chronological copy + centring (means and max|raw| accumulated in double).
  copy_window(&scratch_);
  double raw_max = 0.0;
  std::array<Real*, kChannels> cols{};
  for (int c = 0; c < kChannels; ++c) {
    cols[c] = scratch_[c].data();
    double sum = 0.0;
    for (int i = 0; i < kWindow; ++i) {
      const double v = static_cast<double>(scratch_[c][i]);
      sum += v;
      raw_max = std::max(raw_max, std::fabs(v));
    }
    const Real mean = static_cast<Real>(sum / kWindow);
    for (int i = 0; i < kWindow; ++i) scratch_[c][i] -= mean;
  }
  const int rx = orthonormal_basis(cols.data(), kChannels, raw_max, cfg_.rank_tol);
  if (rx == 0) {
    r.status = Status::kDegenerate;
    mark_reject(&r);
    return r;
  }

  constexpr int N = (kChannels > kRefRows) ? kChannels : kRefRows;
  std::array<Real, kTargets> scores{};
  for (int t = 0; t < kTargets; ++t) {
    const int ry = qy_rank_[t];
    if (ry == 0) {
      r.status = Status::kDegenerate;
      mark_reject(&r);
      return r;
    }
    // M = Qx^T Qy (rx x ry). Jacobi on whichever orientation has fewer columns.
    detail::Small<Real, N> m{};
    for (int i = 0; i < rx; ++i) {
      for (int j = 0; j < ry; ++j) {
        const Real* a = scratch_[i].data();
        const Real* b = qy_[t][j].data();
        Real acc = 0;
        for (int k = 0; k < kWindow; ++k) acc += a[k] * b[k];
        if (rx >= ry) {
          m[i][j] = acc;
        } else {
          m[j][i] = acc;
        }
      }
    }
    const int rows = rx >= ry ? rx : ry;
    const int cols_m = rx >= ry ? ry : rx;
    std::array<Real, N> sv{};
    detail::Small<Real, N> u{};
    detail::jacobi_svd<Real, N>(m, rows, cols_m, &sv, &u);
    r.n_correlations[t] = cols_m;
    for (int k = 0; k < cols_m && k < kMaxCorr; ++k) {
      r.all_correlations[t][k] = std::min(Real(1), std::max(Real(0), sv[k]));
    }
    scores[t] = r.all_correlations[t][0];
  }
  r.status = Status::kOk;
  decide(scores, cfg_, &r);
  return r;
}

}  // namespace synapsys
