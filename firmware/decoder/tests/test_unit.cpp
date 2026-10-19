// Host unit tests for the decoder core (built once per precision).
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "linalg.hpp"
#include "synapsys/config_v1.hpp"
#include "synapsys/decoder.hpp"

using namespace synapsys;

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond)                                                      \
  do {                                                                   \
    ++g_checks;                                                          \
    if (!(cond)) {                                                       \
      ++g_failures;                                                      \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
    }                                                                    \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                                          \
  do {                                                                                 \
    ++g_checks;                                                                        \
    const double va_ = static_cast<double>(a), vb_ = static_cast<double>(b);           \
    if (!(std::fabs(va_ - vb_) <= (tol))) {                                            \
      ++g_failures;                                                                    \
      std::printf("FAIL %s:%d: %s = %.17g vs %s = %.17g (tol %g)\n", __FILE__, __LINE__, \
                  #a, va_, #b, vb_, static_cast<double>(tol));                         \
    }                                                                                  \
  } while (0)

constexpr double kRealTol = sizeof(Real) == sizeof(double) ? 1e-12 : 1e-5;

Config passthrough_config() {
  Config c = config_synapsys_v1();
  c.n_sections = 0;
  c.remove_dc = false;
  return c;
}

// ---------------------------------------------------------------------------------------
// Biquad: hand-computed impulse response of one section, b = [1/2, 1/4, 1/8],
// a = [1, -1/2, 1/4]: y[n] = x/2 + x[n-1]/4 + x[n-2]/8 + y[n-1]/2 - y[n-2]/4.
// All values are exact in binary floating point.
void test_biquad_hand_computed() {
  Config c = config_synapsys_v1();
  c.n_sections = 1;
  c.sos[0] = {{0.5, 0.25, 0.125, 1.0, -0.5, 0.25}};
  c.remove_dc = false;
  auto d = std::make_unique<Decoder>(c);
  const double expected[] = {0.5, 0.5, 0.25, 0.0, -0.0625, -0.03125, 0.0, 0.0078125};
  for (int n = 0; n < 8; ++n) {
    Real x[kChannels] = {Real(n == 0 ? 1 : 0), Real(n == 0 ? 2 : 0), 0, Real(n == 0 ? -1 : 0)};
    d->push(x, nullptr);
    CHECK(d->last_filtered()[0] == Real(expected[n]));
    CHECK(d->last_filtered()[1] == Real(2 * expected[n]));
    CHECK(d->last_filtered()[2] == Real(0));
    CHECK(d->last_filtered()[3] == Real(-expected[n]));
  }
}

// DC removal: the first sample of each channel is subtracted from the whole stream.
void test_dc_removal() {
  Config c = config_synapsys_v1();
  auto d = std::make_unique<Decoder>(c);
  for (int n = 0; n < 50; ++n) {
    Real x[kChannels] = {Real(0.3), Real(-0.25), Real(1e-3), Real(5)};
    d->push(x, nullptr);
    for (int ch = 0; ch < kChannels; ++ch) CHECK(d->last_filtered()[ch] == Real(0));
  }
  Config nodc = c;
  nodc.remove_dc = false;
  auto e = std::make_unique<Decoder>(nodc);
  Real x[kChannels] = {Real(0.3), Real(0.3), Real(0.3), Real(0.3)};
  e->push(x, nullptr);
  CHECK(e->last_filtered()[0] != Real(0));  // no subtraction -> start-up transient
}

// Streaming cascade == batch DF2T reference (sosfilt) on the whole array, computed
// in double here; and reset() / interleaved evaluate() calls do not change the stream.
std::vector<std::array<double, kChannels>> test_signal(int n) {
  std::vector<std::array<double, kChannels>> s(static_cast<size_t>(n));
  uint32_t lcg = 12345u;
  for (int i = 0; i < n; ++i) {
    for (int ch = 0; ch < kChannels; ++ch) {
      lcg = lcg * 1664525u + 1013904223u;
      const double noise = (static_cast<double>(lcg >> 8) / 16777216.0 - 0.5) * 2e-5;
      s[static_cast<size_t>(i)][static_cast<size_t>(ch)] =
          0.01 * (ch + 1) + 1e-5 * std::sin(2 * 3.14159265358979 * 12.0 * i / 250.0 + ch) +
          noise;
    }
  }
  return s;
}

void test_streaming_equals_batch() {
  const Config c = config_synapsys_v1();
  const int n = 1500;
  auto sig = test_signal(n);
  // Batch reference in double.
  std::vector<std::array<double, kChannels>> ref(sig.size());
  for (int ch = 0; ch < kChannels; ++ch) {
    double z[kMaxSections][2] = {};
    const double off = sig[0][static_cast<size_t>(ch)];
    for (int i = 0; i < n; ++i) {
      double x = sig[static_cast<size_t>(i)][static_cast<size_t>(ch)] - off;
      for (int s = 0; s < c.n_sections; ++s) {
        const auto& k = c.sos[s];
        const double y = k[0] * x + z[s][0];
        z[s][0] = k[1] * x - k[4] * y + z[s][1];
        z[s][1] = k[2] * x - k[5] * y;
        x = y;
      }
      ref[static_cast<size_t>(i)][static_cast<size_t>(ch)] = x;
    }
  }
  auto a = std::make_unique<Decoder>(c);
  auto b = std::make_unique<Decoder>(c);
  // b: run a different prefix, reset, then the same stream with evaluate() interleaved.
  for (int i = 0; i < 700; ++i) {
    Real junk[kChannels] = {Real(i), Real(-i), Real(0.5), Real(1)};
    b->push(junk, nullptr);
  }
  b->reset();
  CHECK(b->samples_seen() == 0);
  double max_err = 0.0, max_ref = 0.0;
  int decisions = 0;
  for (int i = 0; i < n; ++i) {
    Real x[kChannels];
    for (int ch = 0; ch < kChannels; ++ch) {
      x[ch] = static_cast<Real>(sig[static_cast<size_t>(i)][static_cast<size_t>(ch)]);
    }
    WindowResult ra, rb;
    const bool da = a->push(x, &ra);
    const bool db = b->push(x, &rb);
    if (b->samples_seen() >= kWindow && i % 37 == 0) (void)b->evaluate();
    CHECK(da == db);
    for (int ch = 0; ch < kChannels; ++ch) {
      CHECK(a->last_filtered()[ch] == b->last_filtered()[ch]);
      const double r = ref[static_cast<size_t>(i)][static_cast<size_t>(ch)];
      max_err = std::fmax(max_err, std::fabs(static_cast<double>(a->last_filtered()[ch]) - r));
      max_ref = std::fmax(max_ref, std::fabs(r));
    }
    if (da) {
      ++decisions;
      for (int t = 0; t < kTargets; ++t) CHECK(ra.scores[t] == rb.scores[t]);
      CHECK(ra.winner == rb.winner);
    }
  }
  CHECK(decisions == 8);
  // double: bitwise-identical operation order -> essentially exact; float: input rounding.
  CHECK(max_err <= (sizeof(Real) == sizeof(double) ? 1e-18 : 1e-3 * max_ref));
}

// Ring buffer: window is the last kWindow samples in chronological order across wraps.
void test_ring_wrap() {
  auto d = std::make_unique<Decoder>(passthrough_config());
  auto w = std::make_unique<std::array<std::array<Real, kWindow>, kChannels>>();
  for (int i = 0; i < 2000; ++i) {
    Real x[kChannels];
    for (int ch = 0; ch < kChannels; ++ch) x[ch] = Real(i + 10000 * ch);
    d->push(x, nullptr);
    const int seen = i + 1;
    if (seen == 625 || seen == 626 || seen == 1249 || seen == 1250 || seen == 2000) {
      d->copy_window(w.get());
      bool ok = true;
      for (int ch = 0; ch < kChannels; ++ch) {
        for (int k = 0; k < kWindow; ++k) {
          ok = ok && (*w)[ch][k] == Real(seen - kWindow + k + 10000 * ch);
        }
      }
      CHECK(ok);
    }
  }
}

// Decision schedule: true exactly at 625, 750, ...; evaluate() before a full window
// reports kDegenerate / kReject.
void test_schedule() {
  const Config c = config_synapsys_v1();
  auto d = std::make_unique<Decoder>(c);
  auto sig = test_signal(1500);
  WindowResult early = d->evaluate();
  CHECK(early.status == Status::kDegenerate);
  CHECK(early.decision == kReject);
  for (int i = 0; i < 1500; ++i) {
    Real x[kChannels];
    for (int ch = 0; ch < kChannels; ++ch) {
      x[ch] = static_cast<Real>(sig[static_cast<size_t>(i)][static_cast<size_t>(ch)]);
    }
    WindowResult r;
    const bool due = d->push(x, &r);
    const int seen = i + 1;
    const bool expect = seen >= kWindow && (seen - kWindow) % 125 == 0;
    CHECK(due == expect);
    if (due) CHECK(r.window_end_sample == static_cast<uint32_t>(seen));
  }
}

// ---------------------------------------------------------------------------------------
// Householder QR on known matrices (double and Real).
template <class T>
void test_householder(double tol) {
  const int m = 6, p = 3;
  const double a0[3][6] = {{1, 2, 3, 4, 5, 6}, {2, -1, 0, 3, 1, -2}, {0.5, 0.25, -1, 2, 0, 1}};
  T a[3][6];
  for (int j = 0; j < p; ++j)
    for (int i = 0; i < m; ++i) a[j][i] = T(a0[j][i]);
  T* cols[3] = {a[0], a[1], a[2]};
  T tau[3];
  detail::householder_qr(cols, m, p, tau);
  double r[3][3] = {};
  for (int j = 0; j < p; ++j)
    for (int i = 0; i <= j; ++i) r[i][j] = static_cast<double>(a[j][i]);
  detail::form_q(cols, m, p, tau);
  // Q^T Q = I and Q R = A.
  for (int j = 0; j < p; ++j) {
    for (int k = 0; k < p; ++k) {
      double dot = 0;
      for (int i = 0; i < m; ++i) dot += double(a[j][i]) * double(a[k][i]);
      CHECK_NEAR(dot, j == k ? 1.0 : 0.0, tol);
    }
    for (int i = 0; i < m; ++i) {
      double qr = 0;
      for (int k = 0; k <= j; ++k) qr += double(a[k][i]) * r[k][j];
      CHECK_NEAR(qr, a0[j][i], tol * 10);
    }
  }
  // |R| diagonal: |r00| = ||a_0|| = sqrt(91).
  CHECK_NEAR(std::fabs(r[0][0]), std::sqrt(91.0), tol * 10);

  // Dead (all-zero) middle column: tau = 0, Q still orthonormal, R column zero.
  T b[3][6];
  for (int i = 0; i < m; ++i) {
    b[0][i] = T(a0[0][i]);
    b[1][i] = T(0);
    b[2][i] = T(a0[2][i]);
  }
  T* bc[3] = {b[0], b[1], b[2]};
  detail::householder_qr(bc, m, p, tau);
  CHECK(tau[1] == T(0));
  CHECK(b[1][0] == T(0) && b[1][1] == T(0));
  detail::form_q(bc, m, p, tau);
  for (int j = 0; j < p; ++j) {
    for (int k = 0; k < p; ++k) {
      double dot = 0;
      for (int i = 0; i < m; ++i) dot += double(b[j][i]) * double(b[k][i]);
      CHECK_NEAR(dot, j == k ? 1.0 : 0.0, tol);
      CHECK(std::isfinite(dot));
    }
  }
}

template <class T>
void test_jacobi(double tol) {
  constexpr int N = 4;
  // [[3, 0], [4, 5]]: singular values 3*sqrt(5), sqrt(5).
  detail::Small<T, N> a{};
  a[0][0] = 3;
  a[1][0] = 4;
  a[1][1] = 5;
  std::array<T, N> s{};
  detail::Small<T, N> u{};
  detail::jacobi_svd<T, N>(a, 2, 2, &s, &u);
  CHECK_NEAR(s[0], 3 * std::sqrt(5.0), tol * 10);
  CHECK_NEAR(s[1], std::sqrt(5.0), tol * 10);
  // A^T u_k has norm s_k, u orthonormal.
  for (int k = 0; k < 2; ++k) {
    double v0 = double(a[0][0]) * double(u[0][k]) + double(a[1][0]) * double(u[1][k]);
    double v1 = double(a[0][1]) * double(u[0][k]) + double(a[1][1]) * double(u[1][k]);
    CHECK_NEAR(std::sqrt(v0 * v0 + v1 * v1), s[k], tol * 10);
  }
  CHECK_NEAR(double(u[0][0]) * double(u[0][1]) + double(u[1][0]) * double(u[1][1]), 0.0, tol);

  // Permuted diagonal -> sorted descending.
  detail::Small<T, N> d{};
  d[0][1] = T(0.25);
  d[1][3] = T(2);
  d[2][0] = T(1);
  d[3][2] = T(0.5);
  detail::jacobi_svd<T, N>(d, 4, 4, &s, &u);
  CHECK(s[0] == T(2) && s[1] == T(1) && s[2] == T(0.5) && s[3] == T(0.25));

  // Rank-deficient 3x3 (col2 = col0 + col1): smallest singular value ~ 0, others finite.
  detail::Small<T, N> r{};
  const double rv[3][2] = {{1, 2}, {3, -1}, {0.5, 4}};
  for (int i = 0; i < 3; ++i) {
    r[i][0] = T(rv[i][0]);
    r[i][1] = T(rv[i][1]);
    r[i][2] = T(rv[i][0] + rv[i][1]);
  }
  detail::jacobi_svd<T, N>(r, 3, 3, &s, &u);
  CHECK(s[0] >= s[1] && s[1] >= s[2]);
  CHECK_NEAR(s[2], 0.0, tol * 100);
  // Frobenius norm preserved: sum s^2 = sum a^2.
  double fro = 0;
  for (int i = 0; i < 3; ++i) fro += rv[i][0] * rv[i][0] + rv[i][1] * rv[i][1] +
                                     (rv[i][0] + rv[i][1]) * (rv[i][0] + rv[i][1]);
  CHECK_NEAR(double(s[0]) * double(s[0]) + double(s[1]) * double(s[1]) +
                 double(s[2]) * double(s[2]),
             fro, tol * 1000);
}

// ---------------------------------------------------------------------------------------
void test_tie_break() {
  const Config c = config_synapsys_v1();
  WindowResult r;
  decide({Real(0.5), Real(0.5), Real(0.1)}, c, &r);
  CHECK(r.winner == 0 && r.margin == Real(0) && r.decision == 0 && !r.gated && r.accepted);
  decide({Real(0.1), Real(0.5), Real(0.5)}, c, &r);
  CHECK(r.winner == 1 && r.margin == Real(0));
  decide({Real(0.3), Real(0.3), Real(0.3)}, c, &r);
  CHECK(r.winner == 0);
  decide({Real(0.1), Real(0.2), Real(0.5)}, c, &r);
  CHECK(r.winner == 2 && r.confidence == Real(0.5));
  CHECK_NEAR(r.margin, 0.3, kRealTol * 10);
}

void test_gating() {
  Config c = config_synapsys_v1();
  WindowResult r;
  const std::array<Real, kTargets> s = {Real(0.25), Real(0.5), Real(0.375)};  // margin 0.125
  c.has_confidence_threshold = true;
  c.confidence_threshold = 0.5;  // score == T -> accepted (>=)
  decide(s, c, &r);
  CHECK(r.gated && r.accepted && r.decision == 1);
  c.confidence_threshold = 0.5000001;
  decide(s, c, &r);
  CHECK(r.gated && !r.accepted && r.decision == kReject && r.winner == 1);
  c.has_confidence_threshold = false;
  c.has_margin_threshold = true;
  c.margin_threshold = 0.125;  // margin == M -> accepted
  decide(s, c, &r);
  CHECK(r.gated && r.accepted && r.decision == 1);
  c.margin_threshold = 0.2;
  decide(s, c, &r);
  CHECK(!r.accepted && r.decision == kReject);
  c.has_confidence_threshold = true;
  c.confidence_threshold = 0.1;  // T passes, M fails -> reject
  decide(s, c, &r);
  CHECK(!r.accepted && r.decision == kReject);
  c.margin_threshold = 0.1;  // both pass
  decide(s, c, &r);
  CHECK(r.accepted && r.decision == 1);
}

// ---------------------------------------------------------------------------------------
// End-to-end CCA sanity: a pure 12 Hz sinusoid (no filter) scores ~1 for NO; flat input
// is degenerate (never NaN).
void test_cca_end_to_end() {
  auto d = std::make_unique<Decoder>(passthrough_config());
  WindowResult r;
  for (int i = 0; i < kWindow; ++i) {
    Real x[kChannels];
    for (int ch = 0; ch < kChannels; ++ch) {
      x[ch] = static_cast<Real>(1e-5 * std::sin(2 * 3.14159265358979 * 12.0 * i / 250.0 + ch) +
                                1e-7 * ch * std::cos(0.37 * i * (ch + 1)));
    }
    d->push(x, &r);
  }
  CHECK(r.status == Status::kOk);
  CHECK(r.winner == 1);
  CHECK(r.scores[1] > Real(0.99) && r.scores[1] <= Real(1));
  CHECK(r.n_correlations[0] == 4);

  auto f = std::make_unique<Decoder>(config_synapsys_v1());
  for (int i = 0; i < kWindow; ++i) {
    Real x[kChannels] = {Real(0.01), Real(0.01), Real(0.01), Real(0.01)};
    f->push(x, &r);
  }
  CHECK(r.status == Status::kDegenerate);
  CHECK(r.decision == kReject && r.winner == kReject);
  for (int t = 0; t < kTargets; ++t) CHECK(std::isfinite(static_cast<double>(r.scores[t])));
}

}  // namespace

int main() {
  test_biquad_hand_computed();
  test_dc_removal();
  test_streaming_equals_batch();
  test_ring_wrap();
  test_schedule();
  test_householder<double>(1e-13);
  test_householder<float>(2e-6);
  test_jacobi<double>(1e-13);
  test_jacobi<float>(2e-6);
  test_tie_break();
  test_gating();
  test_cca_end_to_end();
  std::printf("[%s] %d checks, %d failures\n",
              sizeof(Real) == sizeof(double) ? "double" : "float", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
