// Host micro-benchmark (reference only; the ESP32-S3 numbers are measured in Phase 5).
// Prints microseconds per push (filter + ring buffer) and per decision (3x CCA).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>

#include "synapsys/config_v1.hpp"
#include "synapsys/decoder.hpp"

using namespace synapsys;
using Clock = std::chrono::steady_clock;

int main() {
  auto dec = std::make_unique<Decoder>(config_synapsys_v1());
  const int n = 250 * 600;  // 10 minutes of data
  volatile Real sink = 0;
  uint32_t lcg = 1u;
  auto sample = [&](int i, Real* x) {
    for (int ch = 0; ch < kChannels; ++ch) {
      lcg = lcg * 1664525u + 1013904223u;
      x[ch] = static_cast<Real>(0.01 + 1e-5 * std::sin(0.3 * i + ch) +
                                1e-6 * (static_cast<double>(lcg >> 8) / 16777216.0 - 0.5));
    }
  };
  // Pre-generate input so the timing covers only push().
  auto input = std::make_unique<Real[]>(static_cast<size_t>(n) * kChannels);
  for (int i = 0; i < n; ++i) sample(i, &input[static_cast<size_t>(i) * kChannels]);

  // Filter-only cost: a decoder with a huge hop never decides after the first window.
  Config no_dec = config_synapsys_v1();
  no_dec.hop_samples = 1 << 30;
  auto filt = std::make_unique<Decoder>(no_dec);
  auto t0 = Clock::now();
  for (int i = 0; i < n; ++i) filt->push(&input[static_cast<size_t>(i) * kChannels], nullptr);
  auto t1 = Clock::now();
  sink = sink + filt->last_filtered()[0];
  const double us_push = std::chrono::duration<double, std::micro>(t1 - t0).count() / n;

  // Decision cost: evaluate() on a full window, repeated.
  for (int i = 0; i < kWindow; ++i) dec->push(&input[static_cast<size_t>(i) * kChannels], nullptr);
  const int reps = 2000;
  t0 = Clock::now();
  for (int r = 0; r < reps; ++r) {
    WindowResult res = dec->evaluate();
    sink = sink + res.scores[0];
  }
  t1 = Clock::now();
  const double us_dec = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;

  std::printf("[host bench, %s] %.3f us per push (filter, 4 ch x %d sections), "
              "%.1f us per decision (3 targets CCA); sizeof(Decoder) = %zu bytes\n",
              sizeof(Real) == sizeof(double) ? "double" : "float", us_push,
              config_synapsys_v1().n_sections, us_dec, sizeof(Decoder));
  (void)sink;
  return 0;
}
