// Command re-arming: turns per-window decisions into discrete command events.
//
// Exact port of the Python reference src/synapsys_decoder/rearm.py (see its state table).
// Integer-only logic; time is the window_end_sample of each decision point.
//
//   ARMED       REJECT -> ARMED; label L -> CANDIDATE(L,1) (K == 1: emit L -> REFRACTORY)
//   CANDIDATE   same L -> k+1, emit at K -> REFRACTORY(L, t, 0); other M -> CANDIDATE(M,1);
//               REJECT -> ARMED
//   REFRACTORY  L -> released = 0; REJECT or != L -> released += 1; then if
//               t - emitted_at >= R and released >= N -> ARMED. Never emits. The point that
//               completes the release does not start a candidate.
#pragma once

#include <cstdint>

#include "synapsys/decoder.hpp"

namespace synapsys {

struct RearmConfig {
  int confirm_count = 2;            // K >= 1
  uint32_t refractory_samples = 625;  // R
  int release_count = 2;            // N >= 0
};

class Rearm {
 public:
  enum class State : uint8_t { kArmed, kCandidate, kRefractory };

  explicit Rearm(const RearmConfig& cfg);

  // Feed one decision point. Returns the target index to EMIT, or kReject for no event.
  // Degenerate / rejected windows count as REJECT.
  int update(const WindowResult& r);
  int update(int decision, uint32_t window_end_sample);

  void reset();
  State state() const { return state_; }
  const char* state_name() const;
  // False if the config was invalid (K < 1 or N < 0); update() then never emits.
  bool valid() const { return valid_; }

 private:
  RearmConfig cfg_;
  bool valid_ = true;
  State state_ = State::kArmed;
  int label_ = kReject;
  int count_ = 0;
  int released_ = 0;
  uint32_t emitted_at_ = 0;
};

}  // namespace synapsys
