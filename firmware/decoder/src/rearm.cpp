#include "synapsys/rearm.hpp"

namespace synapsys {

Rearm::Rearm(const RearmConfig& cfg) : cfg_(cfg) {
  valid_ = cfg_.confirm_count >= 1 && cfg_.release_count >= 0;
  reset();
}

void Rearm::reset() {
  state_ = State::kArmed;
  label_ = kReject;
  count_ = 0;
  released_ = 0;
  emitted_at_ = 0;
}

const char* Rearm::state_name() const {
  switch (state_) {
    case State::kArmed: return "ARMED";
    case State::kCandidate: return "CANDIDATE";
    case State::kRefractory: return "REFRACTORY";
  }
  return "?";
}

int Rearm::update(const WindowResult& r) {
  const int d = (r.status == Status::kOk) ? r.decision : kReject;
  return update(d, r.window_end_sample);
}

int Rearm::update(int decision, uint32_t t) {
  if (!valid_) return kReject;
  switch (state_) {
    case State::kArmed:
      if (decision == kReject) return kReject;
      label_ = decision;
      count_ = 1;
      state_ = State::kCandidate;
      break;
    case State::kCandidate:
      if (decision == kReject) {
        state_ = State::kArmed;
        label_ = kReject;
        count_ = 0;
        return kReject;
      }
      if (decision == label_) {
        ++count_;
      } else {
        label_ = decision;
        count_ = 1;
      }
      break;
    case State::kRefractory:
      if (decision == label_) {
        released_ = 0;
      } else {
        ++released_;
      }
      if (t - emitted_at_ >= cfg_.refractory_samples && released_ >= cfg_.release_count) {
        state_ = State::kArmed;
        label_ = kReject;
        count_ = 0;
        released_ = 0;
      }
      return kReject;
  }
  if (count_ >= cfg_.confirm_count) {  // reached from ARMED (K == 1) or CANDIDATE
    const int emit = label_;
    state_ = State::kRefractory;
    emitted_at_ = t;
    released_ = 0;
    return emit;
  }
  return kReject;
}

}  // namespace synapsys
