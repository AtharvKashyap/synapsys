// Types shared between the sample sources, the decoder task and the output task.
#pragma once

#include <atomic>
#include <cstdint>

#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "synapsys/decoder.hpp"

namespace app {

// One multichannel sample in decoder channel order (O1, Oz, O2, POz), volts.
// double: the decoder removes the first-sample DC offset in double before rounding to
// float32 (Decoder::push_f64); electrode offsets of ~100 mV would otherwise cost precision.
struct Sample {
  double v[synapsys::kChannels];
  uint32_t seq;     // producer sequence number (0, 1, 2, ... since source start)
  uint8_t flags;    // SampleFlag bits
  uint8_t stream;   // replay: fixture index; live: 0
  uint16_t pass;    // replay: pass number (0-based); live: 0
};

enum SampleFlag : uint8_t {
  kFlagStreamStart = 1u << 0,  // reset decoder + re-arm before pushing this sample
  kFlagStreamEnd = 1u << 1,    // last sample of a replay stream: finalise its self-check
  kFlagPassEnd = 1u << 2,      // last sample of a replay pass: print the summary
};

// Counters written by the producer (acquisition or replay), read by the output task.
struct SourceStats {
  std::atomic<uint32_t> produced{0};       // samples successfully queued
  std::atomic<uint32_t> dropped{0};        // queue full -> sample lost
  std::atomic<uint32_t> drdy_timeouts{0};  // no DRDY within the timeout
  std::atomic<uint32_t> missed_drdy{0};    // >1 DRDY edges before the task ran
  std::atomic<uint32_t> spi_errors{0};     // spi transaction failures
  std::atomic<uint32_t> bad_status{0};     // status word top nibble != 0xC
  std::atomic<uint32_t> queue_high_water{0};
  std::atomic<bool> running{false};        // source initialised and streaming
  std::atomic<bool> failed{false};         // source failed to initialise
};

// Counters written by the decoder task, read by the output task.
struct DecoderStats {
  std::atomic<uint32_t> samples{0};
  std::atomic<uint32_t> decisions{0};
  std::atomic<uint32_t> degenerate{0};
  std::atomic<uint32_t> commands{0};
  std::atomic<uint32_t> dec_us_min{UINT32_MAX};  // lifetime, per decision point
  std::atomic<uint32_t> dec_us_max{0};
  std::atomic<uint32_t> dec_us_avg{0};   // lifetime mean (decoder task keeps 64-bit sums)
  std::atomic<uint32_t> filt_ns_avg{0};  // mean time of a non-decision push (filter only), ns
  std::atomic<uint32_t> events_dropped{0};  // output queue full
};

enum class EventType : uint8_t {
  kDecision,       // one decision point (+ optional replay self-check verdict)
  kCommand,        // re-arm emitted a command
  kTiming,         // min/avg/max decision compute over the last N decisions
  kStreamSummary,  // replay: one fixture finished
  kPassSummary,    // replay: one pass over all fixtures finished
};

enum class Check : uint8_t { kNone, kPass, kFail, kNear, kUnexpected };

struct Event {
  EventType type;
  uint32_t t;                  // samples since stream start (= window end for decisions)
  uint8_t stream;
  uint16_t pass;
  // kDecision
  synapsys::WindowResult result;
  const char* rearm_state;     // static string from Rearm::state_name()
  uint32_t compute_us;
  // replay self-check (kDecision)
  Check check;
  float max_score_err;
  float margin_err;
  int8_t exp_winner;
  int8_t exp_decision;
  bool exp_degenerate;
  // kCommand
  int8_t target;
  // kTiming
  uint32_t t_min_us, t_avg_us, t_max_us, t_n;
  // kStreamSummary / kPassSummary
  uint32_t n_pass, n_fail, n_near, n_missing, n_unexpected;
};

// Everything the tasks need, created once in app_main (never freed).
struct Context {
  QueueHandle_t samples = nullptr;  // Sample
  QueueHandle_t events = nullptr;   // Event
  SourceStats* source = nullptr;
  DecoderStats* decoder = nullptr;
  const synapsys::Config* cfg = nullptr;
  const char* source_name = "";
};

// Task handles for stack high-water reporting (set once at task creation, read-only after).
inline TaskHandle_t g_source_task = nullptr;
inline TaskHandle_t g_decoder_task = nullptr;
inline TaskHandle_t g_output_task = nullptr;

// Update a running maximum held in an atomic (single writer is typical, but safe anyway).
inline void atomic_max(std::atomic<uint32_t>& a, uint32_t v) {
  uint32_t cur = a.load(std::memory_order_relaxed);
  while (v > cur && !a.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {
  }
}
inline void atomic_min(std::atomic<uint32_t>& a, uint32_t v) {
  uint32_t cur = a.load(std::memory_order_relaxed);
  while (v < cur && !a.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {
  }
}

// Label for a target index (kReject -> "REJECT").
inline const char* label_of(const synapsys::Config& cfg, int idx) {
  if (idx < 0 || idx >= synapsys::kTargets || cfg.labels[idx] == nullptr) return "REJECT";
  return cfg.labels[idx];
}

}  // namespace app
