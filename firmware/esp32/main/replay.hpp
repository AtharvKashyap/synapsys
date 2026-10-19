// Fixture replay: streams the embedded Python-reference fixtures through the real sample
// queue at 250 Hz (esp_timer) and checks every decision point against the expected
// outputs (|score err| < 1e-3, |margin err| < 1e-3, exact winner and decision; degenerate
// points must report Status::kDegenerate). Needs no hardware besides the ESP32-S3.
#pragma once

#include <cstdint>

#include "app_types.hpp"

namespace replay {

inline constexpr double kScoreTol = 1e-3;

// Validates the embedded fixtures against the decoder config, then spawns the replay task
// (core 1) and its esp_timer. Returns false if the task/timer could not be created.
bool start(const app::Context& ctx);

int num_streams();
const char* stream_name(uint8_t stream);

// ---- Self-check, called from the decoder task only (not thread-safe by design). ----
void begin_stream(uint8_t stream);
// Compares one decision point; fills the check fields of *ev (type kDecision).
void check_point(uint8_t stream, uint32_t t, const synapsys::WindowResult& r,
                 const synapsys::Config& cfg, app::Event* ev);
// Finalises a stream (counts missing points); fills *ev as kStreamSummary.
void end_stream(uint8_t stream, uint16_t pass, app::Event* ev);
// Finalises a pass; fills *ev as kPassSummary and resets the pass counters.
void end_pass(uint16_t pass, app::Event* ev);

}  // namespace replay
