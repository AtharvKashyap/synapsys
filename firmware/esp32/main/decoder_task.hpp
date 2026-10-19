// Decoder task (core 0): pops samples, runs synapsys::Decoder + Rearm, posts events
// (decision points, emitted commands, timing, replay self-check) to the output task.
#pragma once

#include "app_types.hpp"
#include "synapsys/rearm.hpp"

namespace decoder_task {

// Constructs the statically allocated Decoder (~50 kB, .bss) and Rearm from *ctx.cfg and
// rearm_cfg, then starts the task. Call once from app_main.
bool start(const app::Context& ctx, const synapsys::RearmConfig& rearm_cfg, bool replay_mode);

}  // namespace decoder_task
