// ADS1299 acquisition: DRDY falling edge (ISR) -> task notification -> SPI frame read ->
// volts in decoder channel order -> sample queue. Runs pinned to core 1.
#pragma once

#include "app_types.hpp"

namespace acquisition {

// Spawns the acquisition task (core 1). It initialises the ADS1299, then streams.
// Initialisation outcome is reported through ctx.source->running / ->failed.
// test_signal = true selects the ADS1299 internal test signal on the used channels.
bool start(const app::Context& ctx, bool test_signal);

}  // namespace acquisition
