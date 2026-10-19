// Output task (core 0, low priority): turns decoder events into
//   * one grep-able JSON line per event on the console, prefixed "SSVEP "
//   * BLE notifications (Command on emitted commands, Debug per decision point)
//   * a periodic health line + BLE Status value
//   * optional status LED.
// Keeping formatting/IO here means a slow console or BLE link can never stall the decoder.
#pragma once

#include "app_types.hpp"

namespace output {

inline constexpr const char* kPrefix = "SSVEP ";

bool start(const app::Context& ctx, bool replay_mode, bool ble_enabled);

}  // namespace output
