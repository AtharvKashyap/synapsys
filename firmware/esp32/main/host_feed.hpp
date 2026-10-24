// Host feed: the Mac streams recorded EEG to the board over the console port, the board
// runs the unchanged decoder on it and prints its results. Proves the on-chip processing on
// real brain data (e.g. the Wang benchmark) without an ADS1299. Driven by
// scripts/esp32_benchmark.py.
//
// Protocol (host -> board), one ASCII line each, '\n'-terminated:
//   B <id>                start stream <id> (uint32): the decoder and re-arm are reset
//   S <v0> <v1> <v2> <v3> one sample, volts, config channel order (send %.17g: exact doubles)
//   E                     evaluate the window ending at the current sample now (off-schedule)
//   Y <n>                 sync: the board answers SSVEP {"event":"ack","n":<n>,...} after
//                         every earlier line has been processed and its output printed
// The console driver drops bytes when its RX buffer (kRxBufferBytes) is full, so the host
// must keep less than that in flight: send a block of lines, then Y, then wait for the ack.
#pragma once

#include "app_types.hpp"

namespace host_feed {

inline constexpr int kRxBufferBytes = 16384;

bool start(const app::Context& ctx);

}  // namespace host_feed
