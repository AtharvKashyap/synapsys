#!/usr/bin/env bash
# Build the laptop Bluetooth relay into live_demo/build/ble_relay (macOS, Xcode command-line tools).
set -euo pipefail
DEMO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$DEMO/build"
swiftc -O "$DEMO/relay/ble_relay.swift" -o "$DEMO/build/ble_relay" \
  -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker "$DEMO/relay/Info.plist"
echo "built $DEMO/build/ble_relay"
