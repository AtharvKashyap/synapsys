#!/usr/bin/env bash
# Compile CommandLogic.swift + main.swift for the Mac and run them (no simulator needed).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/../../build/ios_logic_tests"
mkdir -p "$OUT"
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
xcrun swiftc -O -o "$OUT/logic_tests" "$HERE/../SynapSysDemo/CommandLogic.swift" "$HERE/main.swift"
"$OUT/logic_tests"
