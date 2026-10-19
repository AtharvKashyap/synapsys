#!/usr/bin/env bash
# Run the real ESP32-S3 firmware in Espressif's QEMU (no board needed) and check that the
# fixture-replay self-check passes. BLE is disabled (radio not emulated) and the console is
# the emulated UART; everything else is the production code path.
#
# Prereqs: ESP-IDF v5.3 (~/esp/esp-idf), `idf_tools.py install qemu-xtensa`,
#          `brew install libgcrypt glib pixman sdl2 libslirp`.
# Usage:   firmware/tools/run_qemu.sh [work_dir]     (exit 0 = replay PASS)
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${1:-${TMPDIR:-/tmp}/synapsys-qemu}"
mkdir -p "$WORK"
# shellcheck disable=SC1091
. "${IDF_PATH:-$HOME/esp/esp-idf}/export.sh" >/dev/null
cd "$HERE/esp32"
rm -f "$WORK/sdkconfig"
idf.py -B "$WORK/build" -D SDKCONFIG="$WORK/sdkconfig" \
  -D SDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.qemu.defaults" build >"$WORK/build.log"
(cd "$WORK/build" && esptool.py --chip esp32s3 merge_bin --fill-flash-size 4MB \
  -o "$WORK/flash.bin" @flash_args >/dev/null)
QEMU="$(ls "$HOME"/.espressif/tools/qemu-xtensa/*/qemu/bin/qemu-system-xtensa | head -1)"
rm -f "$WORK/serial.log"
"$QEMU" -nographic -machine esp32s3 -drive file="$WORK/flash.bin",if=mtd,format=raw \
  -serial file:"$WORK/serial.log" -monitor none >/dev/null 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
  if grep -q '"event":"replay_summary"' "$WORK/serial.log" 2>/dev/null; then break; fi
  sleep 2
done
grep '"event":"replay_stream"\|"event":"replay_summary"\|"event":"command"' "$WORK/serial.log" || true
grep -q '"event":"replay_summary".*"result":"PASS"' "$WORK/serial.log"
