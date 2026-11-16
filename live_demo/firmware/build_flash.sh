#!/usr/bin/env bash
# Build the live-demo firmware (unchanged firmware/esp32 sources + the demo overlay) into
# live_demo/build/esp32. With a port argument it also flashes; add --monitor to watch the boot
# log (close the monitor, Ctrl-], before starting the stage app: two readers on one port
# steal each other's bytes). PORT = the board's COM connector, not its native USB port.
#   live_demo/firmware/build_flash.sh                                  # build only
#   live_demo/firmware/build_flash.sh /dev/cu.usbmodemXXXX             # build + flash
#   live_demo/firmware/build_flash.sh /dev/cu.usbmodemXXXX --monitor   # + boot log
#   NO_BT=1 live_demo/firmware/build_flash.sh /dev/cu.usbmodemXXXX     # Bluetooth off (build/esp32_nobt)
set -euo pipefail
DEMO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$DEMO/../firmware/esp32"
OUT="$DEMO/build/esp32"
DEFAULTS="sdkconfig.defaults;$DEMO/firmware/sdkconfig.demo.defaults"
OVERLAYS=("$SRC/sdkconfig.defaults" "$DEMO/firmware/sdkconfig.demo.defaults")
if [[ -n "${NO_BT:-}" ]]; then
  OUT="$DEMO/build/esp32_nobt"
  DEFAULTS="$DEFAULTS;$DEMO/firmware/sdkconfig.nobt.defaults"
  OVERLAYS+=("$DEMO/firmware/sdkconfig.nobt.defaults")
fi
# shellcheck disable=SC1091
. "${IDF_PATH:-$HOME/esp/esp-idf}/export.sh" >/dev/null
mkdir -p "$OUT"
# Defaults only apply to a fresh sdkconfig, so each overlay version gets its own sdkconfig
# file (named by the overlay's hash). Nothing is ever deleted.
HASH="$(cat "${OVERLAYS[@]}" | shasum | cut -c1-8)"
SDKCFG="$OUT/sdkconfig.$HASH"
cd "$SRC"
ARGS=(-B "$OUT" -D SDKCONFIG="$SDKCFG"
      -D SDKCONFIG_DEFAULTS="$DEFAULTS")
idf.py "${ARGS[@]}" build
if [[ $# -ge 1 ]]; then
  idf.py "${ARGS[@]}" -p "$1" -b "${FLASH_BAUD:-921600}" flash
  if [[ "${2:-}" == "--monitor" ]]; then
    idf.py "${ARGS[@]}" -p "$1" monitor -b 921600
  fi
fi
