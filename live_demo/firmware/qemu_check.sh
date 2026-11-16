#!/usr/bin/env bash
# Run the demo firmware (same overlay, BLE off: QEMU has no radio) in Espressif's QEMU and play
# the five scenarios to it over an emulated UART socket. Proves the real C++ firmware, with the
# demo threshold, gives YES / NO / HELP / quiet / quiet. Exit 0 = PASS.
set -euo pipefail
DEMO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$DEMO/../firmware/esp32"
OUT="$DEMO/build/esp32_qemu"
# shellcheck disable=SC1091
. "${IDF_PATH:-$HOME/esp/esp-idf}/export.sh" >/dev/null
mkdir -p "$OUT"
HASH="$(cat "$DEMO/firmware/sdkconfig.demo.defaults" "$DEMO/firmware/sdkconfig.qemu_check.defaults" | shasum | cut -c1-8)"
cd "$SRC"
idf.py -B "$OUT" -D SDKCONFIG="$OUT/sdkconfig.$HASH" \
  -D SDKCONFIG_DEFAULTS="sdkconfig.defaults;$DEMO/firmware/sdkconfig.demo.defaults;$DEMO/firmware/sdkconfig.qemu_check.defaults" \
  build >"$OUT/build.log"
(cd "$OUT" && esptool.py --chip esp32s3 merge_bin --fill-flash-size 4MB -o "$OUT/flash.bin" @flash_args >/dev/null)
QEMU="$(ls "$HOME"/.espressif/tools/qemu-xtensa/*/qemu/bin/qemu-system-xtensa | head -1)"
PORT=5561
"$QEMU" -nographic -machine esp32s3 -drive file="$OUT/flash.bin",if=mtd,format=raw \
  -serial tcp::$PORT,server,nowait -monitor none >"$OUT/qemu.log" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT
sleep 4
cd "$DEMO"
"$DEMO/../.venv/bin/python" - "$PORT" <<'PY'
import sys
sys.path.insert(0, ".")
from feeder.board_link import BoardLink
from feeder.runner import Runner
from feeder.scenarios import load_scenarios
link = BoardLink(f"socket://localhost:{sys.argv[1]}", ack_timeout_s=20.0)
ack = link.handshake(60.0)
print("board:", ack)
assert ack["config"] == "benchmark_v1", ack
r, fails = Runner(link), 0
for sc in load_scenarios():
    for label, eeg in [(sc.source, sc.eeg)] + sc.alternates:
        st = r.start(eeg, sc.expected)
        r._thread.join()
        ok = st.error is None and st.outcome_ok
        fails += not ok
        print(f"scenario {sc.number} {label}: expected {sc.expected or 'quiet'}, board "
              f"{st.command or 'quiet'}, {len(st.decisions)} decisions"
              + (f", error {st.error}" if st.error else "") + ("  ok" if ok else "  FAIL"))
open("build/esp32_qemu/link.log", "w").write("\n".join(link.log) + "\n")
print("PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
PY
