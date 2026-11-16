"""Check the board's BLE output from the Mac (stands in for the iPhone app).

Scans for "SynapSys", connects, reads Status, subscribes to Command + Debug and prints every
notification with a timestamp. With --feed it also plays the five demo scenarios over the COM
port and checks that each Command notification (or its absence, for the quiet ones) arrived
over BLE as expected.

    cd live_demo
    ../.venv/bin/python firmware/ble_probe.py                         # listen only
    ../.venv/bin/python firmware/ble_probe.py --feed /dev/cu.usbmodemXXXX   # full check (COM port)
Exit code 0 = every scenario arrived over BLE as expected. macOS asks once for Bluetooth
permission for the terminal app.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bleak import BleakClient, BleakScanner

SERVICE = "a5f30001-2c7e-4b8a-9d61-53594e415053"
COMMAND = "a5f30002-2c7e-4b8a-9d61-53594e415053"
DEBUG = "a5f30003-2c7e-4b8a-9d61-53594e415053"
STATUS = "a5f30004-2c7e-4b8a-9d61-53594e415053"
NAME = "SynapSys"
MAP = {"F8_0": "YES", "F12_0": "NO", "F15_0": "HELP", "YES": "YES", "NO": "NO", "HELP": "HELP"}


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--feed", metavar="PORT", help="COM port: also play the 5 scenarios")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--listen", type=float, default=60.0, help="listen-only duration (s)")
    args = ap.parse_args()

    # Open COM first: opening a serial port can reset the board, which would drop BLE.
    link = runner = None
    if args.feed:
        from feeder.board_link import BoardLink
        from feeder.runner import Runner

        link = BoardLink(args.feed, baud=args.baud)
        ack = link.handshake(10.0)
        if ack.get("config") != "benchmark_v1":
            print(
                f"FAIL: board runs config {ack.get('config')!r}, demo firmware needs benchmark_v1"
            )
            return 1
        runner = Runner(link)
        print(f"COM ok ({args.feed}, config {ack['config']})")

    print(f"scanning for {NAME!r} ...")
    dev = await BleakScanner.find_device_by_name(NAME, timeout=15.0)
    if dev is None:
        print("FAIL: board not found (powered? demo firmware? another phone connected?)")
        return 1
    t0 = time.monotonic()
    commands: list[tuple[float, str]] = []

    def on_cmd(_, data: bytearray) -> None:
        label = data.decode(errors="replace")
        commands.append((time.monotonic(), label))
        print(f"[{time.monotonic() - t0:7.2f}s] COMMAND {label} -> {MAP.get(label, '?')}")

    def on_dbg(_, data: bytearray) -> None:
        print(f"[{time.monotonic() - t0:7.2f}s] debug   {data.decode(errors='replace')}")

    async with BleakClient(dev) as client:
        status = (await client.read_gatt_char(STATUS)).decode(errors="replace")
        print(f"connected; status: {status}; MTU {client.mtu_size}")
        await client.start_notify(COMMAND, on_cmd)
        await client.start_notify(DEBUG, on_dbg)
        if not args.feed:
            await asyncio.sleep(args.listen)
            return 0

        from feeder.scenarios import load_scenarios

        failures = 0
        for sc in load_scenarios():
            commands.clear()
            st = runner.start(sc.eeg, sc.expected)
            while not st.done:
                await asyncio.sleep(0.05)
            await asyncio.sleep(1.0)  # BLE notify can trail the COM line slightly
            got = [MAP.get(lab, lab) for _, lab in commands]
            want = [sc.expected] if sc.expected else []
            ok = st.error is None and got == want and st.command == sc.expected
            failures += not ok
            lat = (commands[0][0] - st.started_at) if commands else None
            print(
                f"scenario {sc.number} ({sc.spec.freq_hz:g} Hz, expect {sc.expected or 'quiet'}): "
                f"COM {st.command or 'quiet'}, BLE {got or 'quiet'}"
                + (f", stream start -> BLE {lat:.2f} s" if lat is not None else "")
                + ("" if st.error is None else f", LINK ERROR {st.error}")
                + ("  ok" if ok else "  FAIL")
            )
        link.close()
        print("PASS" if failures == 0 else f"FAIL ({failures} scenario(s))")
        return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
