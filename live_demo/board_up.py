"""One command from "board plugged in" to "demo ready", with PASS/FAIL at every step.

    cd live_demo
    ../.venv/bin/python board_up.py              # waits for the board, flashes, tests everything
    ../.venv/bin/python board_up.py --check      # only the laptop-side checks (no board needed)
    ../.venv/bin/python board_up.py --port /dev/cu.usbmodemXXXX --no-flash

Steps:
  1. laptop checks: Python packages, cached demo trials, ESP-IDF, demo firmware built
  2. find the board's COM port (waits for it to be plugged in)
  3. flash the demo firmware (build_flash.sh)
  4. talk to it: handshake at 921600, config benchmark_v1
  5. real-time check: play YES and a quiet scenario over COM, confirm 250 samples/s
  6. Bluetooth check: ble_probe.py --feed (all 5 scenarios must arrive over BLE as expected)
Everything is logged to build/board_up_<time>.log.
"""

from __future__ import annotations

import argparse
import glob
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

DEMO = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO))
PY = str(DEMO.parent / ".venv" / "bin" / "python")
IDF_EXPORT = Path.home() / "esp" / "esp-idf" / "export.sh"
LOG = DEMO / "build" / f"board_up_{time.strftime('%Y%m%d-%H%M%S')}.log"
PORT_PATTERNS = (
    "/dev/cu.usbmodem*",
    "/dev/cu.wchusbserial*",
    "/dev/cu.usbserial*",
    "/dev/cu.SLAB*",
)


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(msg + "\n")


def ok(step: str, detail: str = "") -> None:
    log(f"  PASS  {step}" + (f"  ({detail})" if detail else ""))


def fail(step: str, detail: str, fix: str) -> None:
    log(f"  FAIL  {step}: {detail}")
    log(f"        fix: {fix}")
    log(f"\nlog: {LOG}")
    sys.exit(1)


def ports() -> set[str]:
    return {p for pat in PORT_PATTERNS for p in glob.glob(pat)}


def laptop_checks() -> None:
    log("1. Laptop checks")
    for mod, pip in (("pygame", "pygame"), ("serial", "pyserial"), ("bleak", "bleak")):
        try:
            importlib.import_module(mod)
        except ImportError:
            fail(f"python package {pip}", "missing", f"../.venv/bin/pip install {pip}")
    ok("python packages", "pygame, pyserial, bleak")
    from feeder.scenarios import CACHE, load_scenarios

    try:
        scs = load_scenarios()
    except Exception as e:  # cache missing and no data, or stale cache
        fail("demo trials", str(e), "../.venv/bin/python -m feeder.scenarios --rebuild")
    ok("demo trials", f"{len(scs)} scenarios cached in {CACHE.relative_to(DEMO)}")
    if not IDF_EXPORT.exists():
        fail("ESP-IDF", f"{IDF_EXPORT} not found", "install ESP-IDF v5.3.2 at ~/esp/esp-idf")
    ok("ESP-IDF", str(IDF_EXPORT.parent))
    binary = DEMO / "build" / "esp32" / "synapsys_bci.bin"
    if binary.exists():
        ok("demo firmware built", f"{binary.stat().st_size // 1024} KB")
    else:
        log("  ....  demo firmware not built yet; step 3 builds it (~3 min)")


def find_port(arg: str | None, wait_s: float) -> str:
    log("2. Board COM port")
    if arg:
        if "://" not in arg and not Path(arg).exists():
            fail("port", f"{arg} does not exist", "check `ls /dev/cu.*`")
        ok("port", arg)
        return arg
    before = ports()
    if len(before) == 1:
        (p,) = before
        ok("port", f"{p} (the only serial port)")
        return p
    log(f"  ....  plug the board's COM connector in now (waiting up to {wait_s:.0f} s)")
    t_end = time.monotonic() + wait_s
    while time.monotonic() < t_end:
        new = ports() - before
        if new:
            time.sleep(1.0)  # both of a board's ports can appear a moment apart
            new = ports() - before
            if len(new) > 1:
                fail(
                    "port",
                    f"several new ports {sorted(new)} (both USB connectors plugged in?)",
                    "unplug the native USB cable, or pass --port with the COM one",
                )
            (p,) = new
            ok("port", p)
            return p
        time.sleep(0.5)
    if before:
        fail("port", f"no new port; existing: {sorted(before)}", "pass --port <the COM port>")
    fail("port", "no serial port appeared", "use a data cable on the COM connector")


def flash(port: str) -> None:
    log("3. Flash demo firmware")
    script = DEMO / "firmware" / "build_flash.sh"
    for baud in ("921600", "460800"):
        env = dict(os.environ, FLASH_BAUD=baud)
        with LOG.open("a") as f:
            r = subprocess.run([str(script), port], stdout=f, stderr=subprocess.STDOUT, env=env)
        if r.returncode == 0:
            ok("flash", f"{baud} baud")
            time.sleep(2.0)  # boot + BLE start
            return
        log(f"  ....  flash at {baud} failed, retrying slower" if baud == "921600" else "")
    fail(
        "flash",
        "esptool could not write the board (see log)",
        "hold BOOT, tap RST, release BOOT, then run again; or try another cable",
    )


def talk(port: str) -> None:
    log("4. Talk to the board")
    from feeder.board_link import BoardLink, LinkError

    try:
        link = BoardLink(port)
        ack = link.handshake(12.0)
    except (LinkError, OSError) as e:
        fail("handshake", str(e)[:160], "is this the COM port? press RST and run with --no-flash")
    try:
        if ack.get("config") != "benchmark_v1":
            fail("config", f"board runs {ack.get('config')!r}", "re-run without --no-flash")
        ok("handshake at 921600", f"config {ack['config']}")
        log("5. Real-time streaming check (no phone needed)")
        from feeder.runner import Runner
        from feeder.scenarios import load_scenarios

        scs = load_scenarios()
        r = Runner(link)
        slow = False
        for sc in (scs[0], scs[3]):  # YES and a quiet one
            st = r.start(sc.eeg, sc.expected)
            r._thread.join(timeout=30)
            secs = (st.command_at - st.started_at) if st.command_at else None
            rate = st.sent / max(1e-9, (time.monotonic() - st.started_at) - 0.6)
            log(
                f"        scenario {sc.number}: expected {sc.expected or 'quiet'}, board "
                f"{st.command or 'quiet'}"
                + (f" after {secs:.1f} s" if secs else "")
                + f", {len(st.decisions)} decisions, worst lag {st.max_lag} samples"
            )
            if st.error:
                fail("streaming", st.error, "see log; try another cable / USB port")
            if not st.outcome_ok:
                fail("decode", f"scenario {sc.number} gave {st.command}", "see the log")
            if st.max_lag > 50:
                slow = True
                log(
                    f"  WARN  real time: link fell {st.max_lag / 250:.2f} s behind "
                    f"(~{rate:.0f} samples/s). Demo still works, commands just come later. "
                    "Close other serial apps; try another USB port/cable."
                )
        if slow:
            ok("decoding", "YES and quiet as expected (slower than real time, see WARN)")
        else:
            ok("real time + decoding", "250 samples/s, YES and quiet as expected")
        link.drain_events()
        time.sleep(5.5)  # the board prints a health line every 5 s
        health = [e for e in link.drain_events() if e.get("event") == "health"]
        if health:
            h = health[-1]
            dec = h.get("dec_ms", {}).get("avg")
            ble_state, dropped = h.get("ble"), h.get("dropped")
            log(f"        board health: ble={ble_state}, decode {dec} ms avg, dropped {dropped}")
    finally:
        link.close()


def ble(port: str) -> None:
    log("6. Bluetooth check (close the SynapSys app on the iPhone first: one connection at a time)")
    with LOG.open("a") as f:
        r = subprocess.run(
            [PY, str(DEMO / "firmware" / "ble_probe.py"), "--feed", port],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        f.write(r.stdout)
    for line in r.stdout.splitlines():
        if line.startswith(("scenario", "connected", "PASS", "FAIL")):
            log("        " + line)
    if r.returncode != 0:
        fail("BLE", "not every scenario arrived over Bluetooth", "see the lines above and the log")
    ok("BLE", "all 5 scenarios arrived as expected")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port")
    ap.add_argument("--no-flash", action="store_true")
    ap.add_argument("--check", action="store_true", help="laptop-side checks only")
    ap.add_argument("--skip-ble", action="store_true")
    ap.add_argument("--wait", type=float, default=180.0, help="seconds to wait for the board")
    args = ap.parse_args()
    log(f"SynapSys demo bring-up {time.strftime('%Y-%m-%d %H:%M:%S')}")
    laptop_checks()
    if args.check:
        log("\nLaptop side ready. Plug in the board and run without --check.")
        return 0
    port = find_port(args.port, args.wait)
    if not args.no_flash:
        flash(port)
    talk(port)
    if not args.skip_ble:
        ble(port)
    log("\nREADY. Next:")
    log("  1. open the SynapSys app on the iPhone -> it should say 'Connected to SynapSys'")
    log(f"  2. ../.venv/bin/python -m feeder.app --port {port}")
    log(f"log: {LOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
