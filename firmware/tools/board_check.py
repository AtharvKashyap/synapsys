#!/usr/bin/env python3
"""Capture the SynapSys board's console and print a PASS/FAIL summary per bring-up stage.

    python firmware/tools/board_check.py --port /dev/cu.usbmodem1101 --seconds 60 \
        --stage replay --log logs/replay.log           # then press the board's RST button
    python firmware/tools/board_check.py --from-log logs/replay.log --stage replay

Stages and what must hold:
    replay     replay_summary PASS on every pass seen; fs 245-255; dropped 0
    ads_test   ADS1299 ID found, register read-back ok; fs 245-255; no DRDY timeouts /
               missed DRDY / SPI errors / bad status; with RAW stream on: ~1 Hz square wave
               of 1.5-5 mV peak-to-peak on every channel (datasheet: VREF/2400 = 1.875 mV)
    live       as ads_test without the square-wave check; reports decisions/commands
    soak       live or replay over a long capture: also heap_min stable, no drops at all

Needs pyserial for --port (it ships with ESP-IDF's Python env, or `pip install pyserial`).
Writes the raw capture to --log so it can be re-checked or archived as evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def capture(port: str, seconds: float, log: Path) -> list[str]:
    try:
        import serial
    except ImportError:
        sys.exit("pyserial missing: run inside `. ~/esp/esp-idf/export.sh` or pip install pyserial")
    log.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    with serial.Serial(port, 115200, timeout=0.5) as s, log.open("w") as f:
        print(f"capturing {seconds:g} s from {port} -> {log}  (press RST on the board now)")
        t_end = time.monotonic() + seconds
        last_report = 0.0
        while time.monotonic() < t_end:
            raw = s.readline()
            if not raw:
                continue
            line = ANSI.sub("", raw.decode("utf-8", errors="replace")).rstrip("\r\n")
            f.write(line + "\n")
            lines.append(line)
            if line.startswith("SSVEP ") and (
                '"event":"replay_summary"' in line or '"event":"command"' in line
            ):
                print("  " + line[6:])
            now = time.monotonic()
            if now - last_report > 10:
                last_report = now
                print(f"  ... {len(lines)} lines, {t_end - now:4.0f} s left")
    return lines


def events(lines):
    out = []
    for ln in lines:
        p = ln.find("SSVEP ")
        if p == -1:
            continue
        try:
            obj = json.loads(ln[p + 6 :])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def square_wave_check(lines):
    """Peak-to-peak (mV) and dominant frequency per channel from RAW lines, or None."""
    vals = []
    for ln in lines:
        if ln.startswith("RAW "):
            parts = ln.split()
            try:
                vals.append([float(x) for x in parts[2:6]])
            except ValueError:
                pass
    if len(vals) < 1000:
        return None
    import numpy as np

    x = np.asarray(vals[250:])  # skip first second
    res = []
    for c in range(x.shape[1]):
        v = x[:, c] - x[:, c].mean()
        ptp_mv = float(np.percentile(v, 99) - np.percentile(v, 1)) / 1000.0
        spec = np.abs(np.fft.rfft(v))
        freqs = np.fft.rfftfreq(len(v), 1 / 250.0)
        f0 = float(freqs[1 + int(np.argmax(spec[1:]))])
        res.append((ptp_mv, f0))
    return res


def summarize(lines, stage: str) -> bool:
    ev = events(lines)
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        ok &= bool(cond)

    text = "\n".join(lines)
    boot = [ln for ln in lines if "SynapSys BCI" in ln or "input source:" in ln]
    print("boot:", *(b.strip() for b in boot[:2]), sep="\n  ")
    errors = [ln for ln in lines if re.match(r"^E \(\d+\)", ln)]
    if errors:
        print(f"error log lines ({len(errors)}), first 5:")
        for e in errors[:5]:
            print("  " + e)

    health = [e for e in ev if e.get("event") == "health"]
    timing = [e for e in ev if e.get("event") == "timing"]
    cmds = [e for e in ev if e.get("event") == "command"]
    decisions = [e for e in ev if "scores" in e and "t" in e]
    print(f"seen: {len(decisions)} decisions, {len(cmds)} commands, {len(health)} health lines")
    if not ev:
        print("  [FAIL] no 'SSVEP' lines at all: wrong port/connector, or board not running")
        return False

    if stage == "replay":
        sums = [e for e in ev if e.get("event") == "replay_summary"]
        check(sums, f"replay finished at least one pass ({len(sums)} seen; one pass ~40 s)")
        for s in sums:
            check(
                s.get("result") == "PASS",
                f"pass {s.get('pass')}: {s.get('passed')} passed, {s.get('failed')} failed, "
                f"{s.get('missing')} missing",
            )
    if stage in ("ads_test", "live", "soak") and 'source":"replay' not in text:
        check("ID = 0x" in text and "no ADS1299 found" not in text, "ADS1299 ID register read")
        check(
            "register read-back mismatch" not in text and "register    wrote" in text,
            "ADS1299 register read-back ok",
        )
        check("acquisition stopped" not in text, "acquisition running")

    if health:
        h = health[-1]
        steady = health[1:] or health
        fs_vals = [x.get("fs", 0) for x in steady]
        if stage == "replay":  # 3 s pause between replay passes lowers some readings
            med = sorted(fs_vals)[len(fs_vals) // 2]
            check(245 <= med <= 255, f"sample rate 245-255 Hz (median {med:.1f})")
        else:
            check(
                all(245 <= f <= 255 for f in fs_vals),
                f"sample rate 245-255 Hz (seen {min(fs_vals):.1f}-{max(fs_vals):.1f})",
            )
        for k in (
            "dropped",
            "drdy_timeouts",
            "missed_drdy",
            "spi_errors",
            "bad_status",
            "events_dropped",
            "raw_dropped",
        ):
            if k in h:
                if stage == "replay" and k in (
                    "drdy_timeouts",
                    "missed_drdy",
                    "spi_errors",
                    "bad_status",
                ):
                    continue
                check(h[k] == 0, f"{k} = {h[k]}")
        print(
            f"  info: heap_free {h.get('heap_free')} B, heap_min {h.get('heap_min')} B, "
            f"stack_free {h.get('stack_free')}, ble {h.get('ble')}, uptime {h.get('uptime_s')} s"
        )
        if stage == "soak" and len(health) >= 3:
            mins = [x.get("heap_min", 0) for x in health[1:]]
            check(
                mins[-1] == mins[0], f"heap_min stable over the capture ({mins[0]} -> {mins[-1]})"
            )
    else:
        check(False, "no health line (they come every 5 s; capture longer)")

    if timing:
        mx = max(t["max_ms"] for t in timing)
        avg = sum(t["avg_ms"] for t in timing) / len(timing)
        check(mx < 50, f"decision compute: avg {avg:.2f} ms, max {mx:.2f} ms (gate < 50 ms)")

    if stage == "ads_test":
        sq = square_wave_check(lines)
        if sq is None:
            print("  [SKIP] square-wave check: enable the RAW stream to check the test signal")
        else:
            for c, (ptp, f0) in zip(("O1", "Oz", "O2", "POz"), sq, strict=False):
                check(
                    1.5 <= ptp <= 5.0 and 0.5 <= f0 <= 2.5,
                    f"{c}: test signal {ptp:.2f} mV p-p at {f0:.2f} Hz "
                    "(expect ~1.9 or ~3.75 mV, ~1 Hz)",
                )
    if cmds:
        print("commands:", ", ".join(f"{c.get('label')}@t={c.get('t')}" for c in cmds[:20]))
    print(f"\nOVERALL ({stage}): {'PASS' if ok else 'FAIL'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--port")
    src.add_argument("--from-log", type=Path)
    ap.add_argument("--stage", choices=("replay", "ads_test", "live", "soak"), required=True)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--log", type=Path, default=Path("logs/board_check.log"))
    args = ap.parse_args(argv)
    if args.port:
        lines = capture(args.port, args.seconds, args.log)
    else:
        lines = [ANSI.sub("", ln) for ln in args.from_log.read_text(errors="replace").splitlines()]
    return 0 if summarize(lines, args.stage) else 1


if __name__ == "__main__":
    sys.exit(main())
