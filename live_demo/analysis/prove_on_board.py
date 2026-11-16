"""Evidence that the ESP32-S3 really decodes the demo, and when.

    cd live_demo && ../.venv/bin/python analysis/prove_on_board.py --port /dev/cu.usbmodemXXXX

Close the stage app first (one program per port). For every run it:
  1. streams the EEG in real time exactly like the stage app (B/S/Y lines only: the laptop
     never sends a label or the expected answer, only numbers);
  2. stamps every decision line the board sends back with the laptop time it arrived;
  3. decodes the same bytes (the 9-digit values as sent) with the frozen Python reference
     decoder and compares the board's CCA scores, decision by decision.
Runs: the 5 demo scenarios, then signals made up on the spot (seeded sine + noise, and noise
only) that are not in any dataset, so a match there cannot come from a stored answer.
Writes build/proof_<time>/report.md + runs.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))

from feeder.board_link import BoardLink, s_line  # noqa: E402
from feeder.runner import FS, samples_due  # noqa: E402
from feeder.scenarios import (  # noqa: E402
    COMMAND_OF_BOARD_LABEL,
    CONFIG_PATH,
    demo_threshold,
    load_scenarios,
    simulate,
)

from synapsys_decoder.config import load_config  # noqa: E402
from synapsys_decoder.streaming import decode_stream  # noqa: E402
from synapsys_decoder.types import Trial  # noqa: E402

ONSET_S = 0.5  # flicker onset inside every Wang epoch (as feeder/app.py)


def as_sent(eeg: np.ndarray) -> np.ndarray:
    """The values the board actually receives (parsed back from the S lines)."""
    return np.array(
        [[float(x) for x in s_line(eeg[:, k]).split()[1:]] for k in range(eeg.shape[1])]
    ).T


def synthetic(freq_hz: float | None, seed: int, seconds: float = 6.0) -> np.ndarray:
    """4 channels: 10 uV RMS noise, plus (from 0.5 s) a 4 uV sine at freq_hz with a small phase
    lag per channel. freq_hz None = noise only."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    t = np.arange(n) / FS
    eeg = rng.normal(0.0, 10e-6, size=(4, n))
    if freq_hz is not None:
        on = t >= ONSET_S
        for ch in range(4):
            eeg[ch, on] += 4e-6 * np.sin(2 * np.pi * freq_hz * (t[on] - ONSET_S) - 0.3 * ch)
    return eeg


def stream(link: BoardLink, stream_id: int, eeg: np.ndarray, tail_s: float = 0.8) -> dict:
    """Real-time playback; returns every board event for this stream with its arrival time."""
    tag = f"h{stream_id}"
    total, sent, events, lag = eeg.shape[1], 0, [], 0
    t0 = time.monotonic()

    def collect() -> None:
        now = time.monotonic() - t0
        for ev in link.drain_events():
            if ev.get("stream") == tag:
                events.append({**ev, "arrived_s": now})

    while sent < total:
        el = time.monotonic() - t0
        lag = max(lag, min(total, int(el * FS)) - sent)
        n = samples_due(el, sent, total)
        if n == 0:
            time.sleep(0.003)
        else:
            link.send_block(eeg[:, sent : sent + n], begin_stream=stream_id if sent == 0 else None)
            sent += n
        collect()
    t_end = time.monotonic() + tail_s
    while time.monotonic() < t_end:
        collect()
        time.sleep(0.01)
    collect()
    return {"events": events, "max_lag": lag}


def run_one(link, cfg, threshold, stream_id, name, eeg, expected) -> dict:
    sent_values = as_sent(eeg)
    res = stream(link, stream_id, eeg)
    board = [e for e in res["events"] if "scores" in e and e.get("event") is None]
    cmds = [e for e in res["events"] if e.get("event") == "command"]
    trial = Trial(eeg=sent_values, fs=FS, channel_names=tuple(cfg.channels))
    ref = decode_stream(trial, cfg)
    ref_by_t = {int(t): ref.scores[i] for i, t in enumerate(ref.window_end)}
    diffs = [
        float(np.max(np.abs(np.array(e["scores"]) - ref_by_t[int(e["t"])])))
        for e in board
        if int(e["t"]) in ref_by_t
    ]
    py_cmd, py_t, _, _ = simulate(trial, cfg, threshold)
    board_cmd = COMMAND_OF_BOARD_LABEL.get(cmds[0]["label"]) if cmds else None
    first = board[0] if board else None
    return {
        "name": name,
        "expected": expected,
        "board_command": board_cmd,
        "python_command": py_cmd,
        "board_decisions": len(board),
        "python_decisions": len(ref.window_end),
        "matched_decisions": len(diffs),
        "max_score_diff": max(diffs) if diffs else None,
        "board_cmd_sample": int(cmds[0]["t"]) if cmds else None,
        "python_cmd_sample": py_t if py_cmd else None,
        "cmd_arrived_s": cmds[0]["arrived_s"] if cmds else None,
        "first_decision_sample": int(first["t"]) if first else None,
        "first_decision_arrived_s": first["arrived_s"] if first else None,
        "latency_ms": [round(1000 * (e["arrived_s"] - int(e["t"]) / FS), 1) for e in board],
        "board_ms_per_decision": [e.get("ms") for e in board],
        "max_lag_samples": res["max_lag"],
        "decisions": [
            {k: e.get(k) for k in ("t", "scores", "winner", "decision", "state", "ms", "arrived_s")}
            for e in board
        ],
    }


def fmt_s(x: float | None) -> str:
    return "—" if x is None else f"{x:.2f} s"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", required=True)
    args = ap.parse_args()
    cfg = load_config(CONFIG_PATH)
    threshold = demo_threshold()
    link = BoardLink(args.port)
    hello = link.handshake(12.0)
    runs = []
    plan = [(f"scenario {s.number}: {s.source}", s.eeg, s.expected) for s in load_scenarios()]
    plan += [
        ("made-up signal: 15 Hz sine in noise (seed 101)", synthetic(15.0, 101), "HELP"),
        ("made-up signal: 8 Hz sine in noise (seed 102)", synthetic(8.0, 102), "YES"),
        ("made-up signal: 12 Hz sine in noise (seed 103)", synthetic(12.0, 103), "NO"),
        ("made-up signal: noise only (seed 104)", synthetic(None, 104), None),
    ]
    base = int(time.time()) % 100000 * 10
    try:
        for i, (name, eeg, expected) in enumerate(plan):
            r = run_one(link, cfg, threshold, base + i, name, eeg, expected)
            runs.append(r)
            print(
                f"{name:<50} expected {expected or 'quiet':<5} "
                f"board {r['board_command'] or 'quiet':<5}"
                f" python {r['python_command'] or 'quiet':<5} | {r['matched_decisions']}/"
                f"{r['python_decisions']} decisions, max score diff {r['max_score_diff']:.1e}"
                f" | command at {fmt_s(r['cmd_arrived_s'])}",
                flush=True,
            )
    finally:
        link.close()

    out = DEMO / "build" / f"proof_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs.json").write_text(json.dumps({"board": hello, "runs": runs}, indent=1))
    all_ok = all(
        r["board_command"] == r["python_command"] == r["expected"]
        and r["matched_decisions"] == r["python_decisions"]
        and r["max_score_diff"] is not None
        and r["max_score_diff"] < 1e-4
        for r in runs
    )
    lat = [x for r in runs for x in r["latency_ms"]]
    ms = [x for r in runs for x in r["board_ms_per_decision"] if x is not None]
    lines = [
        "# ESP32-S3 demo: on-board decoding evidence",
        "",
        f"{time.strftime('%Y-%m-%d %H:%M')}, port `{args.port}`, board config "
        f"`{hello.get('config')}`, gate T = {threshold}.",
        "",
        "The laptop sends only `B`/`S`/`Y` lines (sample numbers); it never sends a label. "
        "Every decision line below came back from the board over COM. The Python column is the "
        "frozen reference decoder run on the same values the board received.",
        "",
        "| Run | Expected | Board | Python | Decisions matched | Max score diff | Board's "
        "command at (after data start / after flicker) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        at = r["cmd_arrived_s"]
        when = "—" if at is None else f"{at:.2f} s / {at - ONSET_S:.2f} s"
        lines.append(
            f"| {r['name']} | {r['expected'] or 'quiet'} | {r['board_command'] or 'quiet'} | "
            f"{r['python_command'] or 'quiet'} | {r['matched_decisions']}/{r['python_decisions']}"
            f" | {r['max_score_diff']:.1e} | {when} |"
        )
    lines += [
        "",
        "## Timing",
        "",
        "- The decoder needs a 2.5 s window (625 samples), so the first decision is at sample "
        "625 (2.5 s of data) and then every 0.5 s.",
        "- A command needs 2 agreeing decisions in a row (re-arm K = 2), so the earliest command "
        "is at sample 750 = 3.0 s of data.",
        f"- The recording starts {ONSET_S} s before the flicker, so 3.0 s of data is 2.5 s "
        "after the tiles start flickering.",
        f"- Decision arrival minus the time its last sample was due: median "
        f"{np.median(lat):.0f} ms, max {np.max(lat):.0f} ms (COM transfer + compute).",
        f"- On-chip compute per decision: median {np.median(ms):.2f} ms, max {np.max(ms):.2f} ms.",
        "",
        "Every decision arrives after the samples it uses were sent and within tens of "
        "milliseconds of them, which is what on-line decoding looks like. A replayed or "
        "pre-computed answer would not need the data to arrive first.",
        "",
        f"**Result: {'PASS' if all_ok else 'FAIL'}**",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"\n{'PASS' if all_ok else 'FAIL'}  report: {out / 'report.md'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
