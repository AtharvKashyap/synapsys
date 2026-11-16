"""Choose the demo-only gating thresholds (T, M) that keep off-target Wang trials quiet.

Demo only. These thresholds come from the public Wang benchmark, never go into configs/,
and are not SynapSys thresholds. T/M are chosen on blocks 0-2 of S1-S3 and reported on
blocks 3-5, the only blocks the demo plays. The demo's off-target frequencies (10, 14 Hz)
and its trials are then picked from blocks 3-5 using these results, i.e. pre-screened.

Device schedule: each 6 s epoch is one host-feed stream (decoder + re-arm reset at its
start), decisions every 125 samples from 625 on, re-arm K=2 R=625 N=2. A trial "commands"
if re-arm emits anything; we keep the first command.

    target trial  (8 / 12 / 15 Hz)      want: first command == its label
    off-target    (|f - target| > 0.4)  want: no command at all ("quiet")

Rule (fixed before looking at blocks 3-5): over the grid, maximise
    target_correct_rate - offtarget_command_rate
on blocks 0-2; ties -> smaller T, then smaller M.

Usage: .venv/bin/python live_demo/analysis/quiet_threshold.py
Writes live_demo/analysis/quiet_threshold.json and prints a summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from synapsys_decoder.config import load_config
from synapsys_decoder.io.benchmark import load_benchmark_subject
from synapsys_decoder.rearm import RearmConfig, RearmStateMachine
from synapsys_decoder.streaming import decode_stream

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SUBJECTS = (1, 2, 3)
CAL_BLOCKS = (0, 1, 2)
DEMO_BLOCKS = (3, 4, 5)
OFF_TARGET_GAP_HZ = 0.4
T_GRID = np.round(np.arange(0.0, 0.61, 0.02), 2)
M_GRID = np.round(np.arange(0.0, 0.31, 0.01), 2)
REARM = RearmConfig(confirm_count=2, refractory_samples=625, release_count=2)


def first_command(decisions: list[str], ends: np.ndarray) -> str | None:
    sm = RearmStateMachine(REARM)
    for d, t in zip(decisions, ends, strict=True):
        c = sm.update(d, int(t))
        if c is not None:
            return c
    return None


def main() -> int:
    cfg = load_config(ROOT / "configs" / "benchmark.yaml")
    freq_to_label = {f: lab for lab, f in cfg.targets.items()}
    rows = []  # one per trial: subject, block, freq, label or None, StreamScores
    for s in SUBJECTS:
        for t in load_benchmark_subject(
            ROOT / "data" / "benchmark", s, channels=list(cfg.channels)
        ):
            f = round(float(t.true_frequency), 1)
            lab = next((v for k, v in freq_to_label.items() if abs(k - f) < 1e-6), None)
            if lab is None and min(abs(f - k) for k in freq_to_label) <= OFF_TARGET_GAP_HZ:
                continue  # too close to a target to call "off-target"
            rows.append((s, int(t.block_id), f, lab, decode_stream(t, cfg)))
    print(f"{len(rows)} trials decoded")

    def evaluate(T: float, M: float, blocks: tuple[int, ...]) -> dict:
        tgt_ok = tgt_n = off_cmd = off_n = 0
        for _, b, _, lab, ss in rows:
            if b not in blocks:
                continue
            c = first_command(ss.decisions(T, M), ss.window_end)
            if lab is None:
                off_n += 1
                off_cmd += c is not None
            else:
                tgt_n += 1
                tgt_ok += c == lab
        return {
            "target_correct": tgt_ok / tgt_n,
            "target_n": tgt_n,
            "offtarget_command": off_cmd / off_n,
            "offtarget_n": off_n,
        }

    best = None
    for T in T_GRID:
        for M in M_GRID:
            r = evaluate(float(T), float(M), CAL_BLOCKS)
            j = r["target_correct"] - r["offtarget_command"]
            key = (-j, T, M)
            if best is None or key < best[0]:
                best = (key, float(T), float(M), r)
    _, T, M, cal = best
    demo = evaluate(T, M, DEMO_BLOCKS)
    ungated = evaluate(0.0, 0.0, DEMO_BLOCKS)

    # Per-frequency quiet rate on the demo blocks (to pick the two demo off-target tiles).
    per_freq: dict[float, list[int]] = {}
    for _, b, f, lab, ss in rows:
        if b in DEMO_BLOCKS and lab is None:
            c = first_command(ss.decisions(T, M), ss.window_end)
            per_freq.setdefault(f, []).append(int(c is None))
    quiet_by_freq = {f"{f:.1f}": sum(v) / len(v) for f, v in sorted(per_freq.items())}

    out = {
        "note": "Demo-only thresholds from the public Wang benchmark; not SynapSys thresholds.",
        "rule": "max target_correct - offtarget_command on blocks 0-2; ties -> smaller T, M",
        "confidence_threshold": T,
        "margin_threshold": M,
        "rearm": {"K": 2, "R": 625, "N": 2},
        "calibration_blocks_0_2": cal,
        "demo_blocks_3_5": demo,
        "demo_blocks_3_5_ungated": ungated,
        "demo_blocks_quiet_rate_by_frequency": quiet_by_freq,
    }
    (HERE / "quiet_threshold.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if "by_frequency" not in k}, indent=2))
    top = sorted(quiet_by_freq.items(), key=lambda kv: -kv[1])[:10]
    print("most reliably quiet off-target frequencies (demo blocks):", top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
