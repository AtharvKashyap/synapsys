"""Check that the device's printed decisions equal the Python reference on the same samples.

Input: a serial log captured with CONFIG_SYNAPSYS_RAW_STREAM=y (from a real board or
QEMU). For every stream that starts with a ``raw_info`` line (i.e. from the device's
stream start, so filter state and DC offset match) the RAW samples are decoded in Python
on the device schedule and compared with the device's decision lines:

    scores and margin  |device - python| < --tol (default 1e-3, the port's gate)
    winner             identical (points within --tol of a tie are reported, not failed)
    decision           identical for the given thresholds (default: gate off; pass the
                       values the device was flashed with)

Streams without ``raw_info`` (recording started mid-stream) are skipped: their cold-start
Python filter cannot match the device's long-running filter exactly.

Usage:
    python scripts/verify_device_log.py serial.log [--confidence-threshold T] [--margin-threshold M]
Exit code 0 = every compared point passes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from synapsys_decoder.config import load_config
from synapsys_decoder.io.serial_log import parse_serial_lines
from synapsys_decoder.streaming import decode_stream

ROOT = Path(__file__).resolve().parents[1]


def verify(
    lines: list[str],
    cfg,
    *,
    hop: int = 125,
    tol: float = 1e-3,
    confidence_threshold: float | None = None,
    margin_threshold: float | None = None,
) -> dict:
    log = parse_serial_lines(lines)
    report = {"streams": [], "points": 0, "failed": 0, "near_tie": 0, "skipped_streams": 0}
    for st in log.streams:
        name = st.info.get("stream", "?")
        if not st.info or not st.seq or st.seq[0] != 0:
            report["skipped_streams"] += 1
            continue
        trial = st.to_trial(fs=cfg.sample_rate_hz, channels=cfg.channels, allow_gaps=False)
        ss = decode_stream(trial, cfg, hop_samples=hop)
        py = {int(t): i for i, t in enumerate(ss.window_end)}
        decisions = ss.decisions(confidence_threshold, margin_threshold)
        winners = ss.winner_index
        rows = []
        dev_ts = set()
        for d in st.decisions:
            t = int(d["t"])
            dev_ts.add(t)
            if t not in py:
                if t <= trial.n_samples:  # within the recorded samples -> schedule mismatch
                    rows.append({"t": t, "ok": False, "why": "device point not in Python schedule"})
                continue
            i = py[t]
            degenerate_py = bool(ss.degenerate[i])
            if d.get("status") == "DEGENERATE" or degenerate_py:
                ok = (d.get("status") == "DEGENERATE") == degenerate_py
                rows.append({"t": t, "ok": ok, "why": "" if ok else "degenerate mismatch"})
                continue
            err = float(np.max(np.abs(np.asarray(d["scores"], float) - ss.scores[i])))
            merr = abs(float(d["margin"]) - float(ss.margin[i]))
            win_py = ss.labels[winners[i]]
            dec_py = decisions[i]
            near = float(ss.margin[i]) < tol
            ok = err < tol and merr < tol and (d["winner"] == win_py or near)
            ok = ok and (
                d["decision"] == dec_py
                or near
                or _near_gate(ss, i, tol, confidence_threshold, margin_threshold)
            )
            report["near_tie"] += int(near)
            rows.append(
                {
                    "t": t,
                    "ok": ok,
                    "score_err": err,
                    "margin_err": merr,
                    "winner": d["winner"],
                    "winner_py": win_py,
                    "decision": d["decision"],
                    "decision_py": dec_py,
                }
            )
        missing = [int(t) for t in ss.window_end if int(t) not in dev_ts]
        for t in missing:
            rows.append({"t": t, "ok": False, "why": "Python point missing on device"})
        n_fail = sum(not r["ok"] for r in rows)
        errs = [r["score_err"] for r in rows if "score_err" in r]
        report["streams"].append(
            {
                "stream": name,
                "samples": trial.n_samples,
                "points": len(rows),
                "failed": n_fail,
                "max_score_err": max(errs) if errs else None,
                "rows": rows,
            }
        )
        report["points"] += len(rows)
        report["failed"] += n_fail
    return report


def _near_gate(ss, i, tol, t_thr, m_thr) -> bool:
    if t_thr is not None and abs(float(ss.confidence[i]) - t_thr) < tol:
        return True
    return m_thr is not None and abs(float(ss.margin[i]) - m_thr) < tol


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log", type=Path)
    ap.add_argument("--config", default=str(ROOT / "configs" / "synapsys.yaml"))
    ap.add_argument("--hop", type=int, default=125)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--confidence-threshold", type=float, default=None)
    ap.add_argument("--margin-threshold", type=float, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    rep = verify(
        args.log.read_text(errors="replace").splitlines(),
        cfg,
        hop=args.hop,
        tol=args.tol,
        confidence_threshold=args.confidence_threshold,
        margin_threshold=args.margin_threshold,
    )
    for s in rep["streams"]:
        mse = "n/a" if s["max_score_err"] is None else f"{s['max_score_err']:.2e}"
        print(
            f"{s['stream']:<24} samples {s['samples']:>6}  points {s['points']:>4}  "
            f"failed {s['failed']:>3}  max score err {mse}"
        )
        for r in s["rows"]:
            if not r["ok"]:
                print(f"    FAIL t={r['t']}: {r}")
    print(
        f"{rep['points']} points compared, {rep['failed']} failed, {rep['near_tie']} near a "
        f"tie, {rep['skipped_streams']} stream(s) skipped (no raw_info / not from seq 0)"
    )
    if rep["points"] == 0:
        print("nothing compared: capture the log from a stream start with RAW_STREAM enabled")
        return 1
    return 0 if rep["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
