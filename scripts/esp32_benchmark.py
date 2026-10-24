"""Run the Wang benchmark through the ESP32-S3 and compare it with Python, trial by trial.

The board must run the host-feed firmware (Input source "Host feed over USB", decoder config
"Wang benchmark"; see firmware/BRINGUP.md). For each benchmark trial this script streams the
raw samples (exact float64 volts) to the board, which filters and decodes them with the
unchanged C++ core on its own CPU. It then checks the board's output against the frozen
Python reference on the same samples:

    benchmark window   the window Python's 96.3 % was measured on (start 0.64 s, 2.5 s):
                       the board evaluates it on request; scores/margin |err| < 1e-3 and
                       the same winner
    every hop point    all 625 + k*125 decision points of the 6 s epoch, same criteria
    accuracy           board vs Python vs the true target, per trial and overall

Usage:
    python scripts/esp32_benchmark.py --port /dev/cu.usbmodem1101 --subjects 1 2 3
    python scripts/esp32_benchmark.py --port socket://localhost:5555 --subjects 1 --limit 6
                                                        (QEMU, see firmware/BRINGUP.md)
Exit code 0 = every compared point passes. Needs data/benchmark/ (fetch_benchmark.py).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

from synapsys_decoder.config import load_config
from synapsys_decoder.io.benchmark import benchmark_files_present, load_benchmark_subject
from synapsys_decoder.pipeline import decode_trial
from synapsys_decoder.runlog import create_run_dir, save_run
from synapsys_decoder.streaming import decode_stream

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-3
BLOCK = 64  # lines in flight; well under the board's 16 KB RX buffer


class Board:
    def __init__(self, url: str, timeout: float):
        try:
            import serial
        except ImportError:
            sys.exit(
                "pyserial missing: run inside `. ~/esp/esp-idf/export.sh` or "
                'pip install -e ".[record]"'
            )
        self.ser = serial.serial_for_url(url, baudrate=115200, timeout=0.2)
        self.timeout = timeout
        self.events: list[dict] = []
        self.sync_n = 0
        self.log = []

    def _read_until_ack(self, n: int, timeout: float) -> dict | None:
        t_end = time.monotonic() + timeout
        buf = b""
        while time.monotonic() < t_end:
            chunk = self.ser.readline()
            if not chunk:
                continue
            buf += chunk
            if not buf.endswith(b"\n"):
                continue
            line = buf.decode("utf-8", errors="replace").strip()
            buf = b""
            self.log.append(line)
            p = line.find("SSVEP ")
            if p == -1:
                continue
            try:
                ev = json.loads(line[p + 6 :])
            except json.JSONDecodeError:
                continue
            if ev.get("event") == "ack":
                if ev.get("n") == n:
                    return ev
                continue
            self.events.append(ev)
        return None

    def handshake(self, wait_s: float) -> dict:
        t_end = time.monotonic() + wait_s
        while time.monotonic() < t_end:
            self.sync_n += 1
            self.ser.write(f"Y {self.sync_n}\n".encode())
            ack = self._read_until_ack(self.sync_n, 1.0)
            if ack is not None:
                self.events.clear()
                return ack
        sys.exit(
            "no ack from the board: is the host-feed firmware flashed, and the port right? "
            "(press RST and retry)"
        )

    def send_lines(self, lines: list[str]) -> dict:
        for i in range(0, len(lines), BLOCK):
            self.sync_n += 1
            payload = "\n".join(lines[i : i + BLOCK]) + f"\nY {self.sync_n}\n"
            self.ser.write(payload.encode())
            ack = self._read_until_ack(self.sync_n, self.timeout)
            if ack is None:
                sys.exit(f"board stopped answering (sync {self.sync_n})")
            if ack.get("bad_lines"):
                sys.exit(f"board reported {ack['bad_lines']} unparsable line(s)")
        return ack


def label(cfg, freq: float) -> str:
    (lab,) = [k for k, f in cfg.targets.items() if abs(f - freq) <= 1e-3]
    return lab


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", required=True, help="/dev/cu.usbmodem* or socket://host:port")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--config", default=str(ROOT / "configs" / "benchmark.yaml"))
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "benchmark"))
    ap.add_argument("--limit", type=int, default=None, help="only the first N trials")
    ap.add_argument("--timeout", type=float, default=20.0, help="seconds to wait per block")
    ap.add_argument("--results-root", default=str(ROOT / "results"))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    bm = cfg.extra["benchmark"]
    start_s = float(bm["stimulus_onset_s"]) + float(bm["visual_latency_s"])
    start = round(start_s * cfg.sample_rate_hz)
    end = start + cfg.window_samples
    trials = []
    for s in args.subjects:
        if not benchmark_files_present(args.data_dir, s):
            sys.exit(f"S{s} not in {args.data_dir}: run scripts/fetch_benchmark.py {s}")
        for t in load_benchmark_subject(
            args.data_dir, s, channels=list(cfg.channels), frequencies=list(cfg.frequencies)
        ):
            trials.append(dataclasses.replace(t, true_label=label(cfg, t.true_frequency)))
    if args.limit:
        trials = trials[: args.limit]

    board = Board(args.port, args.timeout)
    ack = board.handshake(30.0)
    want = "benchmark_v1" if "benchmark" in Path(args.config).name else "synapsys_v1"
    if ack.get("config") != want:
        sys.exit(
            f"board runs config {ack.get('config')!r}, this run needs {want!r} "
            "(menuconfig: SynapSys BCI -> Decoder config)"
        )
    print(
        f"board ready (config {ack['config']}); {len(trials)} trials, window "
        f"[{start}, {end}) + every hop point"
    )

    rows, all_ok, t0 = [], True, time.monotonic()
    dec_ms = []
    for i, trial in enumerate(trials):
        py = decode_trial(trial, cfg, window_start_s=start_s).result
        ss = decode_stream(trial, cfg)
        lines = [f"B {i}"]
        for n in range(trial.n_samples):
            lines.append("S " + " ".join(f"{v:.17g}" for v in trial.eeg[:, n]))
            if n + 1 == end:
                lines.append("E")
        board.events.clear()
        board.send_lines(lines)
        tag = f"h{i}"
        evs = [e for e in board.events if e.get("stream") == tag]
        ev = next((e for e in evs if e.get("event") == "eval"), None)
        dev_points = {e["t"]: e for e in evs if "scores" in e and e.get("event") is None}

        py_scores = np.array([py.scores[k] for k in cfg.labels])
        if ev is None:
            ok_eval, err, dev_win = False, float("nan"), None
        else:
            err = float(np.max(np.abs(np.array(ev["scores"]) - py_scores)))
            merr = abs(ev["margin"] - py.margin)
            dev_win = ev["winner"]
            ok_eval = err < TOL and merr < TOL and (dev_win == py.winner or py.margin < TOL)
        hop_err, hop_fail = 0.0, 0
        for k, t_end in enumerate(ss.window_end):
            d = dev_points.get(int(t_end))
            if d is None:
                hop_fail += 1
                continue
            dec_ms.append(d.get("ms", 0.0))
            e = float(np.max(np.abs(np.array(d["scores"]) - ss.scores[k])))
            hop_err = max(hop_err, e)
            if e >= TOL or (d["winner"] != ss.labels[ss.winner_index[k]] and ss.margin[k] >= TOL):
                hop_fail += 1
        ok = ok_eval and hop_fail == 0
        all_ok &= ok
        rows.append(
            {
                "trial": i,
                "subject": trial.subject_id,
                "block": trial.block_id,
                "true": trial.true_label,
                "python": py.winner,
                "esp32": dev_win,
                "score_err": err,
                "hop_points": len(ss.window_end),
                "hop_fail": hop_fail,
                "hop_max_err": hop_err,
                "ok": ok,
            }
        )
        mark = "ok " if ok else "FAIL"
        print(
            f"  [{mark}] trial {i:3d} {trial.subject_id} true {trial.true_label:<6} "
            f"python {py.winner:<6} esp32 {dev_win or '-':<6} err {err:.1e}  "
            f"hop {len(ss.window_end) - hop_fail}/{len(ss.window_end)}"
        )

    n = len(rows)
    acc_py = sum(r["python"] == r["true"] for r in rows) / n
    acc_dev = sum(r["esp32"] == r["true"] for r in rows) / n
    agree = sum(r["esp32"] == r["python"] for r in rows)
    summary = {
        "trials": n,
        "subjects": args.subjects,
        "config": ack["config"],
        "accuracy_python": acc_py,
        "accuracy_esp32": acc_dev,
        "winner_agreement": agree,
        "max_score_err_window": float(np.nanmax([r["score_err"] for r in rows]))
        if any(r["esp32"] is not None for r in rows)
        else None,
        "missing_evals": sum(r["esp32"] is None for r in rows),
        "max_score_err_hop": max(r["hop_max_err"] for r in rows),
        "hop_points": sum(r["hop_points"] for r in rows),
        "hop_failed": sum(r["hop_fail"] for r in rows),
        "decision_ms_avg": float(np.mean(dec_ms)) if dec_ms else None,
        "decision_ms_max": float(np.max(dec_ms)) if dec_ms else None,
        "wall_s": time.monotonic() - t0,
        "all_pass": all_ok,
        "note": "Generic CCA on the public Wang benchmark (8/12/15 Hz); not SynapSys "
        "YES/NO/HELP accuracy.",
    }
    run_dir = create_run_dir(args.results_root, "esp32_benchmark")
    save_run(run_dir, config=cfg.to_dict(), metrics=summary, trials=rows, command=sys.argv)
    (run_dir / "board.log").write_text("\n".join(board.log) + "\n")
    print(f"\naccuracy  python {acc_py:.1%}  esp32 {acc_dev:.1%}  ({n} trials, chance 33.3%)")
    werr = summary["max_score_err_window"]
    print(
        f"same winner on {agree}/{n} trials; max score error "
        f"{'n/a' if werr is None else f'{werr:.1e}'} (benchmark window, "
        f"{summary['missing_evals']} missing), {summary['max_score_err_hop']:.1e} (all "
        f"{summary['hop_points']} hop points, {summary['hop_failed']} failed)"
    )
    if dec_ms:
        print(
            f"on-chip decision time: avg {summary['decision_ms_avg']:.2f} ms, "
            f"max {summary['decision_ms_max']:.2f} ms (filter + 3 CCA)"
        )
    print(f"{'PASS' if all_ok else 'FAIL'}; results {run_dir}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
