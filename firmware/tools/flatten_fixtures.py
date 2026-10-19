#!/usr/bin/env python3
"""Flatten fixtures/*/<id>/expected.json into CSVs the C++ equivalence runner can parse.

Writes (never into fixtures/):
    <out>/index.txt                     one line per fixture: "<group>/<id>"
    <out>/<group>/<id>/points_<cfg>.csv one row per decision point

points_<cfg>.csv columns (header line included; numbers %.17g; labels -> config index,
REJECT -> -1):
    window_end, error (0/1), score_0..score_{K-1}, winner, confidence, margin,
    decision_ungated, then per target: n_corr, corr_0..corr_3 (padded with 0),
    then per test-only threshold pair: T, M, decision
Error rows carry zeros in every numeric field after the flag.

Usage: flatten_fixtures.py --out <dir> [--fixtures <repo>/fixtures]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIGS = ("synapsys_v1", "benchmark_v1")
GROUPS = ("synthetic", "wang")
MAX_CORR = 4


def g17(x: float) -> str:
    return f"{float(x):.17g}"


def flatten(points: list[dict], labels: list[str], n_thr: int) -> list[str]:
    idx = {lab: i for i, lab in enumerate(labels)}
    idx["REJECT"] = -1
    k = len(labels)
    header = (
        ["window_end", "error"]
        + [f"score_{i}" for i in range(k)]
        + ["winner", "confidence", "margin", "decision_ungated"]
    )
    for i in range(k):
        header += [f"ncorr_{i}"] + [f"corr_{i}_{j}" for j in range(MAX_CORR)]
    for j in range(n_thr):
        header += [f"T_{j}", f"M_{j}", f"decision_{j}"]
    rows = [",".join(header)]
    for p in points:
        if "error" in p:
            if p["error"] != "DegenerateSignalError":
                raise SystemExit(f"unexpected error kind {p['error']}")
            n_fields = len(header) - 2
            rows.append(",".join([str(p["window_end_sample"]), "1"] + ["0"] * n_fields))
            continue
        f = [str(p["window_end_sample"]), "0"]
        f += [g17(p["scores"][lab]) for lab in labels]
        f += [str(idx[p["winner"]]), g17(p["confidence"]), g17(p["margin"])]
        f += [str(idx[p["decision_ungated"]])]
        for lab in labels:
            c = p["all_correlations"][lab]
            f += [str(len(c))] + [g17(v) for v in c] + ["0"] * (MAX_CORR - len(c))
        gated = p["gated_test_only"]
        if len(gated) != n_thr:
            raise SystemExit("gated_test_only length mismatch")
        for g in gated:
            f += [g17(g["T"]), g17(g["M"]), str(idx[g["decision"]])]
        rows.append(",".join(f))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fixtures", type=Path, default=REPO / "fixtures")
    args = ap.parse_args()

    labels = {}
    n_thr = {}
    for c in CONFIGS:
        cfg = json.loads((args.fixtures / "config" / f"{c}.json").read_text())
        labels[c] = [t["label"] for t in cfg["targets"]]
        n_thr[c] = len(cfg["decision"]["test_only_thresholds"])

    index = []
    for group in GROUPS:
        gdir = args.fixtures / group
        if not gdir.is_dir():
            continue
        for fdir in sorted(p for p in gdir.iterdir() if (p / "expected.json").is_file()):
            exp = json.loads((fdir / "expected.json").read_text())
            out = args.out / group / fdir.name
            out.mkdir(parents=True, exist_ok=True)
            for c in CONFIGS:
                pts = exp["results"][c]["decision_points"]
                text = "\n".join(flatten(pts, labels[c], n_thr[c])) + "\n"
                target = out / f"points_{c}.csv"
                if not target.exists() or target.read_text() != text:
                    target.write_text(text)
            index.append(f"{group}/{fdir.name}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.txt").write_text("\n".join(index) + "\n")
    print(f"flattened {len(index)} fixtures -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
