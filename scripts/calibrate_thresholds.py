"""Choose the gating thresholds T/M from own-system calibration recordings (Phase 7 / M11).

Input: CSV recordings (``record_serial.py``) whose sidecars carry ``true_label`` (a target
or ``REST``) and ``metadata.split``. Recordings marked ``test`` are refused. The rule that
picks T/M is fixed in advance (see ``synapsys_decoder.calibration``); the team must supply
the tolerated false-activation rate, because that is a product decision, not a data one.

Outputs (results/<run_id>/): sweep.csv (every T/M), recordings.csv (per recording, gate
off vs chosen), selection.json, threshold_map.png, apply.txt (YAML + sdkconfig lines),
config.yaml, run.json.

Usage:
    python scripts/calibrate_thresholds.py data/synapsys/S01/cal --max-rest-commands-per-min 0.5
        [--max-wrong-fraction 0.1] [--settle-seconds 3] [--hop 125] [--rearm 2 625 2]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from synapsys_decoder import plotting
from synapsys_decoder.calibration import (
    REST_LABEL,
    evaluate_thresholds,
    prepare_recording,
    select_thresholds,
    sweep,
    threshold_grid,
)
from synapsys_decoder.config import load_config
from synapsys_decoder.io.csv_eeg import load_csv_trial
from synapsys_decoder.rearm import RearmConfig
from synapsys_decoder.runlog import create_run_dir, save_run

ROOT = Path(__file__).resolve().parents[1]


def _csvs(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        out.extend(sorted(p.rglob("*.csv")) if p.is_dir() else [p])
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        fields: list[str] = []
        for r in rows:
            fields.extend(k for k in r if k not in fields)
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _threshold_map(rows, chosen, limit):
    import matplotlib.pyplot as plt

    ts = sorted({r["confidence_threshold"] for r in rows})
    ms = sorted({r["margin_threshold"] for r in rows})
    idx = {(r["confidence_threshold"], r["margin_threshold"]): r for r in rows}
    good = np.array([[idx[(t, m)]["first_command_correct"] for t in ts] for m in ms])
    rest = np.array([[idx[(t, m)]["rest_commands_per_min"] for t in ts] for m in ms])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, z, title, cmap in (
        (axes[0], good, "Target recordings: first command correct", "viridis"),
        (axes[1], rest, "REST: false commands per minute", "magma_r"),
    ):
        im = ax.pcolormesh(ts, ms, z, shading="nearest", cmap=cmap)
        fig.colorbar(im, ax=ax)
        ax.contour(ts, ms, rest, levels=[limit], colors="white", linewidths=1.5)
        if chosen is not None:
            ax.plot(chosen["confidence_threshold"], chosen["margin_threshold"], "r*", ms=14)
        ax.set_xlabel("confidence threshold T")
        ax.set_ylabel("margin threshold M")
        ax.set_title(title)
    fig.suptitle(f"White line: REST limit {limit:g}/min (feasible side = fewer). Star: chosen.")
    return fig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", help="CSV files or directories (searched recursively)")
    ap.add_argument(
        "--max-rest-commands-per-min",
        type=float,
        required=True,
        help="team-chosen false-activation limit on REST recordings",
    )
    ap.add_argument(
        "--max-wrong-fraction",
        type=float,
        default=None,
        help="optional limit on target recordings whose first command is wrong",
    )
    ap.add_argument("--config", default=str(ROOT / "configs" / "synapsys.yaml"))
    ap.add_argument("--settle-seconds", type=float, default=3.0)
    ap.add_argument("--hop", type=int, default=125)
    ap.add_argument("--rearm", type=int, nargs=3, default=(2, 625, 2), metavar=("K", "R", "N"))
    ap.add_argument("--results-root", default=str(ROOT / "results"))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if cfg.decision.confidence_threshold is not None or cfg.decision.margin_threshold is not None:
        print("note: config thresholds are ignored here; the sweep sets T/M itself")
    rearm = RearmConfig(*args.rearm)
    files = _csvs(args.paths)
    if not files:
        raise SystemExit("no CSV recordings found")
    recs = []
    for f in files:
        trial = load_csv_trial(f, channels=list(cfg.channels))
        recs.append(
            prepare_recording(
                trial, cfg, name=str(f), hop_samples=args.hop, settle_seconds=args.settle_seconds
            )
        )
    counts = {lab: sum(r.true_label == lab for r in recs) for lab in (*cfg.labels, REST_LABEL)}
    print(f"{len(recs)} recordings: {counts}")
    missing = [lab for lab, n in counts.items() if n == 0]
    if missing:
        print(f"warning: no recordings for {missing}")

    rows = sweep(recs, threshold_grid(), rearm)
    chosen = select_thresholds(
        rows,
        max_rest_commands_per_min=args.max_rest_commands_per_min,
        max_wrong_fraction=args.max_wrong_fraction,
    )
    ungated = evaluate_thresholds(recs, None, None, rearm)

    run_dir = create_run_dir(args.results_root, "calibration")
    _write_csv(run_dir / "sweep.csv", rows)
    per_rec = []
    for r in recs:
        row = {"recording": r.name, "label": r.true_label, "minutes": round(r.minutes, 3)}
        for tag, (t, m) in {
            "gate_off": (None, None),
            "chosen": (
                (chosen["confidence_threshold"], chosen["margin_threshold"])
                if chosen
                else (None, None)
            ),
        }.items():
            ev = evaluate_thresholds([r], t, m, rearm)
            key = "rest_commands" if r.is_rest else "correct_commands"
            row[f"{tag}_{key}"] = ev[key]
            row[f"{tag}_wrong_commands"] = ev["wrong_commands"]
        per_rec.append(row)
    _write_csv(run_dir / "recordings.csv", per_rec)
    selection = {
        "rule": "max first_command_correct s.t. rest_commands_per_min <= limit "
        "[and first_command_wrong <= max_wrong_fraction]; ties -> fewer wrong, "
        "fewer rest, larger T, larger M",
        "max_rest_commands_per_min": args.max_rest_commands_per_min,
        "max_wrong_fraction": args.max_wrong_fraction,
        "settle_seconds": args.settle_seconds,
        "rearm": {
            "K": rearm.confirm_count,
            "R": rearm.refractory_samples,
            "N": rearm.release_count,
        },
        "recordings": counts,
        "gate_off": ungated,
        "chosen": chosen,
        "warning": "Chosen on calibration data only. Report performance from a separate "
        "held-out test set evaluated once with these fixed values.",
    }
    (run_dir / "selection.json").write_text(json.dumps(selection, indent=2, default=float))
    plotting.save_figure(
        _threshold_map(rows, chosen, args.max_rest_commands_per_min), run_dir / "threshold_map.png"
    )
    if chosen is not None:
        t, m = chosen["confidence_threshold"], chosen["margin_threshold"]
        apply = (
            "# configs/synapsys.yaml\n"
            f"decision:\n  confidence_threshold: {t:.2f}\n  margin_threshold: {m:.2f}\n\n"
            "# firmware/esp32 (idf.py menuconfig -> SynapSys, or sdkconfig)\n"
            f"CONFIG_SYNAPSYS_CONF_THRESHOLD_EN={'y' if t > 0 else 'n'}\n"
            f'CONFIG_SYNAPSYS_CONF_THRESHOLD="{t:.2f}"\n'
            f"CONFIG_SYNAPSYS_MARGIN_THRESHOLD_EN={'y' if m > 0 else 'n'}\n"
            f'CONFIG_SYNAPSYS_MARGIN_THRESHOLD="{m:.2f}"\n'
        )
        (run_dir / "apply.txt").write_text(apply)
    save_run(run_dir, config=cfg.to_dict(), metrics=selection, command=sys.argv)

    def show(tag, r):
        print(
            f"{tag}: first command correct {r['first_command_correct']:.1%}, wrong "
            f"{r['first_command_wrong']:.1%}, none {r['no_command']:.1%}; REST false "
            f"commands {r['rest_commands_per_min']:.2f}/min over {r['rest_minutes']:.1f} min"
        )

    show("gate off", ungated)
    if chosen is None:
        print(f"NO threshold pair meets the limits; see {run_dir}/threshold_map.png")
        return 1
    show(
        f"chosen T={chosen['confidence_threshold']:.2f} M={chosen['margin_threshold']:.2f}", chosen
    )
    print(f"results: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
