"""Run the single held-out evaluation with frozen thresholds (M11). Run it once.

Inputs: recordings marked ``split: test`` (anything else is refused) and the decoder config
whose ``decision`` block holds the T/M chosen by ``calibrate_thresholds.py``. Thresholds may
not be passed on the command line, so they cannot be tuned here.

Reported (results/<run_id>/): per target recording the first emitted command (correct /
wrong / none), the fraction correct over **all** target recordings (no command counts as
wrong: the strict denominator), the same over recordings that produced a command, wrong
commands, and false commands per minute on REST. ``held_out.json`` records the config hash
so a later re-run with different settings is visible.

Usage:
    python scripts/evaluate_heldout.py data/synapsys/S01/test [--settle-seconds 3]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from synapsys_decoder.calibration import (
    REST_LABEL,
    check_calibration_set,
    evaluate_thresholds,
    prepare_recording,
)
from synapsys_decoder.config import load_config
from synapsys_decoder.io.csv_eeg import load_csv_trial
from synapsys_decoder.rearm import RearmConfig
from synapsys_decoder.runlog import create_run_dir, save_run

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--config", default=str(ROOT / "configs" / "synapsys.yaml"))
    ap.add_argument("--settle-seconds", type=float, default=3.0)
    ap.add_argument("--hop", type=int, default=125)
    ap.add_argument("--rearm", type=int, nargs=3, default=(2, 625, 2), metavar=("K", "R", "N"))
    ap.add_argument(
        "--allow-ungated",
        action="store_true",
        help="evaluate with T/M null (reports the ungated baseline explicitly)",
    )
    ap.add_argument("--results-root", default=str(ROOT / "results"))
    args = ap.parse_args(argv)

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    t, m = cfg.decision.confidence_threshold, cfg.decision.margin_threshold
    if t is None and m is None and not args.allow_ungated:
        raise SystemExit("config thresholds are null: calibrate first, or pass --allow-ungated")
    files = []
    for p in map(Path, args.paths):
        files.extend(sorted(p.rglob("*.csv")) if p.is_dir() else [p])
    recs = [
        prepare_recording(
            load_csv_trial(f, channels=list(cfg.channels)),
            cfg,
            name=str(f),
            hop_samples=args.hop,
            settle_seconds=args.settle_seconds,
            purpose="evaluation",
        )
        for f in files
    ]
    check_calibration_set(recs)
    rearm = RearmConfig(*args.rearm)
    total = evaluate_thresholds(recs, t, m, rearm)
    per = []
    for r in recs:
        ev = evaluate_thresholds([r], t, m, rearm)
        per.append(
            {
                "recording": r.name,
                "label": r.true_label,
                "first_command": (
                    "correct"
                    if ev["first_command_correct"] == 1
                    else "wrong"
                    if ev["first_command_wrong"] == 1
                    else "none"
                )
                if not r.is_rest
                else "",
                "rest_commands": ev["rest_commands"],
            }
        )
    commanded = total["first_command_correct"] + total["first_command_wrong"]
    summary = {
        "thresholds": {"confidence": t, "margin": m},
        "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        "rearm": list(args.rearm),
        "recordings": {
            lab: sum(r.true_label == lab for r in recs) for lab in (*cfg.labels, REST_LABEL)
        },
        "accuracy_all_trials": total["first_command_correct"],
        "accuracy_commanded_trials": (
            total["first_command_correct"] / commanded if commanded else None
        ),
        "no_command_rate": total["no_command"],
        "rest_false_commands_per_min": total["rest_commands_per_min"],
        "detail": total,
    }
    run_dir = create_run_dir(args.results_root, "heldout")
    (run_dir / "held_out.json").write_text(json.dumps(summary, indent=2, default=float))
    save_run(run_dir, config=cfg.to_dict(), metrics=summary, trials=per, command=sys.argv)
    print(f"accuracy (all target trials, no command = wrong): {summary['accuracy_all_trials']:.1%}")
    acc_c = summary["accuracy_commanded_trials"]
    print(f"accuracy (trials with a command): {'n/a' if acc_c is None else f'{acc_c:.1%}'}")
    print(
        f"no command: {total['no_command']:.1%}; REST false commands "
        f"{total['rest_commands_per_min']:.2f}/min; results {run_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
