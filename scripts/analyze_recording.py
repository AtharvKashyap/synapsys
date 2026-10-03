"""Inspect a single CSV recording (own SynapSys hardware): PSD + sliding-window CCA.

Debug / biological-validation tool, not an accuracy evaluation. Filters the whole
recording causally (as the ESP32 would), then slides the analysis window.

Outputs (results/<run_id>/): psd_before_after.png, scores_over_time.png,
trials.csv (one row per window), config.yaml, run.json.

Usage:
    python scripts/analyze_recording.py data/synapsys/rec.csv
        [--config configs/synapsys.yaml] [--step-seconds 0.5] [--fs 250 --units uV]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from synapsys_decoder import plotting
from synapsys_decoder.classifier import classify_window_with_config
from synapsys_decoder.config import load_config
from synapsys_decoder.io.csv_eeg import load_csv_trial
from synapsys_decoder.preprocessing import extract_window, preprocess
from synapsys_decoder.runlog import create_run_dir, save_run

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", type=Path)
    ap.add_argument("--config", default=str(ROOT / "configs" / "synapsys.yaml"))
    ap.add_argument("--step-seconds", type=float, default=0.5)
    ap.add_argument("--fs", type=float, default=None, help="override sidecar fs")
    ap.add_argument("--units", default=None, help="override sidecar units (V|mV|uV)")
    ap.add_argument("--results-root", default=str(ROOT / "results"))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    trial = load_csv_trial(args.csv, fs=args.fs, units=args.units, channels=list(cfg.channels))
    if not np.isclose(trial.fs, cfg.sample_rate_hz):
        raise SystemExit(f"recording fs {trial.fs} != config {cfg.sample_rate_hz}; not resampling")
    filtered = preprocess(trial.eeg, trial.fs, cfg.preprocessing)

    rows, ends = [], []
    series: dict[str, list[float]] = {lab: [] for lab in cfg.labels}
    start = 0.0
    while start + cfg.window_seconds <= trial.duration_s + 1e-9:
        w = extract_window(filtered, trial.fs, start, cfg.window_seconds)
        r = classify_window_with_config(w, cfg)
        end = start + cfg.window_seconds
        ends.append(end)
        for lab in cfg.labels:
            series[lab].append(r.scores[lab])
        rows.append({"window_start_s": start, "window_end_s": end, **r.to_dict()})
        start += args.step_seconds
    if not rows:
        raise SystemExit("recording shorter than one analysis window")

    run_dir = create_run_dir(args.results_root, f"recording_{args.csv.stem}")
    plotting.save_figure(
        plotting.plot_psd_before_after(
            trial.eeg,
            filtered,
            trial.fs,
            trial.channel_names,
            mark_frequencies=cfg.targets,
            harmonics=cfg.harmonics,
            fmax=70,
            title=f"{args.csv.name}: raw vs filtered",
        ),
        run_dir / "psd_before_after.png",
    )
    plotting.save_figure(
        plotting.plot_metric_curve(
            ends,
            xlabel="Window end time (s)",
            ylabel="CCA score",
            series=series,
            title=f"{args.csv.name}: sliding {cfg.window_seconds:g} s window",
        ),
        run_dir / "scores_over_time.png",
    )
    flat = [
        {
            **{k: v for k, v in r.items() if k != "scores"},
            **{f"score_{k}": v for k, v in r["scores"].items()},
        }
        for r in rows
    ]
    save_run(
        run_dir,
        config={**cfg.to_dict(), "source_csv": str(args.csv), "step_seconds": args.step_seconds},
        metrics={
            "n_windows": len(rows),
            "recording_metadata": dict(trial.metadata),
            "note": "Debug analysis; filter warm-up affects the first ~1-3 s of windows.",
        },
        trials=flat,
    )
    print(f"run: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
