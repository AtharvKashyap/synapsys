"""Milestones 7-8: validate the generic CCA pipeline on the Wang et al. 2017 Benchmark.

These trials use frequencies that EXIST in the dataset (default 8.0/12.0/15.0 Hz,
labels F8_0/F12_0/F15_0). They are never relabelled as SynapSys YES/NO/HELP, and
no result here shows that SynapSys hardware meets its 85 % requirement.

Experiments (spec section 9):
  1 smoke     : configured channels/harmonics/window -> accuracy, confusion, scores, PSD
  2 window    : accuracy vs window length (benchmark.window_sweep_s)
  3 channels  : configured 4 channels vs benchmark.extended_channels, across windows
  4 harmonics : benchmark.harmonics_sweep, across windows
Every experiment runs the causal (ESP32-portable) filter; experiments 1-2 also run
zero_phase, because the 0.5 s pre-stimulus gives the causal high-pass little time
to settle before the window starts (onset + visual latency = 0.64 s).

Usage:
    python scripts/run_benchmark_validation.py [--config configs/benchmark.yaml]
        [--data-dir data/benchmark] [--subjects 1 2 3]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Sequence
from pathlib import Path

from synapsys_decoder import metrics, plotting
from synapsys_decoder.config import DecoderConfig, load_config
from synapsys_decoder.io.benchmark import benchmark_files_present, load_benchmark_subject
from synapsys_decoder.pipeline import decode_trial
from synapsys_decoder.runlog import create_run_dir, save_run
from synapsys_decoder.types import Trial

ROOT = Path(__file__).resolve().parents[1]
MODES = ("causal", "zero_phase")


def label_trials(trials: Sequence[Trial], cfg: DecoderConfig, tol: float = 1e-3) -> list[Trial]:
    """Attach config labels by matching each trial's true frequency (loader leaves labels None)."""
    out = []
    for t in trials:
        matches = [lab for lab, f in cfg.targets.items() if abs(f - t.true_frequency) <= tol]
        if len(matches) != 1:
            raise ValueError(f"{t.trial_id}: frequency {t.true_frequency} matches {matches}")
        out.append(dataclasses.replace(t, true_label=matches[0]))
    return out


def with_mode(cfg: DecoderConfig, mode: str) -> DecoderConfig:
    return cfg.replace(preprocessing=dataclasses.replace(cfg.preprocessing, mode=mode))


def evaluate(trials, cfg, window_start_s, rows, **tags):
    """Decode every trial; append per-trial rows; return (y_true, y_pred, results)."""
    y_true, y_pred, results = [], [], []
    for t in trials:
        r = decode_trial(t, cfg, window_start_s=window_start_s).result
        y_true.append(t.true_label)
        y_pred.append(r.decision)
        results.append(r)
        rows.append(
            {
                **tags,
                "subject": t.subject_id,
                "block": t.block_id,
                "trial_id": t.trial_id,
                "mode": cfg.preprocessing.mode,
                "window_s": cfg.window_seconds,
                "harmonics": cfg.harmonics,
                "n_channels": len(cfg.channels),
                "true_label": t.true_label,
                "true_frequency": t.true_frequency,
                "decision": r.decision,
                "confidence": r.confidence,
                "margin": r.margin,
                **{f"score_{k}": v for k, v in r.scores.items()},
            }
        )
    return y_true, y_pred, results


def per_subject_accuracy(trials, y_pred) -> dict[str, float]:
    subs: dict[str, tuple[list, list]] = {}
    for t, p in zip(trials, y_pred, strict=True):
        a, b = subs.setdefault(t.subject_id, ([], []))
        a.append(t.true_label)
        b.append(p)
    return {s: metrics.accuracy(a, b) for s, (a, b) in subs.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(ROOT / "configs" / "benchmark.yaml"))
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "benchmark"))
    ap.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=None,
        help="override benchmark.subjects from the config",
    )
    ap.add_argument("--results-root", default=str(ROOT / "results"))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    bm = cfg.extra.get("benchmark")
    if bm is None:
        raise SystemExit(f"{args.config} has no 'benchmark' section")
    subjects = args.subjects or bm["subjects"]
    for s in subjects:
        if not benchmark_files_present(args.data_dir, s):
            raise SystemExit(
                f"benchmark files for S{s} not found in {args.data_dir}; see data/README.md"
            )
    window_start = float(bm["stimulus_onset_s"]) + float(bm["visual_latency_s"])
    windows = [float(w) for w in bm["window_sweep_s"]]
    extended = list(bm["extended_channels"])
    load_channels = list(dict.fromkeys([*cfg.channels, *extended]))

    trials: list[Trial] = []
    for s in subjects:
        trials += label_trials(
            load_benchmark_subject(
                args.data_dir, s, channels=load_channels, frequencies=list(cfg.frequencies)
            ),
            cfg,
        )

    run_dir = create_run_dir(args.results_root, "benchmark_validation")
    rows: list[dict] = []
    out: dict = {
        "disclaimer": "Public-dataset validation of the generic CCA implementation only. "
        "Not SynapSys YES/NO/HELP performance; not evidence for the 85 % "
        "own-hardware requirement.",
        "subjects": subjects,
        "n_trials": len(trials),
        "window_start_s": window_start,
        "chance": 1 / len(cfg.targets),
    }
    labels = list(cfg.labels)

    # --- Experiment 1: smoke test ----------------------------------------------------
    exp1 = {}
    for mode in MODES:
        c = with_mode(cfg, mode)
        y_true, y_pred, results = evaluate(trials, c, window_start, rows, experiment="smoke")
        exp1[mode] = {
            **metrics.summarize(y_true, y_pred, labels),
            "per_subject_accuracy": per_subject_accuracy(trials, y_pred),
        }
        if mode == "causal":
            cm = metrics.confusion_matrix(y_true, y_pred, labels, include_reject=False)
            plotting.save_figure(
                plotting.plot_confusion_matrix(
                    cm,
                    labels,
                    labels,
                    title=f"Benchmark S{subjects} causal, "
                    f"{cfg.window_seconds:g} s, {len(cfg.channels)} ch",
                ),
                run_dir / "smoke_confusion.png",
            )
            dists = {
                f"{lab} trials": {
                    k: [
                        r.scores[k]
                        for t, r in zip(trials, results, strict=True)
                        if t.true_label == lab
                    ]
                    for k in labels
                }
                for lab in labels
            }
            plotting.save_figure(
                plotting.plot_score_distributions(
                    dists, title="Benchmark CCA scores per reference (causal)"
                ),
                run_dir / "smoke_score_distributions.png",
            )
            correct = next(
                (i for i, (a, b) in enumerate(zip(y_true, y_pred, strict=True)) if a == b), None
            )
            if correct is not None:
                t = trials[correct]
                plotting.save_figure(
                    plotting.plot_cca_scores(
                        results[correct].scores,
                        true_label=t.true_label,
                        frequencies=cfg.targets,
                        title=f"{t.trial_id}: correctly classified",
                    ),
                    run_dir / "smoke_cca_scores_example.png",
                )
    out["experiment1_smoke"] = exp1

    # Representative PSD: raw stimulation segment of the first trial of the middle class.
    mid = labels[len(labels) // 2]
    t = next(t for t in trials if t.true_label == mid).select_channels(cfg.channels)
    onset = round(float(bm["stimulus_onset_s"]) * t.fs)
    stim = t.eeg[:, onset : onset + round(5.0 * t.fs)]
    plotting.save_figure(
        plotting.plot_psd(
            stim,
            t.fs,
            t.channel_names,
            mark_frequencies=cfg.targets,
            harmonics=cfg.harmonics,
            fmax=50,
            title=f"{t.trial_id} ({t.true_frequency:g} Hz), 5 s stimulation, unfiltered",
        ),
        run_dir / "psd_example.png",
    )

    # --- Experiment 2: window sweep -----------------------------------------------------
    exp2 = {mode: [] for mode in MODES}
    for mode in MODES:
        for w in windows:
            c = with_mode(cfg, mode).replace(window_seconds=w)
            yt, yp, _ = evaluate(trials, c, window_start, rows, experiment="window")
            exp2[mode].append(metrics.accuracy(yt, yp))
    out["experiment2_window_sweep"] = {"window_s": windows, "accuracy": exp2}
    plotting.save_figure(
        plotting.plot_metric_curve(
            windows,
            xlabel="Window length (s)",
            ylabel="Accuracy",
            series=exp2,
            chance_level=out["chance"],
            title=f"Benchmark accuracy vs window (S{subjects})",
        ),
        run_dir / "accuracy_vs_window.png",
    )

    # --- Experiment 3: channel sets -----------------------------------------------------
    channel_sets = {
        f"{len(cfg.channels)} ch ({'/'.join(cfg.channels)})": list(cfg.channels),
        f"{len(extended)} ch occipital": extended,
    }
    exp3 = {}
    for name, chans in channel_sets.items():
        exp3[name] = []
        for w in windows:
            c = with_mode(cfg, "causal").replace(window_seconds=w, channels=tuple(chans))
            yt, yp, _ = evaluate(trials, c, window_start, rows, experiment="channels")
            exp3[name].append(metrics.accuracy(yt, yp))
    out["experiment3_channels"] = {
        "window_s": windows,
        "channel_sets": channel_sets,
        "accuracy": exp3,
    }
    plotting.save_figure(
        plotting.plot_metric_curve(
            windows,
            xlabel="Window length (s)",
            ylabel="Accuracy",
            series=exp3,
            chance_level=out["chance"],
            title="Benchmark: channel count (causal)",
        ),
        run_dir / "accuracy_channels.png",
    )

    # --- Experiment 4: harmonics --------------------------------------------------------
    exp4 = {}
    for h in bm["harmonics_sweep"]:
        exp4[f"{h} harmonic{'s' if h > 1 else ''}"] = []
        for w in windows:
            c = with_mode(cfg, "causal").replace(window_seconds=w, harmonics=int(h))
            yt, yp, _ = evaluate(trials, c, window_start, rows, experiment="harmonics")
            exp4[f"{h} harmonic{'s' if h > 1 else ''}"].append(metrics.accuracy(yt, yp))
    out["experiment4_harmonics"] = {"window_s": windows, "accuracy": exp4}
    plotting.save_figure(
        plotting.plot_metric_curve(
            windows,
            xlabel="Window length (s)",
            ylabel="Accuracy",
            series=exp4,
            chance_level=out["chance"],
            title="Benchmark: harmonics (causal)",
        ),
        run_dir / "accuracy_harmonics.png",
    )

    save_run(run_dir, config=cfg.to_dict(), metrics=out, trials=rows)
    print(f"run: {run_dir}")
    print(f"subjects {subjects}, {len(trials)} trials, chance {out['chance']:.3f}")
    for mode in MODES:
        print(f"smoke {mode:<10} accuracy {exp1[mode]['accuracy']:.3f}")
    print("window (s)      " + " ".join(f"{w:6.1f}" for w in windows))
    for group in (exp2, exp3, exp4):
        for name, accs in group.items():
            print(f"{name[:15]:<15} " + " ".join(f"{a:6.3f}" for a in accs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
