"""Milestone 6: end-to-end synthetic validation of the decoder pipeline.

Verifies pipeline *behaviour* on synthetic signals. These numbers are NOT human
EEG performance and must not be reported as such.

Outputs (results/<run_id>/): config.yaml, run.json (metrics + environment),
trials.csv (every decoded trial), and figures:
  clean_cca_scores.png, accuracy_vs_snr.png, psd_before_after.png,
  confusion_snr<X>.png, gating_scatter.png, score_distributions.png

Usage:
    python scripts/run_synthetic_validation.py [--config configs/synapsys.yaml]
        [--trials-per-class 40] [--seed 20261003] [--epoch-seconds 6.0]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from synapsys_decoder import metrics, plotting, synthetic
from synapsys_decoder.config import DecoderConfig, PreprocessingConfig, load_config
from synapsys_decoder.pipeline import decode_trial
from synapsys_decoder.runlog import create_run_dir, save_run

ROOT = Path(__file__).resolve().parents[1]

SNR_SWEEP_DB = [0, -5, -10, -15, -20, -25, -30]
CONFUSION_SNR_DB = -20
GATING_SNR_DB = -20
DRIFT_V = 20e-6  # slow electrode drift, much larger than the ~1 uV SSVEP
LINE_V = 5e-6  # 60 Hz mains pickup
OFF_TARGET_HZ = (10.0, 20.5)
MIXTURE_PAIR = ("NO", "HELP")


def preprocessing_variants(base: PreprocessingConfig) -> dict[str, PreprocessingConfig]:
    """causal = ESP32 path; zero_phase = offline-only reference; none = ablation."""
    return {
        "causal": dataclasses.replace(base, mode="causal"),
        "zero_phase": dataclasses.replace(base, mode="zero_phase"),
        "none": PreprocessingConfig(
            remove_dc=False, bandpass_low_hz=None, bandpass_high_hz=None, notch_hz=None
        ),
    }


def noisy_kwargs(snr_db: float | None) -> dict:
    return dict(
        snr_db=snr_db,
        drift_amplitude_v=DRIFT_V,
        line_amplitude_v=LINE_V,
        channel_scaling=True,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(ROOT / "configs" / "synapsys.yaml"))
    ap.add_argument("--trials-per-class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument(
        "--epoch-seconds",
        type=float,
        default=6.0,
        help="generated epoch length; the decoder uses the LAST window_seconds, "
        "so the causal filter has (epoch - window) s of history",
    )
    ap.add_argument("--results-root", default=str(ROOT / "results"))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    fs, chans, harm = cfg.sample_rate_hz, cfg.channels, cfg.harmonics
    rng = np.random.default_rng(args.seed)
    run_dir = create_run_dir(args.results_root, "synthetic_validation")
    rows: list[dict] = []
    figs: dict[str, Figure] = {}

    def decode(trial, c: DecoderConfig, **tags):
        d = decode_trial(trial, c)
        r = d.result
        rows.append(
            {
                **tags,
                "true_label": trial.true_label,
                "true_frequency": trial.true_frequency,
                "decision": r.decision,
                "winner": r.winner,
                "confidence": r.confidence,
                "margin": r.margin,
                **{f"score_{k}": v for k, v in r.scores.items()},
            }
        )
        return d

    # --- 8.1 Clean synthetic SSVEP -------------------------------------------------
    clean_true, clean_pred = [], []
    fig_clean = Figure(figsize=(11, 3.4), layout="constrained")
    axes = fig_clean.subplots(1, len(cfg.targets), sharey=True)
    for ax, (label, f) in zip(axes, cfg.targets.items(), strict=True):
        for i in range(args.trials_per_class):
            t = synthetic.make_synthetic_trial(
                f, fs, args.epoch_seconds, chans, rng, harmonics=harm, label=label
            )
            d = decode(t, cfg, experiment="clean", mode="causal", snr_db=None)
            clean_true.append(label)
            clean_pred.append(d.result.decision)
            if i == 0:
                plotting.plot_cca_scores(
                    d.result.scores,
                    true_label=label,
                    frequencies=cfg.targets,
                    title=f"Clean {f:.3f} Hz ({label})",
                    ax=ax,
                )
    fig_clean.suptitle("Synthetic clean SSVEP: CCA score per target (causal pipeline)")
    figs["clean_cca_scores"] = fig_clean
    clean_summary = metrics.summarize(clean_true, clean_pred, cfg.labels)

    # --- 8.2 Noisy synthetic SSVEP: SNR sweep x preprocessing variant ---------------
    variants = preprocessing_variants(cfg.preprocessing)
    sweep: dict[str, list[float]] = {name: [] for name in variants}
    confusion = None
    psd_example = None
    for snr in SNR_SWEEP_DB:
        # identical trials for every variant -> paired comparison
        trials = [
            synthetic.make_synthetic_trial(
                f,
                fs,
                args.epoch_seconds,
                chans,
                rng,
                harmonics=harm,
                label=label,
                **noisy_kwargs(snr),
            )
            for label, f in cfg.targets.items()
            for _ in range(args.trials_per_class)
        ]
        for name, pp in variants.items():
            c = cfg.replace(preprocessing=pp)
            preds = [
                decode(t, c, experiment="snr_sweep", mode=name, snr_db=snr).result.decision
                for t in trials
            ]
            truth = [t.true_label for t in trials]
            sweep[name].append(metrics.accuracy(truth, preds))
            if name == "causal" and snr == CONFUSION_SNR_DB:
                confusion = metrics.confusion_matrix(truth, preds, cfg.labels, include_reject=False)
        if snr == -10:
            psd_example = trials[len(cfg.targets) // 2 * args.trials_per_class]  # middle class

    figs["accuracy_vs_snr"] = plotting.plot_metric_curve(
        SNR_SWEEP_DB,
        xlabel="Broadband SNR (dB, white noise only)",
        ylabel="Accuracy",
        chance_level=1 / len(cfg.targets),
        series=sweep,
        title=f"Synthetic accuracy vs SNR ({cfg.window_seconds:g} s window, "
        f"+{DRIFT_V * 1e6:g} µV drift, +{LINE_V * 1e6:g} µV 60 Hz)",
    )
    figs[f"confusion_snr{CONFUSION_SNR_DB}"] = plotting.plot_confusion_matrix(
        confusion,
        list(cfg.labels),
        list(cfg.labels),
        title=f"Synthetic confusion, causal, SNR {CONFUSION_SNR_DB} dB",
    )
    sel = psd_example.select_channels(chans)
    filtered = decode_trial(psd_example, cfg).filtered
    figs["psd_before_after"] = plotting.plot_psd_before_after(
        sel.eeg,
        filtered,
        fs,
        chans,
        mark_frequencies=cfg.targets,
        harmonics=harm,
        fmax=70,
        title=f"Synthetic {psd_example.true_label} trial, SNR -10 dB: raw vs causal-filtered",
    )

    # --- 8.3 Rejection mechanics (no thresholds chosen here) ------------------------
    kinds: dict[str, list] = {
        "target": [],
        "pure noise": [],
        "off-target": [],
        "ambiguous mixture": [],
    }
    n = args.trials_per_class
    for label, f in cfg.targets.items():
        kinds["target"] += [
            synthetic.make_synthetic_trial(
                f,
                fs,
                args.epoch_seconds,
                chans,
                rng,
                harmonics=harm,
                label=label,
                **noisy_kwargs(GATING_SNR_DB),
            )
            for _ in range(n)
        ]
    kinds["pure noise"] = [
        synthetic.make_noise_trial(
            fs,
            args.epoch_seconds,
            chans,
            rng,
            drift_amplitude_v=DRIFT_V,
            line_amplitude_v=LINE_V,
            channel_scaling=True,
        )
        for _ in range(n)
    ]
    kinds["off-target"] = [
        synthetic.make_off_target_trial(
            OFF_TARGET_HZ[i % len(OFF_TARGET_HZ)],
            fs,
            args.epoch_seconds,
            chans,
            rng,
            harmonics=harm,
            **noisy_kwargs(GATING_SNR_DB),
        )
        for i in range(n)
    ]
    fa, fb = (cfg.targets[k] for k in MIXTURE_PAIR)
    kinds["ambiguous mixture"] = [
        synthetic.make_mixture_trial(
            fa,
            fb,
            0.5,
            fs,
            args.epoch_seconds,
            chans,
            rng,
            harmonics=harm,
            **noisy_kwargs(GATING_SNR_DB),
        )
        for _ in range(n)
    ]

    gating_points: dict[str, tuple[list[float], list[float]]] = {}
    gating_stats: dict[str, dict] = {}
    score_dists: dict[str, dict[str, list[float]]] = {}
    for kind, trials in kinds.items():
        results = [
            decode(
                t, cfg, experiment="rejection", mode="causal", snr_db=GATING_SNR_DB, kind=kind
            ).result
            for t in trials
        ]
        conf = [r.confidence for r in results]
        marg = [r.margin for r in results]
        gating_points[kind] = (conf, marg)
        gating_stats[kind] = {
            "n": len(results),
            "confidence_median": float(np.median(conf)),
            "confidence_p10_p90": [float(np.percentile(conf, 10)), float(np.percentile(conf, 90))],
            "margin_median": float(np.median(marg)),
            "margin_p10_p90": [float(np.percentile(marg, 10)), float(np.percentile(marg, 90))],
        }
        if kind == "target":  # split genuine trials by true class so references are comparable
            for lab in cfg.labels:
                sub = [r for t, r in zip(trials, results, strict=True) if t.true_label == lab]
                score_dists[f"{lab} trials"] = {c: [r.scores[c] for r in sub] for c in cfg.labels}
        else:
            score_dists[kind] = {c: [r.scores[c] for r in results] for c in cfg.labels}
    figs["gating_scatter"] = plotting.plot_gating_scatter(
        gating_points,
        title=f"Rejection mechanics (synthetic, SNR {GATING_SNR_DB} dB) — no thresholds set",
    )
    figs["score_distributions"] = plotting.plot_score_distributions(
        score_dists,
        title=f"CCA score per reference, grouped by trial kind (synthetic, SNR {GATING_SNR_DB} dB)",
    )

    # --- Save ----------------------------------------------------------------------
    for name, fig in figs.items():
        plotting.save_figure(fig, run_dir / f"{name}.png")
    out_metrics = {
        "disclaimer": "Synthetic pipeline-behaviour validation only; not human EEG performance.",
        "seed": args.seed,
        "trials_per_class": args.trials_per_class,
        "epoch_seconds": args.epoch_seconds,
        "snr_definition": "broadband: mean signal power / white-noise variance (drift and "
        "line noise added on top, not counted)",
        "drift_amplitude_v": DRIFT_V,
        "line_amplitude_v": LINE_V,
        "clean": clean_summary,
        "snr_sweep_db": SNR_SWEEP_DB,
        "accuracy_vs_snr": sweep,
        f"confusion_causal_snr{CONFUSION_SNR_DB}": confusion,
        "rejection_mechanics": gating_stats,
        "n_nonfinite_scores": sum(
            not np.isfinite(v) for r in rows for k, v in r.items() if k.startswith("score_")
        ),
    }
    save_run(
        run_dir,
        config={
            **cfg.to_dict(),
            "variants": {k: dataclasses.asdict(v) for k, v in variants.items()},
        },
        metrics=out_metrics,
        trials=rows,
    )

    print(f"run: {run_dir}")
    print(f"clean accuracy: {clean_summary['accuracy']:.3f}")
    print("SNR (dB):   " + "  ".join(f"{s:>6}" for s in SNR_SWEEP_DB))
    for name, accs in sweep.items():
        print(f"{name:<11} " + "  ".join(f"{a:6.3f}" for a in accs))
    for kind, st in gating_stats.items():
        print(
            f"{kind:<18} conf median {st['confidence_median']:.3f}  "
            f"margin median {st['margin_median']:.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
