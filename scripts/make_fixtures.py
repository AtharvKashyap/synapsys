"""Generate Python->C++ regression fixtures from the frozen v1 reference decoder.

The C++/ESP32 decoder must reproduce THESE outputs trial by trial (not just similar
overall accuracy). Each fixture is one multichannel input stream plus the Python
reference outputs under each exported config:

  fixtures/
    README.md                       format spec + acceptance tolerances
    MANIFEST.json                   every file + SHA-256 + generator git revision
    config/<name>.json              frozen configs incl. exact SOS filter coefficients
    synthetic/<id>/                 committed (seeded, no licensing question)
    wang/<id>/                      GENERATED LOCALLY from data/benchmark (git-ignored:
                                    dataset redistribution terms are unstated)
  each <id>/:
    raw.csv                         input stream, volts, %.17g (exact float64 round trip)
    filtered_<config>.csv           causal-filtered stream, volts
    expected.json                   decision points: window bounds, scores, all canonical
                                    correlations, winner, margin, ungated decision,
                                    gated decisions for TEST-ONLY thresholds

Expected outputs are computed from the values read back from raw.csv, so Python and
C++ start from bit-identical input. Streams are processed as the device would: causal
filtering from the first sample (first-sample DC removal), a 625-sample window ending
at each decision point, references restarting at t = 0 in every window.

Usage:
    python scripts/make_fixtures.py [--out fixtures] [--no-wang]
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from synapsys_decoder import synthetic
from synapsys_decoder.cca import DegenerateSignalError, canonical_correlations
from synapsys_decoder.classifier import decide
from synapsys_decoder.config import DecoderConfig, load_config
from synapsys_decoder.io.benchmark import benchmark_files_present, load_benchmark_subject
from synapsys_decoder.pipeline import decode_trial
from synapsys_decoder.preprocessing import build_filter_sos, preprocess
from synapsys_decoder.references import harmonic_references
from synapsys_decoder.runlog import git_revision
from synapsys_decoder.types import Trial

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261018
FORMAT_VERSION = 1
CONFIGS = {"synapsys_v1": "configs/synapsys.yaml", "benchmark_v1": "configs/benchmark.yaml"}
HOP = 125  # decision every 0.5 s at 250 Hz
#: TEST-ONLY gating thresholds to exercise the REJECT path. Not calibrated; never deploy.
TEST_THRESHOLDS = [(t, m) for t in (0.2, 0.3, 0.4) for m in (0.02, 0.05, 0.1)]
CHANNELS = ("O1", "Oz", "O2", "POz")


def write_matrix_csv(path: Path, header: list[str], cols: np.ndarray, fmt: str) -> None:
    """cols: (n_cols, n_rows); one row per sample."""
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for row in cols.T:
            w.writerow([fmt % v for v in row])


def read_matrix_csv(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return data.T.astype(np.float64)


def export_config(name: str, cfg: DecoderConfig) -> dict:
    sos = build_filter_sos(cfg.sample_rate_hz, cfg.preprocessing)
    refs = {
        lab: harmonic_references(f, cfg.sample_rate_hz, cfg.window_samples, cfg.harmonics)
        for lab, f in cfg.targets.items()
    }
    return {
        "name": name,
        "format_version": FORMAT_VERSION,
        "sample_rate_hz": cfg.sample_rate_hz,
        "channels": list(cfg.channels),
        "targets": [{"label": lab, "frequency_hz": f} for lab, f in cfg.targets.items()],
        "harmonics": cfg.harmonics,
        "window_samples": cfg.window_samples,
        "reference_rows": "sin(2*pi*h*f*t), cos(2*pi*h*f*t) for h = 1..H; t = n/fs, "
        "n = 0..window_samples-1 (restarts at 0 in every window)",
        "reference_checks": {
            lab: {"row_sums": r.sum(axis=1).tolist(), "last_column": r[:, -1].tolist()}
            for lab, r in refs.items()
        },
        "preprocessing": {
            **dataclasses.asdict(cfg.preprocessing),
            "dc_policy": "subtract each channel's first sample, then filter from zero state",
            "sos_layout": "rows [b0, b1, b2, a0, a1, a2], a0 = 1, applied in order "
            "(direct form II transposed, scipy.signal.sosfilt)",
            "sos": sos.tolist() if sos is not None else None,
        },
        "cca": {
            "method": "Bjorck-Golub principal angles: centre rows; orthonormal basis of each "
            "side from thin SVD, keep singular values > rank_tol * s_max; side is "
            "degenerate if s_max <= rank_tol * sqrt(n) * max|raw|; canonical "
            "correlations = singular values of Qx^T Qy, sorted, clipped to [0, 1]",
            "rank_tol": 1e-10,
            "min_samples_rule": "n_samples > n_channels + 2 * harmonics",
            "no_regularization": True,
        },
        "decision": {
            "winner": "max score; ties -> first target in config order",
            "margin": "winner score - runner-up score",
            "gating": "accept iff (T unset or score >= T) and (M unset or margin >= M)",
            "test_only_thresholds": TEST_THRESHOLDS,
        },
    }


def decision_points(n: int, window: int, extra: tuple[int, ...] = ()) -> list[int]:
    ends = set(range(window, n + 1, HOP)) | {e for e in extra if window <= e <= n}
    return sorted(ends)


def expected_for(raw: np.ndarray, cfg: DecoderConfig, extra_ends: tuple[int, ...]) -> tuple:
    filtered = preprocess(raw, cfg.sample_rate_hz, cfg.preprocessing)
    w = cfg.window_samples
    points = []
    for end in decision_points(raw.shape[1], w, extra_ends):
        win = filtered[:, end - w : end]
        pt: dict = {"window_start_sample": end - w, "window_end_sample": end}
        try:
            allc = {
                lab: canonical_correlations(
                    win, harmonic_references(f, cfg.sample_rate_hz, w, cfg.harmonics)
                ).tolist()
                for lab, f in cfg.targets.items()
            }
        except DegenerateSignalError as e:
            pt.update({"error": "DegenerateSignalError", "message": str(e)})
            points.append(pt)
            continue
        scores = {lab: c[0] for lab, c in allc.items()}
        r = decide(scores, cfg.targets)
        pt.update(
            {
                "scores": scores,
                "all_correlations": allc,
                "winner": r.winner,
                "winner_frequency": r.winner_frequency,
                "confidence": r.confidence,
                "margin": r.margin,
                "decision_ungated": r.decision,
                "gated_test_only": [
                    {"T": t, "M": m, "decision": decide(scores, cfg.targets, t, m).decision}
                    for t, m in TEST_THRESHOLDS
                ],
            }
        )
        points.append(pt)
    return filtered, points


def write_fixture(
    out: Path, fid: str, trial: Trial, meta: dict, configs: dict, extra_ends: tuple[int, ...] = ()
) -> None:
    d = out / fid
    d.mkdir(parents=True, exist_ok=True)
    names = list(trial.channel_names)
    write_matrix_csv(d / "raw.csv", names, trial.eeg, "%.17g")
    raw = read_matrix_csv(d / "raw.csv")  # expected outputs from exactly what C++ will read
    exp = {
        "id": fid,
        "format_version": FORMAT_VERSION,
        "units": "V",
        "channels": names,
        "n_samples": raw.shape[1],
        **meta,
        "results": {},
    }
    for cname, cfg in configs.items():
        filtered, points = expected_for(raw, cfg, extra_ends)
        write_matrix_csv(d / f"filtered_{cname}.csv", names, filtered, "%.17g")
        exp["results"][cname] = {
            "filtered_file": f"filtered_{cname}.csv",
            "decision_points": points,
        }
    (d / "expected.json").write_text(json.dumps(exp, indent=1))
    print(f"  {fid}")


def synthetic_fixtures(rng) -> list[tuple[str, Trial, dict]]:
    fs, dur = 250.0, 6.0
    noisy = dict(drift_amplitude_v=20e-6, line_amplitude_v=5e-6, channel_scaling=True)
    mk = synthetic.make_synthetic_trial
    out = [
        ("syn_yes_clean", mk(60 / 7, fs, dur, CHANNELS, rng, label="YES"), "clean 60/7 Hz"),
        (
            "syn_yes_snr-15",
            mk(60 / 7, fs, dur, CHANNELS, rng, label="YES", snr_db=-15, **noisy),
            "60/7 Hz, -15 dB broadband + drift + 60 Hz",
        ),
        (
            "syn_no_snr-20",
            mk(12.0, fs, dur, CHANNELS, rng, label="NO", snr_db=-20, **noisy),
            "12 Hz, -20 dB",
        ),
        (
            "syn_help_snr-20",
            mk(15.0, fs, dur, CHANNELS, rng, label="HELP", snr_db=-20, **noisy),
            "15 Hz, -20 dB",
        ),
        (
            "syn_help_snr-28",
            mk(15.0, fs, dur, CHANNELS, rng, label="HELP", snr_db=-28, **noisy),
            "15 Hz, -28 dB: low confidence, may be wrong",
        ),
        (
            "syn_noise_only",
            synthetic.make_noise_trial(fs, dur, CHANNELS, rng, **noisy),
            "no SSVEP: rejection path",
        ),
        (
            "syn_mixture_no_help",
            synthetic.make_mixture_trial(
                12.0, 15.0, 0.5, fs, dur, CHANNELS, rng, snr_db=-15, **noisy
            ),
            "ambiguous 12+15 Hz",
        ),
        (
            "syn_offtarget_10hz",
            synthetic.make_off_target_trial(10.0, fs, dur, CHANNELS, rng, snr_db=-10, **noisy),
            "SSVEP at a non-target 10 Hz",
        ),
    ]
    heavy = mk(12.0, fs, dur, CHANNELS, rng, label="NO", snr_db=-15, line_amplitude_v=50e-6)
    out.append(("syn_heavy_60hz", heavy, "12 Hz with 50 µV 60 Hz mains: exercises the notch"))
    off = mk(15.0, fs, dur, CHANNELS, rng, label="HELP", snr_db=-15, drift_amplitude_v=20e-6)
    off = dataclasses.replace(off, eeg=off.eeg + np.array([[0.3], [-0.25], [0.28], [0.31]]))
    out.append(("syn_large_dc_offset", off, "15 Hz + ~300 mV electrode offsets: float32 stress"))
    dead = mk(12.0, fs, dur, CHANNELS, rng, label="NO", snr_db=-10)
    e = dead.eeg.copy()
    e[2] = 0.0
    out.append(
        (
            "syn_dead_channel",
            dataclasses.replace(dead, eeg=e),
            "O2 all zeros: rank-deficient EEG, must still classify",
        )
    )
    flat = Trial(eeg=np.full((4, 1500), 0.01), fs=fs, channel_names=CHANNELS)
    out.append(
        (
            "syn_flat",
            flat,
            "constant input: every decision point must report "
            "DegenerateSignalError (C++: explicit error, never NaN)",
        )
    )
    return [
        (
            fid,
            t,
            {
                "source": "synthetic",
                "seed": SEED,
                "description": desc,
                "true_label": t.true_label,
                "true_frequency": t.true_frequency,
            },
        )
        for fid, t, desc in out
    ]


def wang_fixtures(data_dir: Path, bench_cfg: DecoderConfig) -> list[tuple[str, Trial, dict]]:
    """Pick real trials by category using the benchmark config at the 0.64 s window."""
    bm = bench_cfg.extra["benchmark"]
    start = float(bm["stimulus_onset_s"]) + float(bm["visual_latency_s"])
    rows = []
    for s in (1, 2, 3):
        if not benchmark_files_present(data_dir, s):
            continue
        for t in load_benchmark_subject(
            data_dir, s, channels=list(CHANNELS), frequencies=list(bench_cfg.frequencies)
        ):
            lab = next(k for k, f in bench_cfg.targets.items() if abs(f - t.true_frequency) < 1e-3)
            t = dataclasses.replace(t, true_label=lab)
            r = decode_trial(t, bench_cfg, window_start_s=start).result
            rows.append((t, r))
    if not rows:
        return []
    wrong = [(t, r) for t, r in rows if r.decision != t.true_label]
    right = sorted(
        [(t, r) for t, r in rows if r.decision == t.true_label], key=lambda tr: tr[1].margin
    )
    picks = [(t, r, "misclassified") for t, r in wrong]
    picks += [(t, r, "correct, lowest margin") for t, r in right[:3]]
    for lab in bench_cfg.labels:
        cls = [(t, r) for t, r in right if t.true_label == lab]
        t, r = cls[len(cls) // 2]
        picks.append((t, r, "correct, median margin"))
    seen, out = set(), []
    for t, r, cat in picks:
        if t.trial_id in seen:
            continue
        seen.add(t.trial_id)
        fid = f"wang_{t.trial_id.lower().replace('-', '_')}_{t.true_frequency:g}hz"
        out.append(
            (
                fid,
                t,
                {
                    "source": "wang2017_benchmark",
                    "trial_id": t.trial_id,
                    "category": cat,
                    "true_label": t.true_label,
                    "true_frequency": t.true_frequency,
                    "benchmark_window_start_sample": round(start * 250),
                    "python_benchmark_decision": r.decision,
                    "python_benchmark_margin": r.margin,
                },
            )
        )
    return out


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "fixtures")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "benchmark")
    ap.add_argument("--no-wang", action="store_true")
    args = ap.parse_args(argv)
    cfgs = {n: load_config(ROOT / p) for n, p in CONFIGS.items()}
    cfgs = {
        n: c.replace(preprocessing=dataclasses.replace(c.preprocessing, mode="causal"))
        for n, c in cfgs.items()
    }
    (args.out / "config").mkdir(parents=True, exist_ok=True)
    for n, c in cfgs.items():
        (args.out / "config" / f"{n}.json").write_text(json.dumps(export_config(n, c), indent=1))

    rng = np.random.default_rng(SEED)
    print("synthetic fixtures:")
    for fid, t, meta in synthetic_fixtures(rng):
        write_fixture(args.out / "synthetic", fid, t, meta, cfgs)
    if not args.no_wang:
        print("wang fixtures:")
        wf = wang_fixtures(args.data_dir, cfgs["benchmark_v1"])
        if not wf:
            print("  (skipped: data/benchmark not present)")
        for fid, t, meta in wf:
            write_fixture(
                args.out / "wang",
                fid,
                t,
                meta,
                cfgs,
                extra_ends=(meta["benchmark_window_start_sample"] + 625,),
            )

    files = sorted(
        p
        for p in args.out.rglob("*")
        if p.is_file() and p.name != "MANIFEST.json" and p.name != "README.md"
    )
    manifest = {
        "format_version": FORMAT_VERSION,
        "generator": "scripts/make_fixtures.py",
        "git_revision": git_revision(),
        "seed": SEED,
        "files": {str(p.relative_to(args.out)): sha256(p) for p in files},
    }
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(files)} files -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
