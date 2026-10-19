"""Stage-by-stage real-EEG walkthrough of the decoder on the Wang 2017 Benchmark.

Runs the pipeline one visible stage at a time and, after EVERY stage, rewrites
``status.json`` + figures in the output folder, so a viewer page can follow along:

  1 data        files present, frequencies/channels exist, 8.571 Hz guardrail
  2 load        epochs load as (channels, samples) volts, 250 Hz, finite, plausible µV
  3 ssvep       is an SSVEP visible in the RAW data at the presented frequency?
  4 preprocess  causal filter on a real trial: drift removed, target band kept
  5 cca         one real trial through window -> CCA -> decision (not cherry-picked)
  6 smoke       experiment 1: every trial, accuracy vs chance (binomial test)
  7 sweeps      experiments 2-4: window length, channel count, harmonics

Data-quality checks here (e.g. "SSVEP SNR > 3 dB") are diagnostics of the public
dataset, NOT classifier rejection thresholds. Results validate the generic CCA
implementation at 8/12/15 Hz; they are not SynapSys YES/NO/HELP performance.

Usage:
    python scripts/benchmark_walkthrough.py [--subjects 1 2 3] [--out results/benchmark_walkthrough]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy import signal as sps
from scipy.stats import binomtest

from synapsys_decoder import metrics, plotting
from synapsys_decoder.config import load_config
from synapsys_decoder.io import benchmark as bmio
from synapsys_decoder.io.benchmark import (
    benchmark_files_present,
    load_benchmark_subject,
    read_channel_names,
    read_freq_phase,
)
from synapsys_decoder.pipeline import decode_trial
from synapsys_decoder.preprocessing import extract_window, preprocess
from synapsys_decoder.runlog import environment_info

ROOT = Path(__file__).resolve().parents[1]
PAGE_TEMPLATE = ROOT / "scripts" / "walkthrough_page.html"

STAGES = [
    ("data", "Dataset files", "Are the files here, and do they contain our frequencies?"),
    ("load", "Load EEG", "Do epochs load with the right shape, rate and units?"),
    ("ssvep", "SSVEP in raw EEG", "Is there a brain response at the flicker frequency?"),
    (
        "preprocess",
        "Causal filtering",
        "Does the ESP32-style filter keep the signal and drop drift?",
    ),
    ("cca", "One trial through CCA", "What does the decoder see and decide for a single trial?"),
    ("smoke", "All trials", "How often is the decision right, and is that better than chance?"),
    (
        "sweeps",
        "Design sweeps",
        "How do window length, channel count and harmonics change accuracy?",
    ),
]


class Status:
    def __init__(self, out: Path, subjects: list[int], cfg) -> None:
        self.out = out
        self.t0 = time.time()
        self.doc = {
            "title": "Benchmark walkthrough",
            "dataset": "Wang et al. 2017 Tsinghua SSVEP Benchmark",
            "subjects": [f"S{s}" for s in subjects],
            "targets": cfg.targets,
            "channels": list(cfg.channels),
            "window_seconds": cfg.window_seconds,
            "harmonics": cfg.harmonics,
            "disclaimer": "Public-dataset check of the generic CCA pipeline at 8, 12 and 15 Hz. "
            "Not SynapSys YES/NO/HELP accuracy and not evidence for the 85 % requirement.",
            "started": datetime.now(UTC).isoformat(timespec="seconds"),
            "updated": None,
            "finished": False,
            "stages": [
                {
                    "id": i,
                    "name": n,
                    "question": q,
                    "status": "pending",
                    "summary": "",
                    "checks": [],
                    "figures": [],
                    "seconds": None,
                }
                for i, n, q in STAGES
            ],
            "example": None,
            "headline": None,
        }
        self.write()

    def stage(self, sid: str) -> dict:
        return next(s for s in self.doc["stages"] if s["id"] == sid)

    def start(self, sid: str) -> None:
        st = self.stage(sid)
        st["status"], st["_t"] = "running", time.time()
        print(f"\n== {st['name']} ==")
        self.write()

    def check(self, sid: str, label: str, value, ok: bool | None, note: str = "") -> None:
        self.stage(sid)["checks"].append(
            {"label": label, "value": str(value), "ok": ok, "note": note}
        )
        mark = {True: "PASS", False: "FAIL", None: "info"}[ok]
        print(f"  [{mark}] {label}: {value}")

    def figure(self, sid: str, fig, name: str, caption: str) -> None:
        plotting.save_figure(fig, self.out / "fig" / f"{name}.png")
        self.stage(sid)["figures"].append({"src": f"fig/{name}.png", "caption": caption})

    def finish(self, sid: str, summary: str) -> None:
        st = self.stage(sid)
        oks = [c["ok"] for c in st["checks"] if c["ok"] is not None]
        st["status"] = "fail" if False in oks else "pass"
        st["summary"] = summary
        st["seconds"] = round(time.time() - st.pop("_t"), 1)
        print(f"  -> {st['status'].upper()}: {summary}")
        self.write()

    def write(self) -> None:
        self.doc["updated"] = datetime.now(UTC).isoformat(timespec="seconds")
        self.doc["elapsed_s"] = round(time.time() - self.t0, 1)
        clean = json.loads(json.dumps(self.doc, default=float))
        for s in clean["stages"]:
            s.pop("_t", None)
        tmp = self.out / "status.json.tmp"
        tmp.write_text(json.dumps(clean, indent=2))
        tmp.replace(self.out / "status.json")


def narrowband_snr_db(freqs: np.ndarray, psd: np.ndarray, f0: float) -> float:
    """Power at the bin nearest f0 vs mean of bins 0.4-1.2 Hz away (classic SSVEP SNR)."""
    k = int(np.argmin(np.abs(freqs - f0)))
    d = np.abs(freqs - freqs[k])
    side = (d >= 0.4) & (d <= 1.2)
    return float(10 * np.log10(psd[k] / np.mean(psd[side])))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--config", default=str(ROOT / "configs" / "benchmark.yaml"))
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "benchmark")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "benchmark_walkthrough")
    ap.add_argument(
        "--pause",
        type=float,
        default=0.0,
        help="seconds to wait between stages (lets a viewer watch progress)",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    bm = cfg.extra["benchmark"]
    onset = float(bm["stimulus_onset_s"])
    win_start = onset + float(bm["visual_latency_s"])
    extended = list(bm["extended_channels"])
    labels = list(cfg.labels)

    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "fig").mkdir(parents=True)
    shutil.copy(PAGE_TEMPLATE, args.out / "index.html")
    st = Status(args.out, args.subjects, cfg)
    st.doc["environment"] = environment_info()
    st.doc["config"] = cfg.to_dict()

    def pause() -> None:
        if args.pause:
            time.sleep(args.pause)

    # 1 ------------------------------------------------------------------ data
    st.start("data")
    for s in args.subjects:
        p = args.data_dir / f"S{s}.mat"
        st.check(
            "data",
            f"S{s}.mat present",
            f"{p.stat().st_size / 1e6:.0f} MB" if p.exists() else "missing",
            benchmark_files_present(args.data_dir, s),
        )
    freqs, _ = read_freq_phase(args.data_dir / "Freq_Phase.mat")
    st.check(
        "data",
        "Targets in Freq_Phase.mat",
        f"{len(freqs)} ({freqs.min():.1f}-{freqs.max():.1f} Hz, 0.2 Hz grid)",
        len(freqs) == 40,
    )
    for lab, f in cfg.targets.items():
        st.check(
            "data",
            f"{f:g} Hz exists in dataset ({lab})",
            "yes" if np.any(np.abs(freqs - f) < 1e-3) else "no",
            bool(np.any(np.abs(freqs - f) < 1e-3)),
        )
    try:
        load_benchmark_subject(
            args.data_dir, args.subjects[0], channels=["Oz"], frequencies=[60 / 7], blocks=[0]
        )
        raised = False
    except ValueError:
        raised = True
    st.check(
        "data",
        "Guardrail: asking for 8.571 Hz (YES) is refused",
        "refused" if raised else "accepted",
        raised,
        "the benchmark has no 8.571 Hz target, so YES/NO/HELP cannot be tested here",
    )
    names = read_channel_names(args.data_dir / "64-channels.loc")
    missing = [c for c in dict.fromkeys([*cfg.channels, *extended]) if c not in names]
    st.check(
        "data",
        "Channels in 64-channels.loc",
        f"{len(names)}; needed "
        f"{'/'.join(cfg.channels)} + 9 occipital: {'all found' if not missing else missing}",
        len(names) == 64 and not missing,
    )
    st.finish(
        "data",
        f"{len(args.subjects)} subjects on disk; 8, 12 and 15 Hz exist; "
        "8.571 Hz correctly refused.",
    )
    pause()

    # 2 ------------------------------------------------------------------ load
    st.start("load")
    load_ch = list(dict.fromkeys([*cfg.channels, *extended]))
    trials = []
    for s in args.subjects:
        ts = load_benchmark_subject(
            args.data_dir, s, channels=load_ch, frequencies=list(cfg.frequencies)
        )
        ts = [
            dataclasses.replace(
                t,
                true_label=next(
                    lab for lab, f in cfg.targets.items() if abs(f - t.true_frequency) < 1e-3
                ),
            )
            for t in ts
        ]
        n_exp = len(cfg.targets) * bmio.N_BLOCKS
        st.check(
            "load",
            f"S{s} trials",
            f"{len(ts)} ({len(cfg.targets)} frequencies x {bmio.N_BLOCKS} blocks)",
            len(ts) == n_exp,
        )
        trials += ts
    shapes = {t.eeg.shape for t in trials}
    st.check(
        "load",
        "Epoch shape (channels, samples)",
        ", ".join(map(str, shapes)),
        shapes == {(len(load_ch), bmio.N_SAMPLES)},
    )
    st.check("load", "Sample rate", f"{trials[0].fs:g} Hz", trials[0].fs == cfg.sample_rate_hz)
    st.check(
        "load",
        "All values finite",
        "yes" if all(np.isfinite(t.eeg).all() for t in trials) else "no",
        all(np.isfinite(t.eeg).all() for t in trials),
    )
    sd_uv = (
        np.median([t.select_channels(cfg.channels).eeg.std(axis=1).mean() for t in trials]) * 1e6
    )
    st.check(
        "load",
        "Typical amplitude (median channel SD)",
        f"{sd_uv:.1f} µV",
        1 <= sd_uv <= 100,
        "scalp EEG is normally a few to tens of µV; confirms µV->V conversion",
    )
    ex = next(
        t
        for t in trials
        if t.subject_id == f"S{args.subjects[0]}"
        and t.block_id == "0"
        and abs(t.true_frequency - 12.0) < 1e-3
    )
    exs = ex.select_channels(cfg.channels)
    st.figure(
        "load",
        plotting.plot_timeseries(
            exs.eeg,
            exs.fs,
            exs.channel_names,
            markers={"flicker on": onset, "flicker off": onset + 5.0},
            shade=(win_start, win_start + cfg.window_seconds),
            shade_label=f"{cfg.window_seconds:g} s analysis window",
            title=f"Raw EEG, {ex.trial_id}: subject looking at the 12 Hz target",
        ),
        "load_raw_trial",
        "Raw EEG of one trial: 0.5 s cue, 5 s of flicker, 0.5 s blank. "
        "The shaded band is the window the decoder analyses (it starts 0.14 s after flicker "
        "onset, the visual latency).",
    )
    st.finish("load", f"{len(trials)} trials loaded as (channels, samples) in volts at 250 Hz.")
    pause()

    # 3 ----------------------------------------------------------------- ssvep
    st.start("ssvep")
    seg = slice(round(win_start * 250), min(round((onset + 5.0) * 250), bmio.N_SAMPLES))
    spectra, snr_table = {}, {}
    for lab, f in cfg.targets.items():
        ps = []
        for t in trials:
            if t.true_label == lab:
                x = t.select_channels(["Oz"]).eeg[0, seg]
                fr, p = sps.periodogram(x - x.mean(), fs=250, window="hann", nfft=2500)
                ps.append(p)
        spectra[f"looking at {f:g} Hz"] = np.mean(ps, axis=0)
        snr_table[lab] = {
            o: narrowband_snr_db(fr, spectra[f"looking at {f:g} Hz"], fo)
            for o, fo in cfg.targets.items()
        }
    for lab, f in cfg.targets.items():
        own = snr_table[lab][lab]
        others = max(v for o, v in snr_table[lab].items() if o != lab)
        st.check(
            "ssvep",
            f"Oz SNR at {f:g} Hz while looking at {f:g} Hz",
            f"{own:.1f} dB (best other target {others:.1f} dB)",
            own > 3 and own > others,
            "diagnostic of the dataset, not a classifier threshold",
        )
    st.figure(
        "ssvep",
        plotting.plot_spectra(
            fr,
            spectra,
            mark_frequencies=cfg.targets,
            harmonics=2,
            fmax=40,
            ylabel="PSD (V²/Hz)",
            title=f"Oz spectrum averaged over trials ({', '.join(st.doc['subjects'])}), unfiltered",
        ),
        "ssvep_spectra",
        "Average spectrum at Oz for each presented frequency. Each line peaks "
        "at its own flicker frequency (dashed) and its 2nd harmonic (dotted); that peak is the "
        "SSVEP the decoder looks for.",
    )
    st.doc["ssvep_snr_db"] = snr_table
    st.finish(
        "ssvep",
        "Clear peaks at each presented frequency: the SSVEP is present in the "
        "raw data before any processing.",
    )
    pause()

    # 4 ------------------------------------------------------------ preprocess
    st.start("preprocess")
    causal = preprocess(exs.eeg, exs.fs, cfg.preprocessing)
    zp = preprocess(exs.eeg, exs.fs, dataclasses.replace(cfg.preprocessing, mode="zero_phase"))
    st.check(
        "preprocess",
        "Shape preserved",
        f"{exs.eeg.shape} -> {causal.shape}",
        causal.shape == exs.eeg.shape,
    )
    st.check(
        "preprocess",
        "Output finite",
        "yes" if np.isfinite(causal).all() else "no",
        bool(np.isfinite(causal).all()),
    )
    f_, p_raw = sps.welch(exs.eeg[:, seg], fs=250, nperseg=len(range(*seg.indices(bmio.N_SAMPLES))))
    _, p_f = sps.welch(causal[:, seg], fs=250, nperseg=len(range(*seg.indices(bmio.N_SAMPLES))))
    # Drift test: inject a known slow drift + electrode offset into the real trial. The filter
    # is linear, so filtered(x + drift) - filtered(x) is exactly its response to the drift.
    # (A short 4.9 s segment cannot resolve < 0.5 Hz spectrally: its bins are 0, 0.2, 0.4 Hz,
    # and 0.4 Hz is in the transition band, so a spectral "below 0.5 Hz" check is ill-posed.)
    tt = np.arange(exs.n_samples) / exs.fs
    drift = 50e-6 * np.sin(2 * np.pi * 0.2 * tt + 0.7) + 200e-6
    wsl = slice(round(win_start * 250), round(win_start * 250) + cfg.window_samples)
    inj = drift[wsl] - drift[wsl].mean()
    for mode in ("causal", "zero_phase"):
        pp = dataclasses.replace(cfg.preprocessing, mode=mode)
        resid = preprocess(exs.eeg + drift, 250, pp) - preprocess(exs.eeg, 250, pp)
        att = 20 * np.log10(np.sqrt(np.mean(resid[:, wsl] ** 2)) / np.sqrt(np.mean(inj**2)))
        if mode == "causal":
            drop = att
        st.check(
            "preprocess",
            f"Injected drift removed, {mode}",
            f"{att:+.1f} dB inside the window (50 µV at 0.2 Hz + 200 µV offset)",
            att <= -20 if mode == "causal" else None,
            "pass = drift amplitude cut at least 10x; steady-state design is -32 dB at "
            "0.2 Hz, the causal filter has only 0.64 s to settle"
            if mode == "causal"
            else "offline reference",
        )
    k12 = int(np.argmin(np.abs(f_ - 12.0)))
    keep = 10 * np.log10(p_f[:, k12].sum() / p_raw[:, k12].sum())
    st.check("preprocess", "Power at 12 Hz (the SSVEP)", f"{keep:+.2f} dB", abs(keep) < 1.0)
    w_c = extract_window(causal, 250, win_start, cfg.window_seconds)
    w_z = extract_window(zp, 250, win_start, cfg.window_seconds)
    r = float(np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(w_c, w_z, strict=True)]))
    st.check(
        "preprocess",
        "Causal vs zero-phase agreement inside window",
        f"r = {r:.3f}",
        None,
        "causal filter has a phase lag and only 0.64 s to settle; both modes are scored "
        "in the next stages",
    )
    st.check(
        "preprocess",
        "Notch",
        "off (benchmark recorded on 50 Hz mains, already notch-filtered)",
        None,
    )
    st.figure(
        "preprocess",
        plotting.plot_psd_before_after(
            exs.eeg,
            causal,
            250,
            exs.channel_names,
            mark_frequencies=cfg.targets,
            harmonics=cfg.harmonics,
            fmax=60,
            title=f"{ex.trial_id}: raw vs causal-filtered (0.5-45 Hz)",
        ),
        "preprocess_psd",
        "Spectrum before and after the causal bandpass. The 12 Hz peak and "
        "its 24 Hz harmonic are unchanged; slow drift below 0.5 Hz and content above 45 Hz "
        "are reduced.",
    )
    st.figure(
        "preprocess",
        plotting.plot_timeseries(
            causal,
            250,
            exs.channel_names,
            markers={"flicker on": onset},
            shade=(win_start, win_start + cfg.window_seconds),
            title=f"{ex.trial_id} after causal filtering",
        ),
        "preprocess_trace",
        "The same trial after filtering: baseline wander is gone and the "
        "rhythmic 12 Hz activity is easier to see inside the analysis window.",
    )
    st.finish("preprocess", f"Drift reduced {drop:+.0f} dB, 12 Hz kept within {abs(keep):.2f} dB.")
    pause()

    # 5 ------------------------------------------------------------------- cca
    st.start("cca")
    d = decode_trial(ex, cfg, window_start_s=win_start)
    res = d.result
    st.check("cca", "Trial", f"{ex.trial_id} (first 12 Hz trial of block 0, not hand-picked)", None)
    st.check(
        "cca",
        "Window",
        f"{d.window.shape[1]} samples = {cfg.window_seconds:g} s from {win_start:.2f} s",
        d.window.shape == (4, cfg.window_samples),
    )
    st.check(
        "cca",
        "Scores finite and in [0, 1]",
        ", ".join(f"{k} {v:.3f}" for k, v in res.scores.items()),
        all(0 <= v <= 1 for v in res.scores.values()),
    )
    st.check(
        "cca",
        "Decision",
        f"{res.decision} ({res.winner_frequency:g} Hz), margin "
        f"{res.margin:.3f}; truth {ex.true_label}",
        res.decision == ex.true_label,
    )
    st.doc["example"] = {
        "trial_id": ex.trial_id,
        "true_label": ex.true_label,
        "true_frequency": ex.true_frequency,
        **res.to_dict(),
        "window_samples": int(d.window.shape[1]),
        "window_start_s": win_start,
    }
    st.figure(
        "cca",
        plotting.plot_cca_scores(
            res.scores,
            true_label=ex.true_label,
            frequencies=cfg.targets,
            title=f"{ex.trial_id}: CCA score per target",
        ),
        "cca_scores",
        "CCA compares the 4-channel window with sine/cosine references at each "
        "target frequency and 2nd harmonic. The highest correlation is the decision.",
    )
    st.finish(
        "cca",
        f"Decided {res.decision} with ρ = {res.confidence:.3f} "
        f"(runner-up {res.confidence - res.margin:.3f}).",
    )
    pause()

    # 6 ----------------------------------------------------------------- smoke
    st.start("smoke")
    smoke = {}
    for mode in ("causal", "zero_phase"):
        c = cfg.replace(preprocessing=dataclasses.replace(cfg.preprocessing, mode=mode))
        res_all = [decode_trial(t, c, window_start_s=win_start).result for t in trials]
        yt = [t.true_label for t in trials]
        yp = [r.decision for r in res_all]
        n_ok = sum(a == b for a, b in zip(yt, yp, strict=True))
        p = binomtest(n_ok, len(yt), 1 / len(labels), alternative="greater").pvalue
        per_sub = {}
        for t, pr in zip(trials, yp, strict=True):
            a = per_sub.setdefault(t.subject_id, [0, 0])
            a[0] += pr == t.true_label
            a[1] += 1
        smoke[mode] = {
            "accuracy": n_ok / len(yt),
            "n_correct": n_ok,
            "n": len(yt),
            "p_value_vs_chance": p,
            "per_subject": {s: a / b for s, (a, b) in per_sub.items()},
        }
        tag = "ESP32 path" if mode == "causal" else "offline reference"
        st.check(
            "smoke",
            f"Accuracy, {mode} ({tag})",
            f"{n_ok}/{len(yt)} = {n_ok / len(yt):.1%}  (chance 33.3 %, p = {p:.1e})",
            p < 0.01,
            "spec exit criterion: meaningfully above chance",
        )
        st.check(
            "smoke",
            f"Per subject, {mode}",
            ", ".join(f"{s} {v:.0%}" for s, v in smoke[mode]["per_subject"].items()),
            None,
        )
        if mode == "causal":
            cm = metrics.confusion_matrix(yt, yp, labels, include_reject=False)
            st.figure(
                "smoke",
                plotting.plot_confusion_matrix(
                    cm,
                    [f"{cfg.targets[lab]:g} Hz" for lab in labels],
                    [f"{cfg.targets[lab]:g} Hz" for lab in labels],
                    title=f"Causal pipeline, {len(yt)} trials",
                ),
                "smoke_confusion",
                "Rows: frequency the subject looked at. Columns: what the decoder chose.",
            )
            dists = {
                f"{cfg.targets[lab]:g} Hz trials": {
                    f"{cfg.targets[k]:g} Hz": [
                        r.scores[k]
                        for t, r in zip(trials, res_all, strict=True)
                        if t.true_label == lab
                    ]
                    for k in labels
                }
                for lab in labels
            }
            st.figure(
                "smoke",
                plotting.plot_score_distributions(
                    dists, title="CCA score per reference, grouped by presented frequency"
                ),
                "smoke_scores",
                "For each group of trials, the matching reference should score "
                "highest. The gap between boxes is what a future margin threshold would use.",
            )
    st.doc["smoke"] = smoke
    st.doc["headline"] = {
        "accuracy_causal": smoke["causal"]["accuracy"],
        "accuracy_zero_phase": smoke["zero_phase"]["accuracy"],
        "n_trials": len(trials),
        "chance": 1 / len(labels),
    }
    st.finish(
        "smoke",
        f"Causal {smoke['causal']['accuracy']:.1%}, zero-phase "
        f"{smoke['zero_phase']['accuracy']:.1%} on {len(trials)} trials "
        "(chance 33 %).",
    )
    pause()

    # 7 ---------------------------------------------------------------- sweeps
    st.start("sweeps")
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_benchmark_validation as rbv

    sweep_root = args.out / "sweeps"
    rbv.main(
        [
            "--config",
            args.config,
            "--data-dir",
            str(args.data_dir),
            "--subjects",
            *map(str, args.subjects),
            "--results-root",
            str(sweep_root),
        ]
    )
    run_dir = next(sweep_root.iterdir())
    m = json.loads((run_dir / "run.json").read_text())["metrics"]
    ws = m["experiment2_window_sweep"]["window_s"]
    acc_w = m["experiment2_window_sweep"]["accuracy"]
    st.check(
        "sweeps",
        "Accuracy vs window (causal)",
        ", ".join(f"{w:g}s {a:.0%}" for w, a in zip(ws, acc_w["causal"], strict=True)),
        None,
    )
    st.check(
        "sweeps",
        "Accuracy vs window (zero-phase)",
        ", ".join(f"{w:g}s {a:.0%}" for w, a in zip(ws, acc_w["zero_phase"], strict=True)),
        None,
    )
    for name, accs in m["experiment3_channels"]["accuracy"].items():
        st.check(
            "sweeps",
            f"Channels: {name}",
            ", ".join(f"{w:g}s {a:.0%}" for w, a in zip(ws, accs, strict=True)),
            None,
        )
    for name, accs in m["experiment4_harmonics"]["accuracy"].items():
        st.check(
            "sweeps",
            f"Harmonics: {name}",
            ", ".join(f"{w:g}s {a:.0%}" for w, a in zip(ws, accs, strict=True)),
            None,
        )
    st.check(
        "sweeps",
        "Trials per point",
        f"{m['n_trials']} (small sample: one trial = {1 / m['n_trials']:.1%})",
        None,
    )
    for name, cap in [
        (
            "accuracy_vs_window",
            "Longer windows give CCA more cycles to lock onto; this is the "
            "latency vs accuracy trade-off for the 2-3 s window choice.",
        ),
        (
            "accuracy_channels",
            "Our 4-channel montage against the 9-channel occipital set the "
            "benchmark literature uses.",
        ),
        ("accuracy_harmonics", "Effect of including 1, 2 or 3 harmonics in the references."),
    ]:
        shutil.copy(run_dir / f"{name}.png", args.out / "fig" / f"sweeps_{name}.png")
        st.stage("sweeps")["figures"].append({"src": f"fig/sweeps_{name}.png", "caption": cap})
    st.doc["sweeps"] = {
        k: m[k]
        for k in ("experiment2_window_sweep", "experiment3_channels", "experiment4_harmonics")
    }
    st.finish("sweeps", "Window, channel and harmonic experiments complete.")

    st.doc["finished"] = True
    st.write()
    print(f"\nwalkthrough: {args.out}")
    return 0 if all(s["status"] == "pass" for s in st.doc["stages"]) else 1


if __name__ == "__main__":
    sys.exit(main())
