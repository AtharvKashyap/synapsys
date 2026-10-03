"""Generate every artifact for the weekly status report, from scratch, in one folder.

Outputs (results/<run_id>_report_artifacts/ unless --out is given):
  INDEX.md                     what each file is and which report placeholder it fills
  repo_tree.txt / .png         repository structure (git-tracked files)
  pytest_output.txt            full verbose pytest log (every test name + PASSED/FAILED)
  test_summary.md / .png       tests per file with what each file verifies
  cca_examples.csv / .md / .png  synthetic trials: true f, rho per target, prediction
  filter_response.png          designed DSP magnitude response, targets vs interferers
  filter_response_table.md     the same response in dB at key frequencies
  psd_before_after.png         synthetic 12 Hz trial with drift + 60 Hz, raw vs filtered
  synthetic/<run>/             full synthetic validation run (accuracy vs SNR, etc.)
  mutation_report.md           planted-bug check (skip with --skip-mutation; ~1 min)

All synthetic data use fixed seeds, so re-running reproduces identical numbers.
Synthetic results verify pipeline behaviour only; they are not human EEG performance.

Usage:
    python scripts/make_report_artifacts.py [--out DIR] [--skip-mutation]
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from synapsys_decoder import plotting, synthetic
from synapsys_decoder.config import load_config
from synapsys_decoder.pipeline import decode_trial
from synapsys_decoder.preprocessing import build_filter_sos, frequency_response_db
from synapsys_decoder.runlog import make_run_id

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261003

TEST_FILE_PURPOSE = {
    "test_config.py": "YAML config loading, validation, YES/NO label guard",
    "test_references.py": "sin/cos reference shapes, values, FFT frequency (incl. 60/7 Hz)",
    "test_cca.py": "CCA bounds, matched > unmatched, invariances, 2 independent cross-checks",
    "test_classifier.py": "3-class decisions, margin, gating/REJECT, tie-break",
    "test_preprocessing.py": "filter dB response, 12 Hz kept / drift-60-80 Hz removed, causality",
    "test_synthetic.py": "synthetic generator: SNR accuracy, frequencies, determinism",
    "test_metrics.py": "accuracy, confusion matrix, REJECT handling, ITR",
    "test_plotting.py": "every figure renders and saves",
    "test_pipeline.py": "end-to-end decode, channel order, gating, run artifacts, scripts",
    "test_io_benchmark.py": "public-dataset loader (fake files): channels, units, frequencies",
    "test_io_csv.py": "CSV recording/fixture format round-trip",
    "test_benchmark_script.py": "benchmark experiment script end-to-end (fake data)",
}


def text_figure(text: str, title: str) -> Figure:
    lines = text.splitlines()
    fig = Figure(figsize=(7.5, 0.5 + 0.17 * len(lines)), layout="constrained")
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(
        0,
        1,
        text,
        family="monospace",
        fontsize=8.5,
        va="top",
        ha="left",
        color=plotting.INK,
        transform=ax.transAxes,
    )
    ax.set_title(title, fontsize=11, loc="left", color=plotting.INK)
    return fig


def repo_tree() -> str:
    files = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    files = [f for f in files if not f.startswith("docs/figures/")] + ["docs/figures/..."]
    tree: dict = {}
    for f in sorted(files):
        node = tree
        for part in f.split("/"):
            node = node.setdefault(part, {})
    out = [ROOT.name + "/"]

    def walk(node: dict, prefix: str) -> None:
        items = sorted(node.items(), key=lambda kv: (not kv[1], kv[0]))  # dirs first
        for i, (name, child) in enumerate(items):
            last = i == len(items) - 1
            out.append(f"{prefix}{'└── ' if last else '├── '}{name}{'/' if child else ''}")
            walk(child, prefix + ("    " if last else "│   "))

    walk(tree, "")
    return "\n".join(out)


def run_tests(out: Path) -> tuple[str, list[list[str]]]:
    junit = out / "junit.xml"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider", f"--junitxml={junit}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    (out / "pytest_output.txt").write_text(proc.stdout + proc.stderr)
    total, failed, skipped = Counter(), Counter(), Counter()
    for case in ET.parse(junit).getroot().iter("testcase"):
        fname = case.get("classname", "").split(".")[-1] + ".py"
        total[fname] += 1
        if case.find("failure") is not None or case.find("error") is not None:
            failed[fname] += 1
        if case.find("skipped") is not None:
            skipped[fname] += 1
    junit.unlink()
    rows = []
    for fname in TEST_FILE_PURPOSE:
        if total[fname]:
            passed = total[fname] - failed[fname] - skipped[fname]
            note = f" ({skipped[fname]} skipped: needs real data)" if skipped[fname] else ""
            rows.append([fname, TEST_FILE_PURPOSE[fname], str(total[fname]), f"{passed}{note}"])
    n, nf, ns = sum(total.values()), sum(failed.values()), sum(skipped.values())
    summary = f"{n - nf - ns} passed, {nf} failed, {ns} skipped (of {n})"
    return summary, rows


def cca_examples(cfg, rng) -> tuple[list[str], list[list[str]], list[int]]:
    """Several synthetic trials per command at increasing noise; full CCA scores."""
    conditions = [("clean", None), ("SNR -15 dB", -15), ("SNR -20 dB", -20), ("SNR -25 dB", -25)]
    header = [
        "Condition",
        "True command",
        *[f"ρ {f:.2f} Hz ({lab})" for lab, f in cfg.targets.items()],
        "Predicted",
        "Margin",
        "OK",
    ]
    rows, wrong = [], []
    for cond, snr in conditions:
        for label, f in cfg.targets.items():
            kw = (
                {}
                if snr is None
                else dict(
                    snr_db=snr, drift_amplitude_v=20e-6, line_amplitude_v=5e-6, channel_scaling=True
                )
            )
            t = synthetic.make_synthetic_trial(
                f,
                cfg.sample_rate_hz,
                6.0,
                cfg.channels,
                rng,
                harmonics=cfg.harmonics,
                label=label,
                **kw,
            )
            r = decode_trial(t, cfg).result
            ok = r.decision == label
            if not ok:
                wrong.append(len(rows))
            rows.append(
                [
                    cond,
                    f"{label} ({f:.2f} Hz)",
                    *[f"{r.scores[lab]:.3f}" for lab in cfg.labels],
                    f"{r.decision} ({r.winner_frequency:.2f} Hz)",
                    f"{r.margin:.3f}",
                    "✓" if ok else "✗",
                ]
            )
    return header, rows, wrong


def md_table(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--skip-mutation", action="store_true")
    args = ap.parse_args(argv)
    out = args.out or ROOT / "results" / make_run_id("report_artifacts")
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs" / "synapsys.yaml")
    rng = np.random.default_rng(SEED)
    fs = cfg.sample_rate_hz

    # 1. Repository structure
    tree = repo_tree()
    (out / "repo_tree.txt").write_text(tree + "\n")
    plotting.save_figure(
        text_figure(tree, "Repository structure (git-tracked)"), out / "repo_tree.png"
    )

    # 2. Tests
    summary, rows = run_tests(out)
    (out / "test_summary.md").write_text(
        f"**pytest: {summary}**\n\n"
        + md_table(["Test file", "Verifies", "Tests", "Passed"], rows)
        + "\n"
    )
    plotting.save_figure(
        plotting.plot_table(
            ["Test file", "Verifies", "Tests", "Passed"],
            rows,
            title=f"pytest: {summary}",
        ),
        out / "test_summary.png",
    )
    print(f"tests: {summary}")

    # 3. Example CCA output
    header, rows, wrong = cca_examples(cfg, rng)
    with (out / "cca_examples.csv").open("w", newline="") as fh:
        csv.writer(fh).writerows([header, *rows])
    note = (
        "Synthetic 4-channel trials (O1/Oz/O2/POz), 2.5 s window, causal filtering. "
        "Noisy rows add white noise at the stated broadband SNR plus 20 µV drift and "
        "5 µV 60 Hz interference. ρ = largest canonical correlation per target; the "
        "prediction is the largest ρ. No rejection thresholds are set."
    )
    (out / "cca_examples.md").write_text(md_table(header, rows) + f"\n\n{note}\n")
    plotting.save_figure(
        plotting.plot_table(
            header,
            rows,
            title="Example CCA output (synthetic trials, known true frequency)",
            highlight_rows=wrong,
        ),
        out / "cca_examples.png",
    )
    print(f"cca examples: {len(rows) - len(wrong)}/{len(rows)} correct")

    # 4. DSP verification: designed response + PSD before/after
    freqs = np.geomspace(0.1, 120, 4000)
    full = build_filter_sos(fs, cfg.preprocessing)
    no_notch = build_filter_sos(
        fs, cfg.preprocessing.__class__(**{**cfg.preprocessing.__dict__, "notch_hz": None})
    )
    plotting.save_figure(
        plotting.plot_filter_response(
            freqs,
            {
                "bandpass 0.5-45 Hz + 60 Hz notch (configured)": frequency_response_db(
                    full, fs, freqs
                ),
                "bandpass only": frequency_response_db(no_notch, fs, freqs),
            },
            keep_frequencies={lab: f for lab, f in cfg.targets.items()},
            reject_frequencies={"drift 0.2": 0.2, "mains 60": 60.0, "80 Hz": 80.0},
            log_freq=True,
            title="Designed preprocessing response (4th-order Butterworth bandpass, Q=30 notch)",
        ),
        out / "filter_response.png",
    )
    key = [0.2, 0.5, *cfg.frequencies, *(2 * f for f in cfg.frequencies), 45, 60, 80]
    db_full = frequency_response_db(full, fs, np.array(key))
    db_nn = frequency_response_db(no_notch, fs, np.array(key))
    (out / "filter_response_table.md").write_text(
        md_table(
            ["Frequency (Hz)", "Gain, configured (dB)", "Gain, bandpass only (dB)"],
            [
                [f"{f:.3f}", f"{a:.2f}" if a > -100 else "< -100", f"{b:.2f}"]
                for f, a, b in zip(key, db_full, db_nn, strict=True)
            ],
        )
        + "\n"
    )

    t = synthetic.make_synthetic_trial(
        cfg.targets["NO"],
        fs,
        6.0,
        cfg.channels,
        rng,
        harmonics=cfg.harmonics,
        label="NO",
        snr_db=-10,
        drift_amplitude_v=20e-6,
        line_amplitude_v=5e-6,
        channel_scaling=True,
    )
    d = decode_trial(t, cfg)
    plotting.save_figure(
        plotting.plot_psd_before_after(
            t.eeg,
            d.filtered,
            fs,
            cfg.channels,
            mark_frequencies=cfg.targets,
            harmonics=cfg.harmonics,
            fmax=70,
            title="Synthetic 12 Hz (NO) trial + 0.2 Hz drift + 60 Hz: raw vs causal-filtered",
        ),
        out / "psd_before_after.png",
    )

    # 5. Full synthetic validation (accuracy vs SNR etc.)
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_synthetic_validation

    run_synthetic_validation.main(["--results-root", str(out / "synthetic")])
    synth_dir = next((out / "synthetic").iterdir())

    # 6. Mutation check
    if not args.skip_mutation:
        import mutation_check

        mutation_check.main(["--out", str(out)])

    index = f"""# Report artifacts

Generated by `python scripts/make_report_artifacts.py` (seed {SEED}; re-running
reproduces the same numbers). Synthetic results check pipeline behaviour only and are
not human-EEG performance.

| Report placeholder | File(s) |
|---|---|
| Screenshot: repository structure | `repo_tree.png` (text: `repo_tree.txt`) |
| Screenshot: pytest output | `test_summary.png`; full log `pytest_output.txt` ({summary}) |
| Table: True f / ρ8.57 / ρ12 / ρ15 / Predicted | `cca_examples.png` / `.md` / `.csv` |
| Graph: PSD before/after preprocessing | `psd_before_after.png` |
| (extra) DSP design evidence | `filter_response.png`, `filter_response_table.md` |
| (extra) accuracy degrades with noise | `synthetic/{synth_dir.name}/accuracy_vs_snr.png` |
| (extra) clean-trial CCA bars per command | `synthetic/{synth_dir.name}/clean_cca_scores.png` |
| (extra) proof the tests catch bugs | `mutation_report.md` |
"""
    (out / "INDEX.md").write_text(index)
    print(f"artifacts: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
