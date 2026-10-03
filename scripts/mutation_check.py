"""Prove the test suite catches real bugs: plant known defects, confirm tests fail.

For each mutation, the repository (src/tests/configs/scripts/pyproject) is copied
to a temporary directory, ONE deliberate bug is injected there, and the full
pytest suite is run against the copy. The working tree is never modified.

A mutation is "caught" if at least one test fails. A suite that passes on correct
code AND fails on every planted bug is evidence the tests check behaviour, not
just that code runs.

Usage:
    python scripts/mutation_check.py [--out results/<dir>]   # writes mutation_report.md
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COPY = ("src", "tests", "configs", "scripts", "pyproject.toml")


@dataclass(frozen=True)
class Mutation:
    name: str
    file: str  # relative to src/synapsys_decoder
    edits: tuple[tuple[str, str], ...]  # (exact old text, new text)
    why: str  # what real bug this simulates


MUTATIONS = (
    Mutation(
        "ref_sin_becomes_cos",
        "references.py",
        (("out[2 * (h - 1)] = np.sin(phase)", "out[2 * (h - 1)] = np.cos(phase)"),),
        "reference generator emits cos instead of sin (CCA loses the quadrature pair)",
    ),
    Mutation(
        "ref_no_harmonics",
        "references.py",
        (
            (
                "phase = 2.0 * np.pi * h * float(frequency_hz) * t",
                "phase = 2.0 * np.pi * float(frequency_hz) * t",
            ),
        ),
        "every 'harmonic' row is generated at the fundamental",
    ),
    Mutation(
        "ref_wrong_time_base",
        "references.py",
        (
            (
                "t = np.arange(n_samples, dtype=np.float64) / float(fs)",
                "t = np.arange(n_samples, dtype=np.float64) / (float(fs) / 2)",
            ),
        ),
        "sample-rate bug: references at twice the intended frequency",
    ),
    Mutation(
        "cca_no_centering",
        "cca.py",
        (("xc = x - x.mean(axis=1, keepdims=True)", "xc = x"),),
        "EEG not mean-centred before CCA (DC offsets corrupt correlations)",
    ),
    Mutation(
        "classifier_picks_lowest",
        "classifier.py",
        (("best = -np.inf", "best = np.inf"), ("if value > best:", "if value < best:")),
        "argmax replaced by argmin: the least-correlated command wins",
    ),
    Mutation(
        "classifier_margin_wrong",
        "classifier.py",
        (("margin = best - max(others)", "margin = best - min(others)"),),
        "margin computed against the worst class instead of the runner-up",
    ),
    Mutation(
        "gating_inverted",
        "classifier.py",
        (("best < confidence_threshold", "best > confidence_threshold"),),
        "confidence gate rejects confident windows and accepts weak ones",
    ),
    Mutation(
        "causal_mode_uses_future",
        "preprocessing.py",
        (("return signal.sosfilt(sos, x, axis=-1)", "return signal.sosfiltfilt(sos, x, axis=-1)"),),
        "'causal' path silently uses future samples (would not port to ESP32)",
    ),
    Mutation(
        "notch_mistuned",
        "preprocessing.py",
        (
            (
                "b, a = signal.iirnotch(notch_hz, q, fs=fs)",
                "b, a = signal.iirnotch(notch_hz + 5.0, q, fs=fs)",
            ),
        ),
        "60 Hz notch designed at 65 Hz",
    ),
    Mutation(
        "pipeline_skips_filtering",
        "pipeline.py",
        (("window = extract_window(filtered,", "window = extract_window(selected.eeg,"),),
        "classifier fed the unfiltered signal (preprocessing silently bypassed)",
    ),
    Mutation(
        "yaml_yes_no_guard_removed",
        "config.py",
        (("        if bad:\n", "        if False:\n"),),
        "YAML YES/NO-as-boolean labels accepted silently",
    ),
)


def run_pytest(workdir: Path) -> tuple[int, int, list[str]]:
    """Run the suite in ``workdir``; return (n_failed, n_passed, failing test ids)."""
    env = {**os.environ, "PYTHONPATH": str(workdir / "src"), "MPLBACKEND": "Agg"}
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rf"],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
    )
    out = proc.stdout + proc.stderr
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", out)) else 0
    errors = int(m.group(1)) if (m := re.search(r"(\d+) error", out)) else 0
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", out)) else 0
    ids = re.findall(r"^FAILED (\S+)", out, flags=re.M)
    return failed + errors, passed, ids


def fresh_copy(dst: Path) -> None:
    for item in COPY:
        src = ROOT / item
        if src.is_dir():
            shutil.copytree(src, dst / item, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(src, dst / item)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None, help="directory for mutation_report.md")
    args = ap.parse_args(argv)

    lines = [
        "# Mutation check: does the test suite catch planted bugs?",
        "",
        "Each row: one deliberate bug injected into a temporary copy of the code, "
        "then the full pytest suite run against it. The real working tree is untouched.",
        "",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "baseline"
        fresh_copy(base)
        f0, p0, _ = run_pytest(base)
        print(f"baseline (unmodified code): {p0} passed, {f0} failed")
        if f0:
            print("baseline must pass before mutation testing means anything")
            return 1
        lines += [
            f"**Baseline (unmodified code): {p0} passed, {f0} failed.**",
            "",
            "| # | Planted bug | Simulates | Tests failed | Caught | Example failing test |",
            "|---|---|---|---|---|---|",
        ]
        caught = 0
        for i, m in enumerate(MUTATIONS, 1):
            work = Path(tmp) / m.name
            fresh_copy(work)
            target = work / "src" / "synapsys_decoder" / m.file
            text = target.read_text()
            for old, new in m.edits:
                if text.count(old) != 1:
                    raise RuntimeError(f"{m.name}: pattern not found exactly once: {old!r}")
                text = text.replace(old, new)
            target.write_text(text)
            failed, _passed, ids = run_pytest(work)
            ok = failed > 0
            caught += ok
            example = ids[0].split("::", 1)[-1] if ids else "-"
            print(f"{i:2d}. {m.name:<28} failed={failed:<3} {'CAUGHT' if ok else 'MISSED'}")
            lines.append(
                f"| {i} | `{m.name}` ({m.file}) | {m.why} | {failed} | "
                f"{'yes' if ok else '**NO**'} | `{example}` |"
            )
    lines += ["", f"**{caught}/{len(MUTATIONS)} planted bugs caught.**", ""]
    report = "\n".join(lines)
    print(f"\n{caught}/{len(MUTATIONS)} planted bugs caught")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "mutation_report.md").write_text(report)
        print(f"wrote {args.out / 'mutation_report.md'}")
    return 0 if caught == len(MUTATIONS) else 2


if __name__ == "__main__":
    sys.exit(main())
