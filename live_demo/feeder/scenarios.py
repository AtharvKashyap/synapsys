"""The five demo scenarios and the recorded Wang trial each one plays.

Each scenario is one 6 s Wang epoch (4 channels O1/Oz/O2/POz, 250 Hz, volts) of a volunteer
looking at one flicker frequency. Trials come only from blocks 3-5 (the demo threshold was
chosen on blocks 0-2, see live_demo/analysis/quiet_threshold.py). A trial is used only if the
Python reference, run exactly like the board (causal filter from the stream start, a decision
every 125 samples from 625, T gate, re-arm K=2 R=625 N=2), gives the scenario's expected
outcome. The board matches Python (54/54 verified), so the outcome is known in advance.
So the played trials are pre-screened: they show the pipeline working, not its error rate
(on blocks 3-5 overall, 27/27 target trials were correct and 15/288 = 5.2 % of off-target
trials still produced a command). Trials whose confidence comes within EDGE of the gate at
any decision point are skipped, so float rounding on the board can never flip the outcome.

The selected trials are cached in live_demo/build/scenarios.npz so the stage app starts in
seconds instead of loading 300 MB of .mat files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from synapsys_decoder.config import DecoderConfig, load_config
from synapsys_decoder.io.benchmark import load_benchmark_subject
from synapsys_decoder.rearm import RearmConfig, RearmStateMachine
from synapsys_decoder.streaming import decode_stream
from synapsys_decoder.types import Trial

DEMO = Path(__file__).resolve().parents[1]
ROOT = DEMO.parent
CONFIG_PATH = ROOT / "configs" / "benchmark.yaml"
DATA_DIR = ROOT / "data" / "benchmark"
THRESHOLD_JSON = DEMO / "analysis" / "quiet_threshold.json"
CACHE = DEMO / "build" / "scenarios.npz"

SUBJECTS = (1, 2, 3)
DEMO_BLOCKS = (3, 4, 5)
N_ALTERNATES = 2
EDGE = 0.01  # min |confidence - T| at every decision point (board float error ~1e-4)
REARM = RearmConfig(confirm_count=2, refractory_samples=625, release_count=2)

# Board label (benchmark_v1 config) -> what the audience sees. 8 Hz stands in for YES.
COMMAND_OF_BOARD_LABEL = {"F8_0": "YES", "F12_0": "NO", "F15_0": "HELP"}


@dataclass(frozen=True)
class ScenarioSpec:
    number: int
    tile: str  # "YES" / "NO" / "HELP", or "" for an unlabelled tile
    freq_hz: float
    expected: str | None  # command the phone should show, None = quiet


SPECS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(1, "YES", 8.0, "YES"),
    ScenarioSpec(2, "NO", 12.0, "NO"),
    ScenarioSpec(3, "HELP", 15.0, "HELP"),
    ScenarioSpec(4, "", 10.0, None),
    ScenarioSpec(5, "", 14.0, None),
)


@dataclass
class Scenario:
    spec: ScenarioSpec
    eeg: np.ndarray  # (4, 1500) float64 volts, the trial played first
    source: str  # e.g. "S2 block 4"
    alternates: list[tuple[str, np.ndarray]] = field(default_factory=list)
    onset_sample: int = 125  # stimulus onset inside the epoch (0.5 s)

    @property
    def number(self) -> int:
        return self.spec.number

    @property
    def expected(self) -> str | None:
        return self.spec.expected


def demo_threshold() -> float:
    return float(json.loads(THRESHOLD_JSON.read_text())["confidence_threshold"])


def simulate(
    trial: Trial, cfg: DecoderConfig, threshold: float
) -> tuple[str | None, int, float, float]:
    """Run the trial like the board: (first command as audience label or None, its sample t,
    highest confidence seen, closest approach |confidence - T| over all decision points).
    Quiet trials get t = -1."""
    ss = decode_stream(trial, cfg)
    sm = RearmStateMachine(REARM)
    peak = float(np.nanmax(ss.confidence))
    edge = float(np.nanmin(np.abs(ss.confidence - threshold)))
    for d, t in zip(ss.decisions(threshold, None), ss.window_end, strict=True):
        c = sm.update(d, int(t))
        if c is not None:
            return COMMAND_OF_BOARD_LABEL.get(c, c), int(t), peak, edge
    return None, -1, peak, edge


def expected_outcome(trial: Trial, cfg: DecoderConfig, threshold: float) -> str | None:
    """First command the board would emit for this trial (audience label), or None = quiet."""
    return simulate(trial, cfg, threshold)[0]


def select_scenarios(
    cfg: DecoderConfig | None = None, data_dir: Path = DATA_DIR, threshold: float | None = None
) -> list[Scenario]:
    """Pick, per spec, the demo-block trial with the expected outcome that is safest on stage:
    commands -> the earliest command (shortest wait); quiet -> the lowest peak confidence
    (furthest below the gate). Ties keep S1->S3, block 3->5 order."""
    cfg = cfg or load_config(CONFIG_PATH)
    threshold = demo_threshold() if threshold is None else threshold
    freqs = sorted({s.freq_hz for s in SPECS})
    by_freq: dict[float, list[tuple[str, Trial]]] = {f: [] for f in freqs}
    for subj in SUBJECTS:
        for t in load_benchmark_subject(
            data_dir, subj, channels=list(cfg.channels), frequencies=freqs, blocks=DEMO_BLOCKS
        ):
            f = round(float(t.true_frequency), 1)
            by_freq[f].append((f"S{subj} block {t.block_id}", t))
    out = []
    for spec in SPECS:
        ranked = []
        for order, (src, t) in enumerate(by_freq[spec.freq_hz]):
            got, t_cmd, peak, edge = simulate(t, cfg, threshold)
            if got == spec.expected and edge > EDGE:
                key = (t_cmd if spec.expected else peak, order)
                ranked.append((key, src, t.eeg))
        good = [(src, eeg) for _, src, eeg in sorted(ranked, key=lambda r: r[0])]
        if not good:
            raise RuntimeError(f"no demo trial gives the expected outcome for scenario {spec}")
        (src, eeg), *rest = good
        out.append(Scenario(spec, eeg, src, alternates=rest[:N_ALTERNATES]))
    return out


def save_cache(scenarios: list[Scenario], path: Path = CACHE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays, meta = {}, []
    for sc in scenarios:
        arrays[f"s{sc.number}_0"] = sc.eeg
        srcs = [sc.source]
        for k, (src, eeg) in enumerate(sc.alternates, start=1):
            arrays[f"s{sc.number}_{k}"] = eeg
            srcs.append(src)
        meta.append({"number": sc.number, "sources": srcs})
    np.savez(path, meta=json.dumps(meta), threshold=demo_threshold(), edge_v=1, **arrays)


def load_scenarios(path: Path = CACHE) -> list[Scenario]:
    """Cached scenarios; builds the cache from data/benchmark on first use."""
    if not path.exists():
        save_cache(select_scenarios(), path)
    with np.load(path) as z:
        if abs(float(z["threshold"]) - demo_threshold()) > 1e-12 or int(z.get("edge_v", 0)) != 1:
            raise RuntimeError(
                f"{path} was built for another threshold; rebuild it with "
                "`cd live_demo && ../.venv/bin/python -m feeder.scenarios --rebuild`"
            )
        meta = {m["number"]: m["sources"] for m in json.loads(str(z["meta"]))}
        out = []
        for spec in SPECS:
            srcs = meta[spec.number]
            eegs = [z[f"s{spec.number}_{k}"] for k in range(len(srcs))]
            out.append(
                Scenario(
                    spec, eegs[0], srcs[0], alternates=list(zip(srcs[1:], eegs[1:], strict=True))
                )
            )
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Select and cache the demo trials.")
    ap.add_argument("--rebuild", action="store_true", help="re-select from data/benchmark")
    args = ap.parse_args()
    if args.rebuild or not CACHE.exists():
        save_cache(select_scenarios())
    for sc in load_scenarios():
        alts = ", ".join(src for src, _ in sc.alternates) or "none"
        print(
            f"{sc.number}: {sc.spec.tile or '-':<4} {sc.spec.freq_hz:4.1f} Hz  expected "
            f"{sc.expected or 'quiet':<5} trial {sc.source} (alternates: {alts})"
        )


if __name__ == "__main__":
    main()
