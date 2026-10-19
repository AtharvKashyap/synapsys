"""Write fixtures/rearm/sequences.csv: decision sequences + expected re-arm output.

Usage: python scripts/make_rearm_fixtures.py [out.csv]
Deterministic (seeded). Used by the C++ port (Gate 4).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from synapsys_decoder.rearm import RearmConfig, RearmStateMachine
from synapsys_decoder.types import REJECT

SEED = 20261018
LABELS = ["YES", "NO", "HELP"]
HEADER = (
    "sequence_id,confirm_count,refractory_samples,release_count,"
    "step,window_end_sample,decision,expected_emit,expected_state"
)

CONFIGS = {
    "default": RearmConfig(),
    "k1": RearmConfig(confirm_count=1),
    "k3": RearmConfig(confirm_count=3, refractory_samples=1250, release_count=3),
    "n0": RearmConfig(release_count=0),
}

Y, N, H, X = "YES", "NO", "HELP", REJECT
HAND = {
    "basic": [Y, Y],
    "stare_long": [Y] * 30,
    "reject_release": [Y, Y, X, X, X, X, X, Y, Y, Y],
    "label_release": [Y, Y, N, H, N, H, N, Y, Y],
    "interrupted_candidate": [Y, X, Y, X, Y, Y],
    "switch_candidate": [Y, N, H, H, N, N],
    "early_release": [Y, Y, X, X, X, X, X, X, Y, Y],
    "blocked_other_label": [Y, Y, N, N, N, N, N, N, N, N],
    "all_reject": [X] * 8,
    "alternating": [Y, N] * 8,
    "stare_with_blips": [Y, Y, Y, X, Y, Y, X, Y, Y, Y, Y, Y, Y, Y],
}


def _random_sequence(rng: np.random.Generator, n: int) -> list[str]:
    seq: list[str] = []
    while len(seq) < n:
        label = str(rng.choice([*LABELS, REJECT], p=[0.3, 0.3, 0.2, 0.2]))
        run = int(rng.integers(1, 12))
        for _ in range(run):
            # occasional single-point flicker inside a run
            seq.append(label if rng.random() > 0.08 else str(rng.choice([*LABELS, REJECT])))
    return seq[:n]


def build_sequences() -> list[tuple[str, list[str]]]:
    rng = np.random.default_rng(SEED)
    seqs = [(f"hand_{k}", v) for k, v in HAND.items()]
    for i in range(24):
        seqs.append((f"rand_{i:02d}", _random_sequence(rng, int(rng.integers(30, 80)))))
    return seqs


def write_fixtures(path: Path) -> None:
    lines = [HEADER]
    for cfg_name, cfg in CONFIGS.items():
        for name, seq in build_sequences():
            sid = f"{cfg_name}/{name}"
            sm = RearmStateMachine(cfg)
            for i, d in enumerate(seq):
                t = 625 + 125 * i
                emit = sm.update(d, t)
                lines.append(
                    f"{sid},{cfg.confirm_count},{cfg.refractory_samples},"
                    f"{cfg.release_count},{i},{t},{d},{emit or ''},{sm.state}"
                )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\n".join(lines) + "\n").encode())


def main() -> None:
    default = Path(__file__).resolve().parents[1] / "fixtures" / "rearm" / "sequences.csv"
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    write_fixtures(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
