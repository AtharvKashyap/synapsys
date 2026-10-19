"""Choose the gating thresholds T (confidence) and M (margin) on own-system calibration data.

CCA itself is not trained. The only data-driven choices are T and M (and the re-arm
parameters), and they must come from **own-system calibration recordings** of real people,
never from the final test set, synthetic data or the Wang benchmark.

Calibration set (one recording per CSV, label in the sidecar ``true_label``):
    * target recordings: the participant fixates one target for the whole recording
      (``true_label`` = a configured label, e.g. ``"YES"``);
    * rest recordings: no target fixated (eyes on a blank/neutral area, looking around,
      blinking, talking) with ``true_label = "REST"``. Rest data is required: without it
      the false-activation rate, which T and M exist to control, cannot be measured.

Every recording is decoded on the device schedule (:mod:`synapsys_decoder.streaming`),
dropping decision points whose window starts within ``settle_seconds`` of the recording
start (the causal filter starts cold there; the device's filter has been running for a
long time). For each (T, M) on a grid the decisions are gated, passed through the re-arm
state machine, and summarised at two levels:

    window level   accepted-correct / accepted-wrong / rejected fractions of target
                   points; accepted fraction of rest points
    command level  per target recording: is the first emitted command correct, wrong, or
                   absent; wrong commands in total; rest false commands per minute

Selection (:func:`select_thresholds`) is a stated rule, not a search for the best test
number: among grid points whose rest false-command rate is at most the team's limit (and
whose wrong-first-command fraction is at most ``max_wrong_fraction`` if given), take the
highest fraction of target recordings whose first command is correct; ties go to fewer
wrong commands, then fewer rest false commands, then the larger T, then the larger M (the
more conservative gate). Evaluate the chosen pair once on separate held-out recordings.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from synapsys_decoder.config import DecoderConfig
from synapsys_decoder.rearm import RearmConfig, RearmStateMachine
from synapsys_decoder.streaming import StreamScores, decode_stream
from synapsys_decoder.types import REJECT, Trial

REST_LABEL = "REST"
FORBIDDEN_SPLITS = ("test", "heldout", "held_out", "evaluation")


@dataclass(frozen=True)
class CalibrationRecording:
    name: str
    true_label: str  # a configured label or REST_LABEL
    scores: StreamScores  # usable decision points only (after settling)

    @property
    def is_rest(self) -> bool:
        return self.true_label == REST_LABEL

    @property
    def minutes(self) -> float:
        """Time covered by the usable decision points (one hop per point)."""
        if len(self.scores.window_end) < 2:
            return 0.0
        hop = int(self.scores.window_end[1] - self.scores.window_end[0])
        return len(self.scores.window_end) * hop / self.scores.fs / 60.0


def prepare_recording(
    trial: Trial,
    cfg: DecoderConfig,
    *,
    name: str,
    hop_samples: int = 125,
    settle_seconds: float = 3.0,
    purpose: str = "calibration",
) -> CalibrationRecording:
    """Validate one recording and decode it (usable points only).

    ``purpose="calibration"`` refuses test/held-out recordings; ``purpose="evaluation"``
    (the single held-out run) accepts only recordings marked ``split: test``.

    Raises:
        ValueError: unknown/missing label, a split that does not fit ``purpose``, or too
            short to give at least two usable decision points.
    """
    if purpose not in ("calibration", "evaluation"):
        raise ValueError(f"purpose must be 'calibration' or 'evaluation'; got {purpose!r}")
    label = trial.true_label
    if label is None:
        raise ValueError(f"{name}: no true_label (set it in the sidecar: a target or 'REST')")
    if label != REST_LABEL and label not in cfg.labels:
        raise ValueError(f"{name}: true_label {label!r} is not a target {cfg.labels} or 'REST'")
    split = str(trial.metadata.get("split", "")).lower()
    if purpose == "calibration" and split in FORBIDDEN_SPLITS:
        raise ValueError(
            f"{name}: metadata.split={split!r}. Thresholds must never be chosen on test data."
        )
    if purpose == "evaluation" and split != "test":
        raise ValueError(f"{name}: metadata.split={split!r}; held-out evaluation needs 'test'")
    if settle_seconds < 0:
        raise ValueError("settle_seconds must be >= 0")
    full = decode_stream(trial, cfg, hop_samples=hop_samples)
    starts = full.window_end - cfg.window_samples
    keep = starts >= round(settle_seconds * trial.fs)
    if keep.sum() < 2:
        need = settle_seconds + cfg.window_seconds + hop_samples / trial.fs
        raise ValueError(
            f"{name}: {trial.duration_s:.1f} s gives {int(keep.sum())} usable decision "
            f"point(s); record at least {need:.1f} s"
        )
    usable = StreamScores(
        labels=full.labels,
        window_end=full.window_end[keep],
        scores=full.scores[keep],
        fs=full.fs,
    )
    return CalibrationRecording(name=name, true_label=label, scores=usable)


def _commands(decisions: Sequence[str], ends: np.ndarray, rearm: RearmConfig) -> list[str]:
    sm = RearmStateMachine(rearm)
    out = []
    for d, t in zip(decisions, ends, strict=True):
        c = sm.update(d, int(t))
        if c is not None:
            out.append(c)
    return out


def evaluate_thresholds(
    recordings: Sequence[CalibrationRecording],
    confidence_threshold: float | None,
    margin_threshold: float | None,
    rearm: RearmConfig | None = None,
) -> dict[str, float]:
    """Window- and command-level summary for one (T, M). See module docstring."""
    rearm = rearm or RearmConfig()
    w_n = w_ok = w_wrong = 0
    r_n = r_acc = 0
    trials = first_ok = first_wrong = first_none = wrong_cmds = ok_cmds = 0
    rest_cmds, rest_min, target_min = 0, 0.0, 0.0
    for rec in recordings:
        ss = rec.scores
        dec = ss.decisions(confidence_threshold, margin_threshold)
        cmds = _commands(dec, ss.window_end, rearm)
        if rec.is_rest:
            r_n += len(dec)
            r_acc += sum(d != REJECT for d in dec)
            rest_cmds += len(cmds)
            rest_min += rec.minutes
            continue
        w_n += len(dec)
        w_ok += sum(d == rec.true_label for d in dec)
        w_wrong += sum(d not in (REJECT, rec.true_label) for d in dec)
        trials += 1
        target_min += rec.minutes
        ok_cmds += sum(c == rec.true_label for c in cmds)
        wrong_cmds += sum(c != rec.true_label for c in cmds)
        if not cmds:
            first_none += 1
        elif cmds[0] == rec.true_label:
            first_ok += 1
        else:
            first_wrong += 1

    def frac(a: int, b: int) -> float:
        return a / b if b else float("nan")

    return {
        "confidence_threshold": np.nan if confidence_threshold is None else confidence_threshold,
        "margin_threshold": np.nan if margin_threshold is None else margin_threshold,
        "target_points": w_n,
        "point_correct": frac(w_ok, w_n),
        "point_wrong": frac(w_wrong, w_n),
        "point_rejected": frac(w_n - w_ok - w_wrong, w_n),
        "rest_points": r_n,
        "rest_point_accepted": frac(r_acc, r_n),
        "target_recordings": trials,
        "first_command_correct": frac(first_ok, trials),
        "first_command_wrong": frac(first_wrong, trials),
        "no_command": frac(first_none, trials),
        "correct_commands": ok_cmds,
        "wrong_commands": wrong_cmds,
        "wrong_commands_per_min": wrong_cmds / target_min if target_min else float("nan"),
        "rest_commands": rest_cmds,
        "rest_minutes": rest_min,
        "rest_commands_per_min": rest_cmds / rest_min if rest_min else float("nan"),
    }


def threshold_grid(
    t_values: Iterable[float] | None = None, m_values: Iterable[float] | None = None
) -> list[tuple[float, float]]:
    """Default grid: T in 0.00..1.00 and M in 0.00..0.30, both in steps of 0.01.

    T = 0 and M = 0 are equivalent to "gate off" (scores are >= 0, margins >= 0).
    """
    ts = np.round(np.arange(0.0, 1.0001, 0.01), 4) if t_values is None else list(t_values)
    ms = np.round(np.arange(0.0, 0.3001, 0.01), 4) if m_values is None else list(m_values)
    return [(float(t), float(m)) for t in ts for m in ms]


def sweep(
    recordings: Sequence[CalibrationRecording],
    grid: Sequence[tuple[float, float]],
    rearm: RearmConfig | None = None,
) -> list[dict[str, float]]:
    check_calibration_set(recordings)
    return [evaluate_thresholds(recordings, t, m, rearm) for t, m in grid]


def check_calibration_set(recordings: Sequence[CalibrationRecording]) -> None:
    """Refuse a set without rest data or without any target recording."""
    if not any(r.is_rest for r in recordings):
        raise ValueError(
            "no REST recordings: false activations cannot be measured, so T/M cannot be chosen"
        )
    if not any(not r.is_rest for r in recordings):
        raise ValueError("no target recordings in the calibration set")


def select_thresholds(
    rows: Sequence[dict[str, float]],
    *,
    max_rest_commands_per_min: float,
    max_wrong_fraction: float | None = None,
) -> dict[str, float] | None:
    """Apply the stated selection rule (module docstring); ``None`` if no point qualifies."""
    if not (max_rest_commands_per_min >= 0):
        raise ValueError("max_rest_commands_per_min must be >= 0")
    ok = [
        r
        for r in rows
        if r["rest_commands_per_min"] <= max_rest_commands_per_min
        and (max_wrong_fraction is None or r["first_command_wrong"] <= max_wrong_fraction)
    ]
    if not ok:
        return None
    return max(
        ok,
        key=lambda r: (
            r["first_command_correct"],
            -r["wrong_commands"],
            -r["rest_commands"],
            r["confidence_threshold"],
            r["margin_threshold"],
        ),
    )
