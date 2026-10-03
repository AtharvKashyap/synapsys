"""Canonical data types shared by every decoder module.

Conventions (never change silently):

* EEG arrays are ``(n_channels, n_samples)``, float64, in **volts**.
* Channel order is whatever ``channel_names`` says; code selects by name.
* Class order is the insertion order of the configured ``targets`` mapping.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Decision label emitted when gating rejects a window.
REJECT = "REJECT"


@dataclass(frozen=True)
class Trial:
    """One multichannel EEG epoch plus its provenance.

    Attributes:
        eeg: ``(n_channels, n_samples)`` array in volts.
        fs: Sample rate in Hz.
        channel_names: One name per row of ``eeg``.
        true_frequency: Presented stimulus frequency in Hz, if known.
        true_label: Class label (e.g. ``"NO"``), if known.
        subject_id / block_id / trial_id: Provenance identifiers.
        metadata: Free-form extra information (stimulus onset, source file, ...).
    """

    eeg: np.ndarray
    fs: float
    channel_names: tuple[str, ...]
    true_frequency: float | None = None
    true_label: str | None = None
    subject_id: str | None = None
    block_id: str | None = None
    trial_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.eeg.ndim != 2:
            raise ValueError(f"eeg must be 2-D (n_channels, n_samples); got shape {self.eeg.shape}")
        if self.eeg.shape[0] != len(self.channel_names):
            raise ValueError(
                f"eeg has {self.eeg.shape[0]} rows but {len(self.channel_names)} channel names"
            )
        if self.fs <= 0:
            raise ValueError(f"fs must be positive; got {self.fs}")

    @property
    def n_channels(self) -> int:
        return self.eeg.shape[0]

    @property
    def n_samples(self) -> int:
        return self.eeg.shape[1]

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs

    def select_channels(self, names: list[str] | tuple[str, ...]) -> Trial:
        """Return a copy containing only ``names``, in that order. Raises on missing names."""
        missing = [n for n in names if n not in self.channel_names]
        if missing:
            raise KeyError(f"channels not present in trial: {missing}")
        idx = [self.channel_names.index(n) for n in names]
        return Trial(
            eeg=self.eeg[idx, :].copy(),
            fs=self.fs,
            channel_names=tuple(names),
            true_frequency=self.true_frequency,
            true_label=self.true_label,
            subject_id=self.subject_id,
            block_id=self.block_id,
            trial_id=self.trial_id,
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class ClassificationResult:
    """Outcome of classifying one analysis window.

    Attributes:
        scores: Largest canonical correlation per class label, in configured class order.
        winner: Label with the highest score (always set, even when rejected).
        winner_frequency: Target frequency of ``winner`` in Hz.
        confidence: The winner's score.
        margin: ``winner score - runner-up score`` (equals ``confidence`` for one class).
        gated: True if at least one threshold was configured and applied.
        accepted: False only if a configured threshold failed. Ungated results are
            accepted by construction — check ``gated`` before treating this as meaningful.
        decision: ``winner`` if accepted, else :data:`REJECT`.
    """

    scores: dict[str, float]
    winner: str
    winner_frequency: float
    confidence: float
    margin: float
    gated: bool
    accepted: bool
    decision: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": dict(self.scores),
            "winner": self.winner,
            "winner_frequency": self.winner_frequency,
            "confidence": self.confidence,
            "margin": self.margin,
            "gated": self.gated,
            "accepted": self.accepted,
            "decision": self.decision,
        }
