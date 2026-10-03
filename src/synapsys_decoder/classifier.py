"""Window classifier: CCA scores per target, argmax, optional threshold gating.

Pipeline for one **already-preprocessed** window (this module never filters):

1. For each target (in mapping order) build the sin/cos harmonic references
   and compute the largest canonical correlation with the EEG window.
2. Winner = target with the highest score. Ties are broken deterministically
   by class order: the **first** label in the mapping wins.
3. ``margin = winner score - runner-up score`` (equals the winner score when
   there is only one class).
4. Gating: each configured threshold must pass —
   ``confidence >= confidence_threshold`` and ``margin >= margin_threshold``.
   Unconfigured (``None``) thresholds are skipped. If no threshold is
   configured the result is ungated (``gated=False``) and accepted by
   construction. A rejected window still reports its winner; only
   ``decision`` becomes :data:`~synapsys_decoder.types.REJECT`.

Standard CCA is not trained; nothing here is fit to data.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from synapsys_decoder.cca import cca_score
from synapsys_decoder.config import DecoderConfig
from synapsys_decoder.references import harmonic_references
from synapsys_decoder.types import REJECT, ClassificationResult


def score_targets(
    eeg_window: np.ndarray,
    fs: float,
    target_frequencies: Mapping[str, float],
    harmonics: int,
) -> dict[str, float]:
    """Largest canonical correlation of ``eeg_window`` with each target's references.

    Args:
        eeg_window: ``(n_channels, n_samples)`` preprocessed EEG.
        fs: Sample rate in Hz.
        target_frequencies: label -> stimulus frequency (Hz); order is class order.
        harmonics: Number of harmonics per reference set.

    Returns:
        ``{label: score}`` in the same order as ``target_frequencies``.

    Raises:
        ValueError: on bad shapes, empty targets, or invalid reference parameters.
        DegenerateSignalError: if CCA is undefined for this window.
    """
    eeg_window = np.asarray(eeg_window, dtype=np.float64)
    if eeg_window.ndim != 2:
        raise ValueError(
            f"eeg_window must be 2-D (n_channels, n_samples); got shape {eeg_window.shape}"
        )
    if not target_frequencies:
        raise ValueError("target_frequencies must be non-empty")
    n_samples = eeg_window.shape[1]
    return {
        label: cca_score(eeg_window, harmonic_references(freq, fs, n_samples, harmonics))
        for label, freq in target_frequencies.items()
    }


def decide(
    scores: Mapping[str, float],
    target_frequencies: Mapping[str, float],
    confidence_threshold: float | None = None,
    margin_threshold: float | None = None,
) -> ClassificationResult:
    """Turn per-class scores into a (possibly gated) decision.

    Args:
        scores: label -> score. Must contain exactly the labels of
            ``target_frequencies``; values must be finite.
        target_frequencies: label -> frequency; its order is the class order
            used for tie-breaking and for the returned ``scores``.
        confidence_threshold: Minimum winner score in ``[0, 1]``, or ``None``.
        margin_threshold: Minimum ``winner - runner-up`` (``>= 0``), or ``None``.

    Returns:
        :class:`ClassificationResult`. Ties go to the earliest label in
        ``target_frequencies``.

    Raises:
        ValueError: on mismatched labels, non-finite scores, or invalid thresholds.
    """
    if not target_frequencies:
        raise ValueError("target_frequencies must be non-empty")
    if set(scores) != set(target_frequencies):
        raise ValueError(
            f"score labels {sorted(scores)} do not match target labels {sorted(target_frequencies)}"
        )
    if confidence_threshold is not None and not (0.0 <= confidence_threshold <= 1.0):
        raise ValueError(f"confidence_threshold must be in [0, 1]; got {confidence_threshold}")
    if margin_threshold is not None and not (margin_threshold >= 0.0):
        raise ValueError(f"margin_threshold must be >= 0; got {margin_threshold}")

    ordered = {label: float(scores[label]) for label in target_frequencies}
    if not all(np.isfinite(v) for v in ordered.values()):
        raise ValueError(f"scores must be finite; got {ordered}")

    winner = None
    best = -np.inf
    for label, value in ordered.items():
        if value > best:  # strict '>' => first label wins ties
            winner, best = label, value
    assert winner is not None

    others = [v for label, v in ordered.items() if label != winner]
    margin = best - max(others) if others else best

    gated = confidence_threshold is not None or margin_threshold is not None
    accepted = True
    if confidence_threshold is not None and best < confidence_threshold:
        accepted = False
    if margin_threshold is not None and margin < margin_threshold:
        accepted = False

    return ClassificationResult(
        scores=ordered,
        winner=winner,
        winner_frequency=float(target_frequencies[winner]),
        confidence=best,
        margin=float(margin),
        gated=gated,
        accepted=accepted,
        decision=winner if accepted else REJECT,
    )


def classify_window(
    eeg: np.ndarray,
    fs: float,
    target_frequencies: Mapping[str, float],
    harmonics: int,
    confidence_threshold: float | None = None,
    margin_threshold: float | None = None,
) -> ClassificationResult:
    """Score and decide one already-preprocessed window.

    This function does **not** filter; apply the preprocessing module first.
    See :func:`score_targets` and :func:`decide` for arguments and errors.
    """
    scores = score_targets(eeg, fs, target_frequencies, harmonics)
    return decide(scores, target_frequencies, confidence_threshold, margin_threshold)


def classify_window_with_config(eeg: np.ndarray, cfg: DecoderConfig) -> ClassificationResult:
    """:func:`classify_window` with fs, targets, harmonics and thresholds from ``cfg``.

    ``eeg`` rows must already be in ``cfg.channels`` order and preprocessed.

    Raises:
        ValueError: if ``eeg`` is not 2-D or ``eeg.shape[0] != len(cfg.channels)``.
    """
    eeg = np.asarray(eeg, dtype=np.float64)
    if eeg.ndim != 2:
        raise ValueError(f"eeg must be 2-D (n_channels, n_samples); got shape {eeg.shape}")
    if eeg.shape[0] != len(cfg.channels):
        raise ValueError(
            f"eeg has {eeg.shape[0]} channels but config lists {len(cfg.channels)}: "
            f"{list(cfg.channels)}"
        )
    return classify_window(
        eeg,
        cfg.sample_rate_hz,
        cfg.targets,
        cfg.harmonics,
        confidence_threshold=cfg.decision.confidence_threshold,
        margin_threshold=cfg.decision.margin_threshold,
    )
