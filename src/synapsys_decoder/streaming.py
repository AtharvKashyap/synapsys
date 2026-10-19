"""Decode a continuous recording on the device's decision schedule (offline).

The ESP32 filters the stream causally from its first sample and evaluates a window
ending at every ``t = window_samples + k * hop_samples`` (``t`` = samples since stream
start; the window is ``[t - window_samples, t)``). :func:`decode_stream` reproduces that
schedule with the frozen reference functions, so its scores are what the device should
print for the same samples, and gating can be re-applied for any thresholds without
recomputing CCA (:meth:`StreamScores.decisions`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from synapsys_decoder.cca import DegenerateSignalError
from synapsys_decoder.classifier import score_targets
from synapsys_decoder.config import DecoderConfig
from synapsys_decoder.preprocessing import preprocess
from synapsys_decoder.types import REJECT, Trial


@dataclass(frozen=True)
class StreamScores:
    """Per-decision-point scores of one stream (gating not yet applied)."""

    labels: tuple[str, ...]  # class order
    window_end: np.ndarray  # (n,) int, samples since stream start (device "t")
    scores: np.ndarray  # (n, n_labels); NaN rows where CCA was degenerate
    fs: float

    @property
    def degenerate(self) -> np.ndarray:
        return np.isnan(self.scores).any(axis=1)

    @property
    def winner_index(self) -> np.ndarray:
        """argmax per row (first label wins ties, as ``classifier.decide``); -1 if degenerate."""
        idx = np.argmax(np.where(np.isnan(self.scores), -np.inf, self.scores), axis=1)
        return np.where(self.degenerate, -1, idx)

    @property
    def confidence(self) -> np.ndarray:
        return np.where(self.degenerate, np.nan, np.nanmax(self._filled(), axis=1))

    @property
    def margin(self) -> np.ndarray:
        """Winner minus runner-up (single class: the winner score), as ``classifier.decide``."""
        s = self._filled()
        if s.shape[1] == 1:
            m = s[:, 0]
        else:
            top2 = np.sort(s, axis=1)[:, -2:]
            with np.errstate(invalid="ignore"):  # -inf - -inf on degenerate rows
                m = top2[:, 1] - top2[:, 0]
        return np.where(self.degenerate, np.nan, m)

    def _filled(self) -> np.ndarray:
        return np.where(np.isnan(self.scores), -np.inf, self.scores)

    def accepted(self, confidence_threshold: float | None, margin_threshold: float | None):
        """Boolean mask: winner passes the configured gates (degenerate -> False)."""
        ok = ~self.degenerate
        with np.errstate(invalid="ignore"):
            if confidence_threshold is not None:
                ok &= self.confidence >= confidence_threshold
            if margin_threshold is not None:
                ok &= self.margin >= margin_threshold
        return ok

    def decisions(
        self, confidence_threshold: float | None = None, margin_threshold: float | None = None
    ) -> list[str]:
        """Decision label (or ``REJECT``) per point for the given thresholds."""
        acc = self.accepted(confidence_threshold, margin_threshold)
        win = self.winner_index
        return [self.labels[w] if a else REJECT for w, a in zip(win, acc, strict=True)]


def decision_ends(n_samples: int, window_samples: int, hop_samples: int) -> np.ndarray:
    """Device decision points ``t`` that fit in ``n_samples`` (may be empty)."""
    if window_samples < 1 or hop_samples < 1:
        raise ValueError("window_samples and hop_samples must be >= 1")
    if n_samples < window_samples:
        return np.zeros(0, dtype=np.int64)
    return np.arange(window_samples, n_samples + 1, hop_samples, dtype=np.int64)


def decode_stream(trial: Trial, cfg: DecoderConfig, *, hop_samples: int = 125) -> StreamScores:
    """Filter the whole recording causally (cold start at its first sample, like the device
    after a stream start), then score every scheduled window.

    Raises:
        ValueError: fs mismatch (never resampled) or ``cfg.preprocessing.mode`` not causal
            (the device is causal; zero-phase would score windows with future samples).
        KeyError: a configured channel is missing.
    """
    if not np.isclose(trial.fs, cfg.sample_rate_hz):
        raise ValueError(f"trial fs {trial.fs} != config sample_rate_hz {cfg.sample_rate_hz}")
    if cfg.preprocessing.mode != "causal":
        raise ValueError("decode_stream mirrors the device: preprocessing.mode must be 'causal'")
    eeg = trial.select_channels(cfg.channels).eeg
    filtered = preprocess(eeg, trial.fs, cfg.preprocessing)
    ends = decision_ends(filtered.shape[1], cfg.window_samples, hop_samples)
    scores = np.full((len(ends), len(cfg.labels)), np.nan)
    for i, t in enumerate(ends):
        win = filtered[:, t - cfg.window_samples : t]
        try:
            sc = score_targets(win, trial.fs, cfg.targets, cfg.harmonics)
        except DegenerateSignalError:
            continue
        scores[i] = [sc[lab] for lab in cfg.labels]
    return StreamScores(labels=cfg.labels, window_end=ends, scores=scores, fs=float(trial.fs))
