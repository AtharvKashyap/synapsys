"""End-to-end decode of one trial: channel selection -> preprocessing -> window -> CCA -> decision.

This is the logical ESP32 pipeline expressed offline. Ordering matters for the
causal path: the whole epoch is filtered first (as the device filters its
continuous stream) and the analysis window is cropped **afterwards**, so the
window sees a filter with history rather than a cold start.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from synapsys_decoder.classifier import classify_window_with_config
from synapsys_decoder.config import DecoderConfig
from synapsys_decoder.preprocessing import extract_window, preprocess
from synapsys_decoder.types import ClassificationResult, Trial


@dataclass(frozen=True)
class DecodedTrial:
    """Classification of one trial plus the intermediate arrays (for plots / fixtures)."""

    result: ClassificationResult
    filtered: np.ndarray
    window: np.ndarray
    window_start_s: float


def decode_trial(
    trial: Trial, cfg: DecoderConfig, *, window_start_s: float | None = None
) -> DecodedTrial:
    """Decode a single trial with ``cfg``.

    Args:
        trial: Epoch in volts. Must contain every channel in ``cfg.channels``
            (selected by name, in config order) and have ``fs == cfg.sample_rate_hz``.
        cfg: Decoder configuration (window length, targets, preprocessing, gating).
        window_start_s: Window start relative to the epoch start. ``None`` takes the
            **last** ``cfg.window_seconds`` of the epoch (the real-time convention).

    Raises:
        ValueError: on sample-rate mismatch (loaders never resample, so neither do we).
        KeyError: if a configured channel is missing.
    """
    if not np.isclose(trial.fs, cfg.sample_rate_hz):
        raise ValueError(f"trial fs {trial.fs} != config sample_rate_hz {cfg.sample_rate_hz}")
    selected = trial.select_channels(cfg.channels)
    filtered = preprocess(selected.eeg, trial.fs, cfg.preprocessing)
    if window_start_s is None:
        window_start_s = (filtered.shape[1] - cfg.window_samples) / trial.fs
    window = extract_window(filtered, trial.fs, window_start_s, cfg.window_seconds)
    result = classify_window_with_config(window, cfg)
    return DecodedTrial(
        result=result, filtered=filtered, window=window, window_start_s=window_start_s
    )
