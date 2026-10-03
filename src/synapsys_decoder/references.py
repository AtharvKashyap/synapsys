"""Sinusoidal harmonic reference signals for CCA-based SSVEP decoding.

For a stimulus frequency ``f`` and ``H`` harmonics, the reference set is the
``2H`` signals

    sin(2π·1·f·t), cos(2π·1·f·t), sin(2π·2·f·t), cos(2π·2·f·t), ..., cos(2π·H·f·t)

sampled at ``t = k / fs`` for ``k = 0 .. n_samples-1``. Row order is fixed
(sin before cos, harmonic 1 first) so the table can be precomputed once per
target and stored on the ESP32 as a constant sin/cos table of identical layout.

Nyquist policy: a reference whose highest harmonic ``H·f`` is at or above
``fs/2`` would alias, so it is **refused** (``ValueError``) rather than
silently truncated.

No project frequencies live here; every value is an explicit argument.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _validate(frequency_hz: float, fs: float, n_samples: int, harmonics: int) -> None:
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError(f"frequency_hz must be a finite value > 0; got {frequency_hz}")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be a finite value > 0; got {fs}")
    if int(n_samples) != n_samples or n_samples < 1:
        raise ValueError(f"n_samples must be an integer >= 1; got {n_samples}")
    if int(harmonics) != harmonics or harmonics < 1:
        raise ValueError(f"harmonics must be an integer >= 1; got {harmonics}")
    top = harmonics * frequency_hz
    if top >= fs / 2:
        raise ValueError(
            f"harmonic {harmonics} of {frequency_hz} Hz ({top:.4f} Hz) is at/above "
            f"Nyquist ({fs / 2} Hz); refusing to build an aliased reference"
        )


def harmonic_references(
    frequency_hz: float, fs: float, n_samples: int, harmonics: int
) -> np.ndarray:
    """Build the sin/cos harmonic reference matrix for one stimulus frequency.

    Args:
        frequency_hz: Fundamental stimulus frequency ``f`` in Hz (> 0).
        fs: Sample rate in Hz (> 0).
        n_samples: Number of samples (>= 1); should equal the EEG window length.
        harmonics: Number of harmonics ``H`` (>= 1).

    Returns:
        float64 array of shape ``(2*H, n_samples)``. Row ``2(h-1)`` is
        ``sin(2π h f t)`` and row ``2(h-1)+1`` is ``cos(2π h f t)`` for
        ``h = 1..H``, with ``t = arange(n_samples) / fs`` (so ``t[0] = 0``).

    Raises:
        ValueError: on non-positive ``frequency_hz``/``fs``, ``n_samples < 1``,
            ``harmonics < 1``, or if ``H·f >= fs/2`` (Nyquist policy: refuse).
    """
    _validate(frequency_hz, fs, n_samples, harmonics)
    n_samples = int(n_samples)
    harmonics = int(harmonics)
    t = np.arange(n_samples, dtype=np.float64) / float(fs)
    out = np.empty((2 * harmonics, n_samples), dtype=np.float64)
    for h in range(1, harmonics + 1):
        phase = 2.0 * np.pi * h * float(frequency_hz) * t
        out[2 * (h - 1)] = np.sin(phase)
        out[2 * (h - 1) + 1] = np.cos(phase)
    return out


def reference_bank(
    frequencies: Sequence[float], fs: float, n_samples: int, harmonics: int
) -> np.ndarray:
    """Stack :func:`harmonic_references` for several targets.

    This is the precomputable lookup table mirrored by the ESP32 firmware
    (one ``2H x n_samples`` sin/cos table per target, in class order).

    Args:
        frequencies: Target frequencies in Hz, in class order (non-empty).
        fs: Sample rate in Hz.
        n_samples: Window length in samples.
        harmonics: Number of harmonics ``H``.

    Returns:
        float64 array of shape ``(n_targets, 2*H, n_samples)``.

    Raises:
        ValueError: if ``frequencies`` is empty or any entry is invalid (see
            :func:`harmonic_references`).
    """
    freqs = list(frequencies)
    if not freqs:
        raise ValueError("frequencies must be non-empty")
    return np.stack([harmonic_references(f, fs, n_samples, harmonics) for f in freqs])
