"""Synthetic EEG generators for verifying pipeline behaviour.

IMPORTANT: this module exists to check that the decoding pipeline behaves
correctly (shapes, units, frequency selectivity, rejection mechanics, SNR
trends). It does NOT model real human EEG and results obtained on synthetic
data must never be quoted as human performance or used to choose thresholds.

Conventions: arrays are ``(n_channels, n_samples)`` float64 in volts; ``fs`` is
always explicit; all randomness comes from a caller-supplied
``numpy.random.Generator``.

SNR definition (used by :func:`noise_std_for_snr` and ``snr_db`` arguments)
---------------------------------------------------------------------------
``SNR_dB = 10 * log10(P_signal / P_noise)`` where ``P_signal`` is the mean of
``x**2`` over all channels and samples of the noise-free SSVEP component (the
mean of per-channel powers) and ``P_noise`` is the variance of the *white
Gaussian* noise (``std_v**2``). This is a BROADBAND ratio: the noise power is
spread over 0..fs/2, so the noise inside the narrow band around the target
(and a CCA reference) is far smaller than ``P_noise``. For fs=250 Hz and a 1 Hz
analysis resolution, narrowband SNR is roughly broadband SNR + 10*log10(125)
~= +21 dB. A broadband -10 dB is therefore a very different (much easier)
situation than a narrowband -10 dB. Drift and line noise are additional
components and are NOT counted in the SNR.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from synapsys_decoder.types import Trial


def _check_basic(fs: float, n_samples: int | None = None, n_channels: int | None = None) -> None:
    if not fs > 0:
        raise ValueError(f"fs must be positive; got {fs}")
    if n_samples is not None and n_samples <= 0:
        raise ValueError(f"n_samples must be positive; got {n_samples}")
    if n_channels is not None and n_channels <= 0:
        raise ValueError(f"n_channels must be positive; got {n_channels}")


def _time(n_samples: int, fs: float) -> np.ndarray:
    return np.arange(n_samples) / fs


def ssvep_component(
    frequency_hz: float,
    fs: float,
    n_samples: int,
    n_channels: int,
    harmonics: int,
    rng: np.random.Generator,
    *,
    amplitude_v: float = 1e-6,
    harmonic_decay: float = 0.5,
) -> np.ndarray:
    """Noise-free SSVEP-like signal, a different mixture per channel.

    Each channel is ``sum_h amplitude_v * decay**(h-1) * a_ch,h * sin(2 pi h f t + phi_ch,h)``
    with ``a`` uniform in [0.5, 1.5] and ``phi`` uniform in [0, 2 pi), drawn
    per channel and harmonic from ``rng``.

    Raises:
        ValueError: on non-positive fs/f/sizes/harmonics, or if ``harmonics * f >= fs/2``.
    """
    _check_basic(fs, n_samples, n_channels)
    if frequency_hz <= 0:
        raise ValueError(f"frequency_hz must be positive; got {frequency_hz}")
    if harmonics < 1:
        raise ValueError(f"harmonics must be >= 1; got {harmonics}")
    if harmonics * frequency_hz >= fs / 2:
        raise ValueError(
            f"harmonic {harmonics} of {frequency_hz} Hz is at/above Nyquist ({fs / 2} Hz)"
        )
    t = _time(n_samples, fs)
    out = np.zeros((n_channels, n_samples), dtype=np.float64)
    for h in range(1, harmonics + 1):
        amps = rng.uniform(0.5, 1.5, size=(n_channels, 1))
        phases = rng.uniform(0.0, 2 * np.pi, size=(n_channels, 1))
        out += (
            amplitude_v
            * harmonic_decay ** (h - 1)
            * amps
            * np.sin(2 * np.pi * h * frequency_hz * t[None, :] + phases)
        )
    return out


def gaussian_noise(shape: tuple[int, ...], rng: np.random.Generator, std_v: float) -> np.ndarray:
    """White Gaussian noise of standard deviation ``std_v`` volts."""
    if std_v < 0:
        raise ValueError(f"std_v must be >= 0; got {std_v}")
    return rng.normal(0.0, std_v, size=shape).astype(np.float64)


def _sinusoid(
    n_channels: int,
    n_samples: int,
    fs: float,
    rng: np.random.Generator,
    freq_hz: float,
    amplitude_v: float,
) -> np.ndarray:
    _check_basic(fs, n_samples, n_channels)
    if freq_hz <= 0 or freq_hz >= fs / 2:
        raise ValueError(f"freq_hz must be in (0, fs/2); got {freq_hz}")
    phases = rng.uniform(0.0, 2 * np.pi, size=(n_channels, 1))
    return amplitude_v * np.sin(2 * np.pi * freq_hz * _time(n_samples, fs)[None, :] + phases)


def drift(
    n_channels: int,
    n_samples: int,
    fs: float,
    rng: np.random.Generator,
    freq_hz: float = 0.2,
    amplitude_v: float = 20e-6,
) -> np.ndarray:
    """Slow sinusoidal baseline drift with random phase per channel."""
    return _sinusoid(n_channels, n_samples, fs, rng, freq_hz, amplitude_v)


def line_noise(
    n_channels: int,
    n_samples: int,
    fs: float,
    rng: np.random.Generator,
    freq_hz: float = 60.0,
    amplitude_v: float = 5e-6,
) -> np.ndarray:
    """Mains interference sinusoid with random phase per channel."""
    return _sinusoid(n_channels, n_samples, fs, rng, freq_hz, amplitude_v)


def scale_channels(
    eeg: np.ndarray, rng: np.random.Generator, low: float = 0.5, high: float = 1.5
) -> np.ndarray:
    """Multiply each channel by an independent gain drawn uniformly from [low, high]."""
    if not 0 < low <= high:
        raise ValueError(f"need 0 < low <= high; got {low}, {high}")
    gains = rng.uniform(low, high, size=(eeg.shape[0], 1))
    return eeg * gains


def noise_std_for_snr(signal: np.ndarray, snr_db: float) -> float:
    """White-noise standard deviation giving the requested broadband SNR.

    ``SNR_dB = 10 log10(P_signal / std**2)`` with ``P_signal = mean(signal**2)``
    over all channels and samples (= mean of per-channel powers). Broadband, not
    narrowband: see the module docstring. Drift and line noise are extra and not
    counted.
    """
    p_signal = float(np.mean(np.asarray(signal, dtype=np.float64) ** 2))
    if p_signal <= 0:
        raise ValueError("signal has zero power; SNR is undefined")
    return float(np.sqrt(p_signal / 10.0 ** (snr_db / 10.0)))


def _assemble(
    signal: np.ndarray,
    fs: float,
    rng: np.random.Generator,
    snr_db: float | None,
    drift_amplitude_v: float,
    line_amplitude_v: float,
    line_freq_hz: float,
    channel_scaling: bool,
) -> np.ndarray:
    n_ch, n = signal.shape
    eeg = signal.copy()
    if snr_db is not None:
        eeg += gaussian_noise(eeg.shape, rng, noise_std_for_snr(signal, snr_db))
    if channel_scaling:
        eeg = scale_channels(eeg, rng)
    if drift_amplitude_v > 0:
        eeg += drift(n_ch, n, fs, rng, amplitude_v=drift_amplitude_v)
    if line_amplitude_v > 0:
        eeg += line_noise(n_ch, n, fs, rng, freq_hz=line_freq_hz, amplitude_v=line_amplitude_v)
    return eeg


def _n_samples(fs: float, duration_s: float) -> int:
    _check_basic(fs)
    if not duration_s > 0:
        raise ValueError(f"duration_s must be positive; got {duration_s}")
    n = round(duration_s * fs)
    if n < 1:
        raise ValueError("duration_s * fs yields no samples")
    return n


def make_synthetic_trial(
    frequency_hz: float,
    fs: float,
    duration_s: float,
    channel_names: Sequence[str],
    rng: np.random.Generator,
    *,
    snr_db: float | None = None,
    harmonics: int = 2,
    label: str | None = None,
    amplitude_v: float = 1e-6,
    drift_amplitude_v: float = 0.0,
    line_amplitude_v: float = 0.0,
    line_freq_hz: float = 60.0,
    channel_scaling: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Trial:
    """Build a synthetic SSVEP trial.

    ``snr_db=None`` gives a clean (noise-free) signal. Otherwise white noise is
    added at the requested broadband SNR (see module docstring). Optional
    channel gains are applied to signal+white noise, then drift and line noise
    are added on top. ``true_frequency``/``true_label`` are set; ``metadata``
    records every generation parameter (not the seed/rng) plus ``metadata`` extras.
    """
    n = _n_samples(fs, duration_s)
    names = tuple(channel_names)
    sig = ssvep_component(frequency_hz, fs, n, len(names), harmonics, rng, amplitude_v=amplitude_v)
    eeg = _assemble(
        sig, fs, rng, snr_db, drift_amplitude_v, line_amplitude_v, line_freq_hz, channel_scaling
    )
    meta: dict[str, Any] = {
        "kind": "synthetic_ssvep",
        "synthetic": True,
        "frequency_hz": float(frequency_hz),
        "fs": float(fs),
        "duration_s": float(duration_s),
        "snr_db": snr_db,
        "harmonics": harmonics,
        "amplitude_v": amplitude_v,
        "drift_amplitude_v": drift_amplitude_v,
        "line_amplitude_v": line_amplitude_v,
        "line_freq_hz": line_freq_hz,
        "channel_scaling": channel_scaling,
    }
    meta.update(metadata or {})
    return Trial(
        eeg=eeg,
        fs=float(fs),
        channel_names=names,
        true_frequency=float(frequency_hz),
        true_label=label,
        metadata=meta,
    )


def make_noise_trial(
    fs: float,
    duration_s: float,
    channel_names: Sequence[str],
    rng: np.random.Generator,
    *,
    std_v: float = 1e-6,
    drift_amplitude_v: float = 0.0,
    line_amplitude_v: float = 0.0,
    line_freq_hz: float = 60.0,
    channel_scaling: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Trial:
    """Pure-noise trial (no SSVEP): ``true_label=None``, ``true_frequency=None``."""
    n = _n_samples(fs, duration_s)
    names = tuple(channel_names)
    eeg = gaussian_noise((len(names), n), rng, std_v)
    if channel_scaling:
        eeg = scale_channels(eeg, rng)
    if drift_amplitude_v > 0:
        eeg += drift(len(names), n, fs, rng, amplitude_v=drift_amplitude_v)
    if line_amplitude_v > 0:
        eeg += line_noise(
            len(names), n, fs, rng, freq_hz=line_freq_hz, amplitude_v=line_amplitude_v
        )
    meta: dict[str, Any] = {
        "kind": "noise",
        "synthetic": True,
        "fs": float(fs),
        "duration_s": float(duration_s),
        "std_v": std_v,
        "drift_amplitude_v": drift_amplitude_v,
        "line_amplitude_v": line_amplitude_v,
        "line_freq_hz": line_freq_hz,
        "channel_scaling": channel_scaling,
    }
    meta.update(metadata or {})
    return Trial(eeg=eeg, fs=float(fs), channel_names=names, metadata=meta)


def make_off_target_trial(
    frequency_hz: float,
    fs: float,
    duration_s: float,
    channel_names: Sequence[str],
    rng: np.random.Generator,
    **kwargs: Any,
) -> Trial:
    """SSVEP at a frequency that is not a command target; ``true_label=None``.

    Thin wrapper over :func:`make_synthetic_trial` (same keyword arguments except
    ``label``). ``true_frequency`` records the presented frequency;
    metadata ``kind="off_target"``.
    """
    if kwargs.get("label") is not None:
        raise ValueError("off-target trials have no label")
    kwargs.pop("label", None)
    extra = dict(kwargs.pop("metadata", None) or {})
    extra["kind"] = "off_target"
    return make_synthetic_trial(
        frequency_hz, fs, duration_s, channel_names, rng, label=None, metadata=extra, **kwargs
    )


def make_mixture_trial(
    f1_hz: float,
    f2_hz: float,
    weight: float,
    fs: float,
    duration_s: float,
    channel_names: Sequence[str],
    rng: np.random.Generator,
    *,
    snr_db: float | None = None,
    harmonics: int = 2,
    amplitude_v: float = 1e-6,
    drift_amplitude_v: float = 0.0,
    line_amplitude_v: float = 0.0,
    line_freq_hz: float = 60.0,
    channel_scaling: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Trial:
    """Ambiguous trial: ``(1 - weight) * SSVEP(f1) + weight * SSVEP(f2)``.

    ``weight`` in [0, 1] is the amplitude fraction of ``f2``. SNR is computed
    on the mixed signal. ``true_label=None``; ``true_frequency=None``.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be in [0, 1]; got {weight}")
    n = _n_samples(fs, duration_s)
    names = tuple(channel_names)
    s1 = ssvep_component(f1_hz, fs, n, len(names), harmonics, rng, amplitude_v=amplitude_v)
    s2 = ssvep_component(f2_hz, fs, n, len(names), harmonics, rng, amplitude_v=amplitude_v)
    sig = (1.0 - weight) * s1 + weight * s2
    if not np.any(sig):
        raise ValueError("mixture has zero power")
    eeg = _assemble(
        sig, fs, rng, snr_db, drift_amplitude_v, line_amplitude_v, line_freq_hz, channel_scaling
    )
    meta: dict[str, Any] = {
        "kind": "mixture",
        "synthetic": True,
        "f1_hz": float(f1_hz),
        "f2_hz": float(f2_hz),
        "weight": float(weight),
        "fs": float(fs),
        "duration_s": float(duration_s),
        "snr_db": snr_db,
        "harmonics": harmonics,
        "amplitude_v": amplitude_v,
        "drift_amplitude_v": drift_amplitude_v,
        "line_amplitude_v": line_amplitude_v,
        "line_freq_hz": line_freq_hz,
        "channel_scaling": channel_scaling,
    }
    meta.update(metadata or {})
    return Trial(eeg=eeg, fs=float(fs), channel_names=names, metadata=meta)
