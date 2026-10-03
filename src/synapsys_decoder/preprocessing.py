"""DSP preprocessing: DC removal, Butterworth bandpass, mains notch, windowing.

Logical pipeline (each stage can be disabled through
:class:`~synapsys_decoder.config.PreprocessingConfig`)::

    raw -> remove DC -> Butterworth bandpass (SOS) -> IIR notch -> window -> CCA

Portability (ESP32-S3 port)
---------------------------
The device cannot see future samples and runs the filters **continuously** on
the sample stream with per-channel state, then takes the most recent 2-3 s
window. The portable path is therefore ``mode="causal"``: the whole cascade is
one array of second-order sections (biquads) applied with ``sosfilt`` (direct
form II transposed), which is exactly what the C port implements.
:class:`StreamingFilter` is the reference for that device loop and is tested to
equal :func:`preprocess` (causal) on the concatenated stream.

``mode="zero_phase"`` (``sosfiltfilt``) runs the cascade forwards and backwards.
It needs the whole epoch (future samples), is **offline-only and not portable**,
and exists only for comparison/ablation.

Filter design and coefficient layout
------------------------------------
* ``bandpass_order`` is the **Butterworth prototype order N** passed to
  ``scipy.signal.butter(N, [lo, hi], "bandpass")``. A bandpass built that way
  has **2N poles = N biquad sections** (the default N=4 gives an 8th-order
  bandpass, 4 biquads: 4th-order 0.5 Hz high-pass skirt + 4th-order 45 Hz
  low-pass skirt). A lone high-pass or low-pass edge of order N has N poles
  (ceil(N/2) biquads).
* The notch is ``scipy.signal.iirnotch(notch_hz, Q)``: one biquad, -3 dB
  bandwidth = notch_hz / Q (2 Hz for 60 Hz, Q=30).
* :func:`build_filter_sos` returns an ``(n_sections, 6)`` array, rows ordered
  bandpass sections first (scipy's order) then the notch. Each row is
  ``[b0, b1, b2, a0, a1, a2]`` with ``a0 == 1``; section k's output feeds
  section k+1. Each section has two state variables per channel.

Design response (fs=250 Hz, defaults: N=4 Butterworth 0.5-45 Hz, 60 Hz notch Q=30)
---------------------------------------------------------------------------------
Measured with :func:`frequency_response_db` (magnitude, dB; rounded to 0.01 dB;
tests assert these with tolerances)::

    freq (Hz)   bandpass only   bandpass + 60 Hz notch   note
    0.2           -32.13           -32.13                drift, attenuated
    0.5            -3.01            -3.01                low -3 dB edge
    8.571          -0.00            -0.00                YES target
    12             -0.00            -0.00                NO target
    15             -0.00            -0.00                HELP target
    17.14          -0.00            -0.00                YES 2nd harmonic
    24             -0.01            -0.01                NO 2nd harmonic
    30             -0.09            -0.09                HELP 2nd harmonic
    45             -3.01            -3.03                high -3 dB edge
    50             -6.03            -6.07
    60            -13.98         < -100 (-292 numeric)   mains: notch adds the depth
    80            -31.89           -31.90

So every SSVEP fundamental and 2nd harmonic is passed with < 0.1 dB loss.
The 45 Hz low-pass alone only gives ~14 dB at 60 Hz; the notch makes 60 Hz
effectively vanish (it is -3 dB at 59/61 Hz, -0.97 dB at 58/62 Hz, -0.17 dB
at 55/65 Hz, i.e. it does not touch the SSVEP band).

Group delay (causal cascade, :func:`group_delay_s`): 11.5 ms at 8.571 Hz,
10.3 ms at 12 Hz, 10.0 ms at 15 Hz (< 3 samples; nearly equal across targets,
so it only shifts the effective stimulus latency by ~10 ms).

Settling of the causal filter
-----------------------------
The 0.5 Hz high-pass dominates. :func:`settling_time_s` (impulse-response
energy tail) gives 0.23 s for tol=1e-2, 1.05 s for tol=1e-3, 2.10 s for
tol=1e-4. A residual DC step (initial-offset mismatch) is slower: the step
response of the bandpass stays above 10 % of the step for 1.57 s and above
1 % for 3.71 s. Consequence: when an epoch is filtered causally from its first
sample, the first ~1-2 s carry a low-frequency start-up transient. The Wang
benchmark window starts only 0.64 s into the epoch (0.5 s pre-stimulus +
0.14 s latency), so it overlaps that transient; the transient is mostly
< 1 Hz (outside the SSVEP band) but is not zero. On the real-time device the
filter runs continuously, so this applies only to the first few seconds after
boot/reset (discard ~2-4 s of output after a reset).

DC policy (``remove_dc``)
-------------------------
* ``causal``: subtract each channel's **first sample** ``x[:, 0]`` (the only
  baseline estimate available at t=0) from the whole stream, then filter from
  zero state. For a cascade containing a high-pass this is mathematically
  identical to initialising the filter state with ``sosfilt_zi(sos) * x[:, 0]``
  (steady state for a constant input equal to the first sample). Any residual
  DC is then removed by the 0.5 Hz high-pass. With the high-pass disabled the
  residual (true baseline minus first sample) stays in the output. This uses
  only the past, so it is portable (one constant per channel).
* ``zero_phase``: subtract each channel's **epoch mean** (non-causal), then
  ``sosfiltfilt``. Offline-only.
* ``remove_dc=False``: no subtraction; causal filtering starts from zero state
  (so a large electrode offset produces a large start-up transient).
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from synapsys_decoder.config import PreprocessingConfig

__all__ = [
    "StreamingFilter",
    "build_filter_sos",
    "design_bandpass",
    "design_notch",
    "extract_window",
    "frequency_response_db",
    "group_delay_s",
    "last_window",
    "preprocess",
    "settling_time_s",
]


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _check_fs(fs: float) -> float:
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be a positive finite number; got {fs}")
    return fs


def _check_eeg(eeg: np.ndarray) -> np.ndarray:
    """Return ``eeg`` as a float64 2-D array, raising on bad shape or non-finite values."""
    x = np.asarray(eeg, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"eeg must be 2-D (n_channels, n_samples); got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("eeg contains NaN or Inf")
    return x


def _check_sos(sos: np.ndarray) -> np.ndarray:
    sos = np.asarray(sos, dtype=np.float64)
    if sos.ndim != 2 or sos.shape[1] != 6 or sos.shape[0] < 1:
        raise ValueError(f"sos must have shape (n_sections, 6); got {sos.shape}")
    return sos


# --------------------------------------------------------------------------- #
# Filter design
# --------------------------------------------------------------------------- #
def design_bandpass(
    fs: float, low_hz: float | None, high_hz: float | None, order: int
) -> np.ndarray | None:
    """Design a Butterworth band/high/low-pass filter as second-order sections.

    Args:
        fs: Sample rate in Hz.
        low_hz: High-pass (lower) -3 dB edge in Hz, or ``None`` to omit it.
        high_hz: Low-pass (upper) -3 dB edge in Hz, or ``None`` to omit it.
        order: Butterworth prototype order N. With both edges this yields a
            bandpass with 2N poles (N biquads); with one edge, N poles.

    Returns:
        ``(n_sections, 6)`` SOS array, or ``None`` if both edges are ``None``.

    Raises:
        ValueError: on non-positive edges, ``low_hz >= high_hz``, an edge at or
            above Nyquist, or ``order < 1``.
    """
    fs = _check_fs(fs)
    if order < 1:
        raise ValueError(f"order must be >= 1; got {order}")
    nyq = fs / 2
    for name, edge in (("low_hz", low_hz), ("high_hz", high_hz)):
        if edge is not None and not (0 < edge < nyq):
            raise ValueError(f"{name}={edge} must be in (0, Nyquist={nyq}) Hz")
    if low_hz is not None and high_hz is not None:
        if low_hz >= high_hz:
            raise ValueError(f"low_hz ({low_hz}) must be < high_hz ({high_hz})")
        return signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    if low_hz is not None:
        return signal.butter(order, low_hz, btype="highpass", fs=fs, output="sos")
    if high_hz is not None:
        return signal.butter(order, high_hz, btype="lowpass", fs=fs, output="sos")
    return None


def design_notch(fs: float, notch_hz: float, q: float) -> np.ndarray:
    """Design a second-order IIR notch (``scipy.signal.iirnotch``) as one SOS row.

    Args:
        fs: Sample rate in Hz.
        notch_hz: Notch centre frequency in Hz (must be in (0, Nyquist)).
        q: Quality factor; -3 dB bandwidth = ``notch_hz / q``.

    Returns:
        ``(1, 6)`` SOS array.
    """
    fs = _check_fs(fs)
    nyq = fs / 2
    if not (0 < notch_hz < nyq):
        raise ValueError(f"notch_hz={notch_hz} must be in (0, Nyquist={nyq}) Hz")
    if q <= 0:
        raise ValueError(f"q must be > 0; got {q}")
    b, a = signal.iirnotch(notch_hz, q, fs=fs)
    return signal.tf2sos(b, a)


def build_filter_sos(fs: float, cfg: PreprocessingConfig) -> np.ndarray | None:
    """Cascade all enabled filter stages into one SOS array.

    Row order: Butterworth sections (scipy order), then the notch section. Each
    row is ``[b0, b1, b2, a0=1, a1, a2]``; this single biquad cascade is exactly
    what the ESP32 port implements. Returns ``None`` if no filter stage is
    enabled. ``cfg.remove_dc`` and ``cfg.mode`` are not part of the SOS.
    """
    stages = []
    bp = design_bandpass(fs, cfg.bandpass_low_hz, cfg.bandpass_high_hz, cfg.bandpass_order)
    if bp is not None:
        stages.append(bp)
    if cfg.notch_hz is not None:
        stages.append(design_notch(fs, cfg.notch_hz, cfg.notch_q))
    if not stages:
        return None
    return np.vstack(stages).astype(np.float64)


# --------------------------------------------------------------------------- #
# Offline application
# --------------------------------------------------------------------------- #
def preprocess(eeg: np.ndarray, fs: float, cfg: PreprocessingConfig) -> np.ndarray:
    """Apply DC removal and the configured filter cascade to a whole epoch.

    Args:
        eeg: ``(n_channels, n_samples)`` array in volts.
        fs: Sample rate in Hz.
        cfg: Preprocessing configuration.

    Returns:
        New float64 array with the same shape as ``eeg``.

    ``cfg.mode == "causal"`` uses ``sosfilt`` and only past samples: output
    sample ``t`` depends only on ``eeg[:, :t+1]`` (DC policy: subtract the first
    sample; see module docstring). Equals :class:`StreamingFilter` on the same
    stream. ``cfg.mode == "zero_phase"`` subtracts the epoch mean and uses
    ``sosfiltfilt`` — **offline only, not portable**; it raises if the epoch is
    shorter than scipy's padding length.

    Raises:
        ValueError: on wrong ndim, NaN/Inf, or invalid filter parameters.
    """
    x = _check_eeg(eeg).copy()
    fs = _check_fs(fs)
    sos = build_filter_sos(fs, cfg)
    if x.shape[1] == 0:
        return x
    if cfg.mode == "causal":
        if cfg.remove_dc:
            x -= x[:, :1]
        if sos is None:
            return x
        return signal.sosfilt(sos, x, axis=-1)
    if cfg.mode == "zero_phase":
        if cfg.remove_dc:
            x -= x.mean(axis=1, keepdims=True)
        if sos is None:
            return x
        return signal.sosfiltfilt(sos, x, axis=-1)
    raise ValueError(f"unknown mode {cfg.mode!r}")


# --------------------------------------------------------------------------- #
# Streaming (device reference)
# --------------------------------------------------------------------------- #
class StreamingFilter:
    """Causal per-channel SOS filter for chunked, continuous streams.

    Reference implementation of the ESP32 loop. Feeding a stream in chunks of
    any size gives the same output as :func:`preprocess` with ``mode="causal"``
    on the concatenated stream (same DC policy: if ``remove_dc`` the first
    sample seen after construction/:meth:`reset` is stored per channel and
    subtracted from every sample; filter state starts at zero).

    Args:
        sos: ``(n_sections, 6)`` SOS array, or ``None`` for no filtering.
        n_channels: Number of channels (rows) per chunk.
        remove_dc: Subtract each channel's first sample (see module docstring).
    """

    def __init__(self, sos: np.ndarray | None, n_channels: int, remove_dc: bool = True) -> None:
        if n_channels < 1:
            raise ValueError(f"n_channels must be >= 1; got {n_channels}")
        self.sos = None if sos is None else _check_sos(sos).copy()
        self.n_channels = int(n_channels)
        self.remove_dc = bool(remove_dc)
        self._zi: np.ndarray | None = None
        self._offset: np.ndarray | None = None
        self.reset()

    @classmethod
    def from_config(cls, fs: float, cfg: PreprocessingConfig, n_channels: int) -> StreamingFilter:
        """Build from a config. Raises if ``cfg.mode`` is not ``"causal"``."""
        if cfg.mode != "causal":
            raise ValueError("StreamingFilter is causal; cfg.mode must be 'causal'")
        return cls(build_filter_sos(fs, cfg), n_channels, remove_dc=cfg.remove_dc)

    def reset(self) -> None:
        """Clear filter state and the stored DC offset (next sample re-initialises)."""
        if self.sos is not None:
            self._zi = np.zeros((self.sos.shape[0], self.n_channels, 2))
        self._offset = None

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Filter one ``(n_channels, k)`` chunk (``k >= 0``) and advance the state."""
        x = _check_eeg(chunk).copy()
        if x.shape[0] != self.n_channels:
            raise ValueError(f"chunk has {x.shape[0]} channels; expected {self.n_channels}")
        if x.shape[1] == 0:
            return x
        if self.remove_dc:
            if self._offset is None:
                self._offset = x[:, :1].copy()
            x -= self._offset
        if self.sos is None:
            return x
        y, self._zi = signal.sosfilt(self.sos, x, axis=-1, zi=self._zi)
        return y


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def _n_samples(seconds: float, fs: float) -> int:
    # Python round(): nearest integer, ties to even. Matches DecoderConfig.window_samples.
    return round(seconds * fs)


def extract_window(eeg: np.ndarray, fs: float, start_s: float, duration_s: float) -> np.ndarray:
    """Return a copy of ``eeg[:, i0 : i0 + n]``.

    ``i0 = round(start_s * fs)`` and ``n = round(duration_s * fs)`` (Python
    ``round``: nearest integer, ties to even — same rule as
    ``DecoderConfig.window_samples``). E.g. fs=250, start 0.64 s, 2.5 s ->
    samples 160..784 (625 samples).

    Raises:
        ValueError: if ``n < 1``, ``i0 < 0`` or the window runs past the end.
    """
    x = np.asarray(eeg)
    if x.ndim != 2:
        raise ValueError(f"eeg must be 2-D (n_channels, n_samples); got shape {x.shape}")
    fs = _check_fs(fs)
    i0 = _n_samples(start_s, fs)
    n = _n_samples(duration_s, fs)
    if n < 1:
        raise ValueError(f"duration_s={duration_s} gives {n} samples at fs={fs}")
    if i0 < 0 or i0 + n > x.shape[1]:
        raise ValueError(
            f"window [{i0}, {i0 + n}) samples is outside the epoch of {x.shape[1]} samples"
        )
    return x[:, i0 : i0 + n].copy()


def last_window(eeg: np.ndarray, fs: float, duration_s: float) -> np.ndarray:
    """Return a copy of the most recent ``round(duration_s * fs)`` samples (real-time view).

    Raises:
        ValueError: if fewer samples are available than requested or ``n < 1``.
    """
    x = np.asarray(eeg)
    if x.ndim != 2:
        raise ValueError(f"eeg must be 2-D (n_channels, n_samples); got shape {x.shape}")
    fs = _check_fs(fs)
    n = _n_samples(duration_s, fs)
    if n < 1:
        raise ValueError(f"duration_s={duration_s} gives {n} samples at fs={fs}")
    if n > x.shape[1]:
        raise ValueError(f"need {n} samples but only {x.shape[1]} available")
    return x[:, x.shape[1] - n :].copy()


# --------------------------------------------------------------------------- #
# Analysis helpers (docs/tests)
# --------------------------------------------------------------------------- #
def frequency_response_db(
    sos: np.ndarray | None, fs: float, freqs_hz: np.ndarray | list[float]
) -> np.ndarray:
    """Magnitude response in dB at ``freqs_hz`` (``sosfreqz``). ``sos=None`` -> 0 dB.

    Exact zeros of the response are floored at -400 dB to avoid ``-inf``.
    """
    fs = _check_fs(fs)
    f = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64))
    if sos is None:
        return np.zeros_like(f)
    _, h = signal.sosfreqz(_check_sos(sos), worN=f, fs=fs)
    return 20.0 * np.log10(np.maximum(np.abs(h), 1e-20))


def group_delay_s(
    sos: np.ndarray | None, fs: float, freqs_hz: np.ndarray | list[float]
) -> np.ndarray:
    """Group delay in seconds at ``freqs_hz`` (sum of per-section group delays).

    Undefined exactly at a notch zero; evaluate away from ``notch_hz``.
    """
    fs = _check_fs(fs)
    f = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64))
    if sos is None:
        return np.zeros_like(f)
    total = np.zeros_like(f)
    for sec in _check_sos(sos):
        _, gd = signal.group_delay((sec[:3], sec[3:]), w=f, fs=fs)
        total += gd
    return total / fs


def settling_time_s(
    sos: np.ndarray | None, fs: float, tol: float = 1e-3, max_s: float = 60.0
) -> float:
    """Time after which the impulse response's remaining energy is ``<= tol`` of its total.

    I.e. the smallest ``t`` with ``sum(h[t:]**2) <= tol * sum(h**2)``. Quantifies
    how long a start-up / initial-state transient of the causal filter lingers.
    Returns 0.0 for ``sos=None``.

    Raises:
        ValueError: if the criterion is not met within ``max_s`` seconds.
    """
    fs = _check_fs(fs)
    if not (0 < tol < 1):
        raise ValueError(f"tol must be in (0, 1); got {tol}")
    if sos is None:
        return 0.0
    n = int(np.ceil(max_s * fs))
    impulse = np.zeros(n)
    impulse[0] = 1.0
    h = signal.sosfilt(_check_sos(sos), impulse)
    energy = h**2
    tail = np.cumsum(energy[::-1])[::-1] / energy.sum()
    below = np.nonzero(tail <= tol)[0]
    if below.size == 0:
        raise ValueError(f"impulse response not settled to tol={tol} within {max_s} s")
    return float(below[0] / fs)
