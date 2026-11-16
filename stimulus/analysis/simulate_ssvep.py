"""Mock SSVEP power-spectrum figure, for presentation slides.

THIS IS SYNTHETIC DATA FOR ILLUSTRATION ONLY -- not a real EEG recording.
It exists to show what a successful SSVEP result *should* look like
(a clean power-spectrum peak at the attended target's frequency) before
real hardware/EEG data is available.

Simulates one mock trial per target, using the ACTUAL configured
frequencies and accent colors from synapsys_stimulus.config, so this
figure stays consistent with whatever the renderer is currently using.

Regenerate with:

    python analysis/simulate_ssvep.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from synapsys_stimulus.config import RendererConfig

OUTPUT_PATH = Path(__file__).parent / "ssvep_mock_spectrum.png"

RNG = np.random.default_rng(seed=7)

FS_HZ = 250.0  # matches the ADS1299's configured sample rate
TRIAL_SECONDS = 4.0
HARMONIC_AMPLITUDES = (2.6, 0.9, 0.3)  # fundamental, 2nd, 3rd harmonic
NON_ATTENDED_LEAKAGE = 0.12  # small residual response at the other targets

# Below this, 1/f background power dominates any real EEG spectrum too --
# a real analysis pipeline high-pass filters it out before looking for
# SSVEP peaks, so the plot excludes it the same way.
DISPLAY_MIN_HZ = 3.0
DISPLAY_MAX_HZ = 30.0


def simulate_trial(
    attended_freq_hz: float,
    all_freqs_hz: list[float],
    fs: float = FS_HZ,
    duration: float = TRIAL_SECONDS,
) -> np.ndarray:
    """Simulate one occipital-electrode trial: pink noise + alpha rhythm + SSVEP."""
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs

    # 1/f "pink" background, approximated by shaping white noise in the
    # frequency domain -- real scalp EEG power falls off roughly like 1/f.
    white = RNG.standard_normal(n_samples)
    freqs = np.fft.rfftfreq(n_samples, d=1 / fs)
    spectrum = np.fft.rfft(white)
    pink_filter = 1.0 / np.sqrt(np.maximum(freqs, 0.5))
    pink = np.fft.irfft(spectrum * pink_filter, n=n_samples)
    pink *= 2.0 / np.std(pink)

    # Resting occipital alpha rhythm (~10 Hz), present regardless of task.
    alpha = 0.6 * np.sin(2 * np.pi * 10.0 * t + RNG.uniform(0, 2 * np.pi))

    # SSVEP response: strong at the attended frequency (with harmonics),
    # weak residual leakage at the other two target frequencies -- real
    # SSVEP responses aren't perfectly selective either.
    signal = np.zeros(n_samples)
    for freq in all_freqs_hz:
        scale = 1.0 if freq == attended_freq_hz else NON_ATTENDED_LEAKAGE
        phase = RNG.uniform(0, 2 * np.pi)
        for harmonic_index, amplitude in enumerate(HARMONIC_AMPLITUDES, start=1):
            signal += scale * amplitude * np.sin(2 * np.pi * freq * harmonic_index * t + phase)

    measurement_noise = 0.4 * RNG.standard_normal(n_samples)
    return pink + alpha + signal + measurement_noise


def welch_psd(x: np.ndarray, fs: float, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    """Minimal Welch power-spectral-density estimate (Hann window, 50% overlap).

    Avoids depending on scipy.signal, whose import hangs in this
    environment (likely an OpenMP/BLAS conflict between PsychoPy's bundled
    numpy/scipy build and the system libraries) -- unrelated to this
    script's logic, just worked around here.
    """
    step = nperseg // 2
    window = np.hanning(nperseg)
    window_power = np.sum(window**2)

    segments = [x[start : start + nperseg] for start in range(0, len(x) - nperseg + 1, step)]
    freqs = np.fft.rfftfreq(nperseg, d=1 / fs)
    psd_accumulator = np.zeros(len(freqs))
    for segment in segments:
        windowed = (segment - np.mean(segment)) * window
        spectrum = np.fft.rfft(windowed)
        psd_accumulator += (np.abs(spectrum) ** 2) / (fs * window_power)

    return freqs, psd_accumulator / len(segments)


def plot_spectra() -> None:
    target_freqs = [s.nominal_frequency_hz for s in RendererConfig.STIMULI]
    labels = [s.label for s in RendererConfig.STIMULI]
    colors = RendererConfig.ACCENT_COLORS

    # Compute all three spectra first so the shared y-axis can be scaled to
    # the displayed range -- not to the excluded low-frequency 1/f hump.
    results = []
    for attended_freq in target_freqs:
        trial = simulate_trial(attended_freq, target_freqs)
        freqs, power = welch_psd(trial, fs=FS_HZ, nperseg=int(FS_HZ * 2))
        mask = (freqs >= DISPLAY_MIN_HZ) & (freqs <= DISPLAY_MAX_HZ)
        results.append((freqs[mask], power[mask]))

    y_max = max(power.max() for _, power in results) * 1.25

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    for ax, (freqs, power), attended_freq, label, color in zip(
        axes, results, target_freqs, labels, colors, strict=True
    ):
        ax.plot(freqs, power, color="#333333", linewidth=1.2)
        ax.fill_between(freqs, power, color="#333333", alpha=0.05)

        peak_index = int(np.argmin(np.abs(freqs - attended_freq)))
        ax.axvline(attended_freq, color=color, linestyle="--", linewidth=1.3)
        ax.plot(freqs[peak_index], power[peak_index], "o", color=color, markersize=6)
        ax.annotate(
            f"{attended_freq:.3f} Hz",
            xy=(attended_freq, power[peak_index]),
            xytext=(attended_freq + 1.2, min(power[peak_index] + y_max * 0.05, y_max * 0.95)),
            fontsize=9,
            color=color,
            fontweight="bold",
        )

        ax.set_title(f"Attending: {label}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xlim(DISPLAY_MIN_HZ, DISPLAY_MAX_HZ)
        ax.set_ylim(0, y_max)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Power (a.u.)")
    fig.suptitle(
        "Simulated occipital SSVEP power spectrum by attended target",
        fontsize=13,
        fontweight="bold",
        y=1.04,
    )
    fig.text(
        0.5,
        -0.03,
        "Synthetic data for illustration only — not a real EEG recording.",
        ha="center",
        fontsize=9,
        color="#888888",
        style="italic",
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    plot_spectra()
