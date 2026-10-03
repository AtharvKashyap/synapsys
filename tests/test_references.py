import numpy as np
import pytest

from synapsys_decoder.references import harmonic_references, reference_bank

FS = 250.0
FREQS = [60 / 7, 12.0, 15.0]


def _fft_peak_hz(signal: np.ndarray, fs: float, n_fft: int = 2**20) -> float:
    spec = np.abs(np.fft.rfft(signal * np.hanning(signal.size), n=n_fft))
    return float(np.fft.rfftfreq(n_fft, d=1 / fs)[np.argmax(spec)])


def test_shape_and_dtype():
    ref = harmonic_references(12.0, FS, 625, 2)
    assert ref.shape == (4, 625)
    assert ref.dtype == np.float64
    assert harmonic_references(12.0, FS, 1, 3).shape == (6, 1)


def test_deterministic():
    a = harmonic_references(60 / 7, FS, 500, 2)
    b = harmonic_references(60 / 7, FS, 500, 2)
    np.testing.assert_array_equal(a, b)


def test_first_sample_values():
    ref = harmonic_references(15.0, FS, 100, 3)
    np.testing.assert_allclose(ref[0::2, 0], 0.0, atol=0)
    np.testing.assert_allclose(ref[1::2, 0], 1.0, atol=0)


def test_row_values_match_formula():
    n, f = 300, 12.0
    t = np.arange(n) / FS
    ref = harmonic_references(f, FS, n, 2)
    np.testing.assert_allclose(ref[0], np.sin(2 * np.pi * f * t), atol=1e-12)
    np.testing.assert_allclose(ref[1], np.cos(2 * np.pi * f * t), atol=1e-12)
    np.testing.assert_allclose(ref[2], np.sin(2 * np.pi * 2 * f * t), atol=1e-12)
    np.testing.assert_allclose(ref[3], np.cos(2 * np.pi * 2 * f * t), atol=1e-12)


@pytest.mark.parametrize("f", FREQS)
def test_fft_peak_at_each_harmonic(f):
    ref = harmonic_references(f, FS, 2500, 2)
    for h in (1, 2):
        for row in (2 * (h - 1), 2 * (h - 1) + 1):
            assert _fft_peak_hz(ref[row], FS) == pytest.approx(h * f, abs=2e-3)


def test_reference_bank_matches_individual():
    bank = reference_bank(FREQS, FS, 625, 2)
    assert bank.shape == (3, 4, 625)
    for i, f in enumerate(FREQS):
        np.testing.assert_array_equal(bank[i], harmonic_references(f, FS, 625, 2))


def test_reference_bank_empty_raises():
    with pytest.raises(ValueError):
        reference_bank([], FS, 625, 2)


@pytest.mark.parametrize(
    "args",
    [
        (0.0, FS, 100, 2),
        (-1.0, FS, 100, 2),
        (12.0, 0.0, 100, 2),
        (12.0, -250.0, 100, 2),
        (12.0, FS, 0, 2),
        (12.0, FS, 100, 0),
        (float("nan"), FS, 100, 2),
    ],
)
def test_invalid_inputs_raise(args):
    with pytest.raises(ValueError):
        harmonic_references(*args)


def test_above_nyquist_refused():
    with pytest.raises(ValueError, match="Nyquist"):
        harmonic_references(50.0, FS, 100, 3)  # 150 Hz > 125 Hz
    with pytest.raises(ValueError, match="Nyquist"):
        harmonic_references(62.5, FS, 100, 2)  # exactly at Nyquist
    with pytest.raises(ValueError, match="Nyquist"):
        reference_bank([12.0, 70.0], FS, 100, 2)
    harmonic_references(62.0, FS, 100, 2)  # just below: fine
