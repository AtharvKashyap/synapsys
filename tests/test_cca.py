import numpy as np
import pytest

from synapsys_decoder.cca import DegenerateSignalError, canonical_correlations, cca_score
from synapsys_decoder.references import harmonic_references

FS = 250.0
N = 625
FREQS = [60 / 7, 12.0, 15.0]


def _ssvep(f, n_channels=4, n=N, noise=0.0, seed=0, amp=1e-5):
    """Channel-specific mixtures of sin/cos at f and 2f plus white noise (volts)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    eeg = np.zeros((n_channels, n))
    for c in range(n_channels):
        for h, a in ((1, 1.0), (2, 0.5)):
            phase = rng.uniform(0, 2 * np.pi)
            gain = rng.uniform(0.5, 1.5) * a * amp
            eeg[c] += gain * np.sin(2 * np.pi * h * f * t + phase)
    return eeg + noise * amp * rng.standard_normal(eeg.shape)


def _textbook_cca(x, y):
    """Independent covariance formulation: sqrt(eig(Cxx^-1 Cxy Cyy^-1 Cyx))."""
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    cxx, cyy, cxy = xc @ xc.T, yc @ yc.T, xc @ yc.T
    m = np.linalg.solve(cxx, cxy) @ np.linalg.solve(cyy, cxy.T)
    ev = np.sort(np.real(np.linalg.eigvals(m)))[::-1][: min(x.shape[0], y.shape[0])]
    return np.sqrt(np.clip(ev, 0, None))


def test_finite_and_in_unit_interval():
    rng = np.random.default_rng(1)
    for f in FREQS:
        rho = canonical_correlations(rng.standard_normal((4, N)), harmonic_references(f, FS, N, 2))
        assert rho.shape == (4,)
        assert np.all(np.isfinite(rho))
        assert np.all((rho >= 0) & (rho <= 1))
        assert np.all(np.diff(rho) <= 0)


def test_perfect_linear_combination_scores_one():
    rng = np.random.default_rng(2)
    ref = harmonic_references(12.0, FS, N, 2)
    eeg = rng.standard_normal((4, 4)) @ ref
    assert cca_score(eeg, ref) == pytest.approx(1.0, abs=1e-10)
    np.testing.assert_allclose(canonical_correlations(eeg, ref), 1.0, atol=1e-10)


@pytest.mark.parametrize("f", FREQS)
def test_matched_frequency_wins(f):
    eeg = _ssvep(f, noise=1.0, seed=3)
    matched = cca_score(eeg, harmonic_references(f, FS, N, 2))
    for g in FREQS:
        if g != f:
            assert matched > cca_score(eeg, harmonic_references(g, FS, N, 2)) + 0.2


def test_noise_reduces_score():
    ref = harmonic_references(15.0, FS, N, 2)
    scores = [cca_score(_ssvep(15.0, noise=s, seed=4), ref) for s in (0.0, 0.5, 3.0, 20.0)]
    assert scores[0] == pytest.approx(1.0, abs=1e-9)
    assert all(np.isfinite(scores))
    assert scores[0] > scores[1] > scores[2] > scores[3]


def test_channel_permutation_invariance():
    eeg = _ssvep(12.0, noise=2.0, seed=5)
    perm = eeg[[2, 0, 3, 1]]
    for f in FREQS:
        ref = harmonic_references(f, FS, N, 2)
        np.testing.assert_allclose(
            canonical_correlations(perm, ref), canonical_correlations(eeg, ref), atol=1e-12
        )


def test_per_channel_scaling_invariance():
    eeg = _ssvep(60 / 7, noise=2.0, seed=6)
    scaled = eeg * np.array([[1e3], [0.01], [-2.0], [7.5]])
    for f in FREQS:
        ref = harmonic_references(f, FS, N, 2)
        assert cca_score(scaled, ref) == pytest.approx(cca_score(eeg, ref), abs=1e-10)


def test_dc_offset_invariance():
    eeg = _ssvep(12.0, noise=2.0, seed=7)
    ref = harmonic_references(12.0, FS, N, 2)
    offset = eeg + np.array([[1e-3], [-2e-3], [5e-4], [0.0]])
    assert cca_score(offset, ref) == pytest.approx(cca_score(eeg, ref), abs=1e-8)


def test_duplicate_channel_rank_deficient():
    eeg = _ssvep(12.0, noise=2.0, seed=8)
    dup = np.vstack([eeg[:3], eeg[2:3]])
    ref = harmonic_references(12.0, FS, N, 2)
    rho = canonical_correlations(dup, ref)
    assert rho.shape == (3,)
    assert np.all(np.isfinite(rho))
    assert rho[0] == pytest.approx(cca_score(eeg[:3], ref), abs=1e-10)


def test_zero_channel_mixed_with_real():
    eeg = _ssvep(15.0, noise=2.0, seed=9)
    eeg[1] = 0.0
    ref = harmonic_references(15.0, FS, N, 2)
    rho = canonical_correlations(eeg, ref)
    assert rho.shape == (3,)
    assert rho[0] == pytest.approx(cca_score(eeg[[0, 2, 3]], ref), abs=1e-10)
    # constant (non-zero) channel behaves the same way
    eeg[1] = 3e-4
    assert cca_score(eeg, ref) == pytest.approx(rho[0], abs=1e-10)


def test_all_constant_eeg_raises():
    ref = harmonic_references(12.0, FS, N, 2)
    with pytest.raises(DegenerateSignalError):
        cca_score(np.zeros((4, N)), ref)
    with pytest.raises(DegenerateSignalError):
        cca_score(np.full((4, N), 1.234567e-5), ref)
    with pytest.raises(DegenerateSignalError):
        cca_score(np.array([[0.1], [0.2], [0.3], [0.4]]) * np.ones((4, N)), ref)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_raises(bad):
    eeg = _ssvep(12.0, noise=1.0, seed=10)
    eeg[2, 100] = bad
    ref = harmonic_references(12.0, FS, N, 2)
    with pytest.raises(DegenerateSignalError):
        cca_score(eeg, ref)
    with pytest.raises(DegenerateSignalError):
        cca_score(ref, eeg)


def test_too_few_samples_raises():
    rng = np.random.default_rng(11)
    x, y = rng.standard_normal((4, 8)), rng.standard_normal((4, 8))
    with pytest.raises(DegenerateSignalError):
        canonical_correlations(x, y)  # n == p + q
    rho = canonical_correlations(rng.standard_normal((4, 9)), rng.standard_normal((4, 9)))
    assert np.all(np.isfinite(rho))  # n == p + q + 1 is the minimum accepted


def test_shape_errors():
    with pytest.raises(ValueError):
        canonical_correlations(np.zeros(10), np.zeros((2, 10)))
    with pytest.raises(ValueError):
        canonical_correlations(np.ones((2, 10)), np.ones((2, 11)))
    with pytest.raises(ValueError):
        canonical_correlations(np.ones((2, 10)), np.ones((2, 10)), rank_tol=0.0)


def test_degenerate_error_is_value_error():
    assert issubclass(DegenerateSignalError, ValueError)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cross_check_textbook_covariance(seed):
    rng = np.random.default_rng(seed)
    n = 500
    x = rng.standard_normal((4, n))
    y = rng.standard_normal((4, n)) + 0.6 * rng.standard_normal((4, 4)) @ x
    np.testing.assert_allclose(canonical_correlations(x, y), _textbook_cca(x, y), atol=1e-9)
    ref = harmonic_references(12.0, FS, n, 2)
    eeg = 0.3 * rng.standard_normal((4, 4)) @ ref + rng.standard_normal((4, n))
    np.testing.assert_allclose(canonical_correlations(eeg, ref), _textbook_cca(eeg, ref), atol=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cross_check_sklearn(seed):
    cross_decomposition = pytest.importorskip("sklearn.cross_decomposition")
    rng = np.random.default_rng(seed)
    n = 625
    ref = harmonic_references(60 / 7, FS, n, 2)
    eeg = 0.4 * rng.standard_normal((4, 4)) @ ref + rng.standard_normal((4, n))
    model = cross_decomposition.CCA(n_components=1, scale=True, max_iter=10000, tol=1e-14)
    u, v = model.fit_transform(eeg.T, ref.T)
    sk = abs(np.corrcoef(u[:, 0], v[:, 0])[0, 1])
    assert cca_score(eeg, ref) == pytest.approx(sk, abs=1e-6)
