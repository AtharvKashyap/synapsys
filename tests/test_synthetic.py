import numpy as np
import pytest

from synapsys_decoder import synthetic as syn

FS = 250.0
CH = ("O1", "Oz", "O2", "POz")


def rng(seed=0):
    return np.random.default_rng(seed)


def peak_amp(x, fs, f):
    spec = np.abs(np.fft.rfft(x, axis=-1)) / x.shape[-1] * 2
    freqs = np.fft.rfftfreq(x.shape[-1], 1 / fs)
    return spec[:, np.argmin(np.abs(freqs - f))]


def test_shape_units_and_labels():
    t = syn.make_synthetic_trial(12.0, FS, 2.0, CH, rng(), label="NO")
    assert t.eeg.shape == (4, 500) and t.eeg.dtype == np.float64
    assert 1e-7 < np.std(t.eeg) < 1e-5
    assert t.true_frequency == 12.0 and t.true_label == "NO"
    assert t.channel_names == CH


def test_determinism():
    a = syn.make_synthetic_trial(12.0, FS, 2.0, CH, rng(1), snr_db=-5)
    b = syn.make_synthetic_trial(12.0, FS, 2.0, CH, rng(1), snr_db=-5)
    c = syn.make_synthetic_trial(12.0, FS, 2.0, CH, rng(2), snr_db=-5)
    np.testing.assert_array_equal(a.eeg, b.eeg)
    assert not np.allclose(a.eeg, c.eeg)


@pytest.mark.parametrize("snr", [-10.0, 0.0, 10.0])
def test_measured_snr(snr):
    r = rng(3)
    clean = syn.ssvep_component(12.0, FS, 25000, 4, 2, r)
    noise_std = syn.noise_std_for_snr(clean, snr)
    noisy = clean + syn.gaussian_noise(clean.shape, r, noise_std)
    p_s = np.mean(clean**2)
    p_n = np.mean((noisy - clean) ** 2)
    assert 10 * np.log10(p_s / p_n) == pytest.approx(snr, abs=0.2)
    t = syn.make_synthetic_trial(12.0, FS, 100.0, CH, rng(4), snr_db=snr)
    assert t.metadata["snr_db"] == snr


def test_clean_peak_and_harmonic():
    t = syn.make_synthetic_trial(12.0, FS, 4.0, CH, rng(), harmonics=2)
    p1, p2, p_off = (peak_amp(t.eeg, FS, f) for f in (12.0, 24.0, 17.0))
    assert np.all(p1 > 0.4e-6) and np.all(p2 > 0.2e-6)
    assert np.all(p_off < 1e-9)
    assert p1.mean() > p2.mean()


def test_drift_and_line_frequencies():
    d = syn.drift(2, 2500, FS, rng(), freq_hz=0.2, amplitude_v=20e-6)
    ln = syn.line_noise(2, 2500, FS, rng(), amplitude_v=5e-6)
    assert np.all(peak_amp(ln, FS, 60.0) == pytest.approx(5e-6, rel=0.02))
    assert np.all(np.abs(d).max(axis=1) <= 20e-6 + 1e-12)
    freqs = np.fft.rfftfreq(2500, 1 / FS)
    assert freqs[np.argmax(np.abs(np.fft.rfft(d - d.mean(1, keepdims=True), axis=1))[0])] < 0.5
    t = syn.make_synthetic_trial(
        12.0, FS, 4.0, CH, rng(), line_amplitude_v=5e-6, drift_amplitude_v=20e-6
    )
    assert np.all(peak_amp(t.eeg, FS, 60.0) > 4e-6)


def test_channel_scaling_changes_rms():
    base = np.ones((4, 100))
    scaled = syn.scale_channels(base, rng(), 0.5, 1.5)
    gains = scaled[:, 0]
    assert np.all((gains >= 0.5) & (gains <= 1.5)) and np.ptp(gains) > 0
    t = syn.make_synthetic_trial(12.0, FS, 2.0, CH, rng(5), channel_scaling=True)
    assert t.metadata["channel_scaling"] is True


def test_mixture_contains_both():
    t = syn.make_mixture_trial(8.0, 12.0, 0.5, FS, 4.0, CH, rng())
    assert np.all(peak_amp(t.eeg, FS, 8.0) > 0.1e-6)
    assert np.all(peak_amp(t.eeg, FS, 12.0) > 0.1e-6)
    assert t.true_label is None and t.metadata["kind"] == "mixture"
    only1 = syn.make_mixture_trial(8.0, 12.0, 0.0, FS, 4.0, CH, rng())
    assert np.all(peak_amp(only1.eeg, FS, 12.0) < 1e-9)


def test_noise_and_off_target():
    n = syn.make_noise_trial(FS, 2.0, CH, rng(), std_v=2e-6)
    assert n.true_label is None and n.true_frequency is None
    assert np.std(n.eeg) == pytest.approx(2e-6, rel=0.1)
    o = syn.make_off_target_trial(10.0, FS, 2.0, CH, rng(), snr_db=0.0)
    assert o.true_label is None and o.metadata["kind"] == "off_target"
    assert o.metadata["snr_db"] == 0.0


def test_metadata_complete():
    t = syn.make_synthetic_trial(
        12.0, FS, 2.0, CH, rng(), snr_db=3.0, metadata={"extra": 1}, label="NO"
    )
    for k in (
        "frequency_hz",
        "fs",
        "duration_s",
        "snr_db",
        "harmonics",
        "amplitude_v",
        "drift_amplitude_v",
        "line_amplitude_v",
        "line_freq_hz",
        "channel_scaling",
        "extra",
    ):
        assert k in t.metadata


def test_invalid_args():
    with pytest.raises(ValueError):
        syn.make_synthetic_trial(12.0, 0.0, 2.0, CH, rng())
    with pytest.raises(ValueError):
        syn.make_synthetic_trial(12.0, FS, 0.0, CH, rng())
    with pytest.raises(ValueError):
        syn.make_mixture_trial(8.0, 12.0, 1.5, FS, 2.0, CH, rng())
    with pytest.raises(ValueError):
        syn.make_mixture_trial(8.0, 12.0, -0.1, FS, 2.0, CH, rng())
    with pytest.raises(ValueError):
        syn.make_synthetic_trial(70.0, FS, 2.0, CH, rng(), harmonics=2)  # 140 > 125
    with pytest.raises(ValueError):
        syn.make_synthetic_trial(12.0, FS, 2.0, CH, rng(), harmonics=0)
    with pytest.raises(ValueError):
        syn.noise_std_for_snr(np.zeros((2, 10)), 0.0)
