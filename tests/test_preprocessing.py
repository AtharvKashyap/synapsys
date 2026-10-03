"""Tests for synapsys_decoder.preprocessing (M5).

dB bounds are derived from the documented design (module docstring table):
fs=250 Hz, Butterworth N=4 (8 poles) 0.5-45 Hz, 60 Hz iirnotch Q=30.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from synapsys_decoder.config import PreprocessingConfig
from synapsys_decoder.preprocessing import (
    StreamingFilter,
    build_filter_sos,
    design_bandpass,
    design_notch,
    extract_window,
    frequency_response_db,
    group_delay_s,
    last_window,
    preprocess,
    settling_time_s,
)

FS = 250.0
CFG = PreprocessingConfig()  # defaults == configs/synapsys.yaml
CFG_NO_NOTCH = replace(CFG, notch_hz=None)
CFG_ZP = replace(CFG, mode="zero_phase")
F_YES = 60.0 / 7.0


# --------------------------------------------------------------------------- #
# Design structure
# --------------------------------------------------------------------------- #
def test_sos_structure_and_order_meaning():
    bp = design_bandpass(FS, 0.5, 45.0, 4)
    assert bp.shape == (4, 6)  # N=4 bandpass -> 2N=8 poles -> 4 biquads
    assert design_bandpass(FS, 0.5, None, 4).shape == (2, 6)  # high-pass only, 4 poles
    assert design_bandpass(FS, None, 45.0, 3).shape == (2, 6)  # low-pass, 3 poles
    assert design_bandpass(FS, None, None, 4) is None
    notch = design_notch(FS, 60.0, 30.0)
    assert notch.shape == (1, 6)
    sos = build_filter_sos(FS, CFG)
    assert sos.shape == (5, 6)
    np.testing.assert_array_equal(sos[:4], bp)  # bandpass rows first
    np.testing.assert_array_equal(sos[4], notch[0])  # then notch
    np.testing.assert_allclose(sos[:, 3], 1.0)  # a0 normalised
    assert build_filter_sos(
        FS, replace(CFG, bandpass_low_hz=None, bandpass_high_hz=None)
    ).shape == (
        1,
        6,
    )
    assert (
        build_filter_sos(
            FS, replace(CFG, bandpass_low_hz=None, bandpass_high_hz=None, notch_hz=None)
        )
        is None
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(low_hz=0.5, high_hz=125.0, order=4),  # at Nyquist
        dict(low_hz=130.0, high_hz=None, order=4),
        dict(low_hz=20.0, high_hz=10.0, order=4),
        dict(low_hz=0.5, high_hz=45.0, order=0),
        dict(low_hz=-1.0, high_hz=45.0, order=4),
    ],
)
def test_design_bandpass_invalid(kwargs):
    with pytest.raises(ValueError):
        design_bandpass(FS, **kwargs)


def test_design_notch_invalid():
    with pytest.raises(ValueError):
        design_notch(FS, 125.0, 30.0)
    with pytest.raises(ValueError):
        design_notch(100.0, 60.0, 30.0)  # 60 Hz above Nyquist of 50 Hz
    with pytest.raises(ValueError):
        design_notch(FS, 60.0, 0.0)


# --------------------------------------------------------------------------- #
# Frequency response (documented table)
# --------------------------------------------------------------------------- #
PASSBAND = [F_YES, 12.0, 15.0, 2 * F_YES, 24.0, 30.0]


@pytest.mark.parametrize("cfg", [CFG, CFG_NO_NOTCH])
def test_passband_targets_and_harmonics(cfg):
    db = frequency_response_db(build_filter_sos(FS, cfg), FS, PASSBAND)
    # Fundamentals and 2nd harmonics: < 0.1 dB loss (worst is 30 Hz: -0.09 dB).
    assert np.all(db <= 1e-6)
    assert np.all(db > -0.1)
    # Fundamentals essentially flat.
    assert np.all(db[:3] > -0.005)


def test_documented_response_table():
    f = [0.2, 0.5, 45.0, 50.0, 60.0, 80.0]
    bp = frequency_response_db(build_filter_sos(FS, CFG_NO_NOTCH), FS, f)
    full = frequency_response_db(build_filter_sos(FS, CFG), FS, f)
    np.testing.assert_allclose(bp, [-32.13, -3.01, -3.01, -6.03, -13.98, -31.89], atol=0.02)
    np.testing.assert_allclose(
        full[[0, 1, 2, 3, 5]], [-32.13, -3.01, -3.03, -6.07, -31.90], atol=0.02
    )
    # Drift and >= 80 Hz attenuated by >= 30 dB.
    assert bp[0] < -30 and bp[5] < -30
    # 60 Hz: low-pass alone ~14 dB; the notch adds > 85 dB more.
    assert -15.0 < bp[4] < -13.0
    assert full[4] < -100.0


def test_notch_is_narrow():
    notch = design_notch(FS, 60.0, 30.0)
    db = frequency_response_db(notch, FS, [55.0, 58.0, 59.0, 61.0, 62.0, 65.0, 30.0])
    np.testing.assert_allclose(db[[2, 3]], -3.01, atol=0.02)  # bandwidth 60/30 = 2 Hz
    np.testing.assert_allclose(db[[1, 4]], -0.97, atol=0.02)
    assert np.all(db[[0, 5]] > -0.2)
    assert db[6] > -0.01


def test_group_delay_at_targets():
    gd = group_delay_s(build_filter_sos(FS, CFG), FS, [F_YES, 12.0, 15.0])
    np.testing.assert_allclose(gd, [0.01146, 0.01027, 0.01000], atol=2e-4)
    assert np.all(gd < 3 / FS)


def test_settling_time():
    sos = build_filter_sos(FS, CFG)
    t2 = settling_time_s(sos, FS, tol=1e-2)
    t3 = settling_time_s(sos, FS, tol=1e-3)
    t4 = settling_time_s(sos, FS, tol=1e-4)
    assert t2 == pytest.approx(0.232, abs=0.01)
    assert t3 == pytest.approx(1.052, abs=0.02)
    assert t4 == pytest.approx(2.096, abs=0.03)
    assert settling_time_s(None, FS) == 0.0
    # Without the 0.5 Hz high-pass the filter settles far faster.
    lp_only = build_filter_sos(FS, replace(CFG, bandpass_low_hz=None))
    assert settling_time_s(lp_only, FS, tol=1e-3) < 0.2


# --------------------------------------------------------------------------- #
# Signal test
# --------------------------------------------------------------------------- #
AMP = {12.0: 10e-6, 0.2: 50e-6, 60.0: 20e-6, 80.0: 5e-6}


def _mixture(n_s: float = 24.0, n_ch: int = 4) -> np.ndarray:
    t = np.arange(round(n_s * FS)) / FS
    rows = []
    for ch in range(n_ch):
        x = 0.01 * (ch + 1) * np.ones_like(t)  # large electrode offset (volts)
        for k, (f, a) in enumerate(AMP.items()):
            x = x + a * np.sin(2 * np.pi * f * t + 0.3 * ch + k)
        rows.append(x)
    return np.array(rows)


def _amplitude(y: np.ndarray, f: float) -> np.ndarray:
    """Per-channel sinusoid amplitude at f by projection (segment holds integer cycles)."""
    t = np.arange(y.shape[1]) / FS
    c = np.cos(2 * np.pi * f * t)
    s = np.sin(2 * np.pi * f * t)
    n = y.shape[1]
    return 2 / n * np.hypot(y @ c, y @ s)


@pytest.mark.parametrize(
    ("cfg", "seg", "power"),
    [(CFG, slice(2500, 5000), 1), (CFG_ZP, slice(1250, 3750), 2)],
    ids=["causal", "zero_phase"],
)
def test_signal_components(cfg, seg, power):
    x = _mixture()
    y = preprocess(x, FS, cfg)
    assert y.shape == x.shape and y.dtype == np.float64
    ys = y[:, seg]  # 10 s, settled (causal) / away from edges (zero-phase)
    sos = build_filter_sos(FS, cfg)
    for f, a in AMP.items():
        measured = _amplitude(ys, f) / a
        expected = 10 ** (power * frequency_response_db(sos, FS, [f])[0] / 20)
        if f == 12.0:
            np.testing.assert_allclose(measured, 1.0, atol=0.01)
        elif f == 60.0:
            assert np.all(measured < 1e-3)
        else:
            # Drift: -32 dB per pass; 80 Hz: -32 dB per pass.
            np.testing.assert_allclose(measured, expected, rtol=0.1, atol=2e-4)
            assert np.all(measured < (0.03 if power == 1 else 1e-3))
    # The electrode offset is gone.
    assert np.all(np.abs(ys.mean(axis=1)) < 1e-6)


def test_signal_60hz_without_notch():
    y = preprocess(_mixture(), FS, CFG_NO_NOTCH)
    measured = _amplitude(y[:, 2500:5000], 60.0) / AMP[60.0]
    np.testing.assert_allclose(measured, 10 ** (-13.98 / 20), rtol=0.05)  # ~0.20


# --------------------------------------------------------------------------- #
# Causality
# --------------------------------------------------------------------------- #
def test_causal_ignores_future_and_zero_phase_does_not():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 2000)) * 1e-5 + 0.02
    k = 1200
    x2 = x.copy()
    x2[:, k:] += rng.standard_normal((4, 2000 - k)) * 1e-4
    y1, y2 = preprocess(x, FS, CFG), preprocess(x2, FS, CFG)
    np.testing.assert_array_equal(y1[:, :k], y2[:, :k])
    assert not np.allclose(y1[:, k:], y2[:, k:])
    z1, z2 = preprocess(x, FS, CFG_ZP), preprocess(x2, FS, CFG_ZP)
    assert not np.allclose(z1[:, :k], z2[:, :k], rtol=0, atol=1e-12)


def test_causal_dc_policy_equals_sosfilt_zi_init():
    from scipy import signal

    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 1500)) * 1e-5 + np.array([[0.01], [-0.02], [0.005], [0.0]])
    sos = build_filter_sos(FS, CFG)
    zi = signal.sosfilt_zi(sos)
    ref = np.stack([signal.sosfilt(sos, row, zi=zi * row[0])[0] for row in x])
    np.testing.assert_allclose(preprocess(x, FS, CFG), ref, rtol=0, atol=1e-15)


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cfg", [CFG, replace(CFG, remove_dc=False), replace(CFG, notch_hz=None)])
def test_streaming_matches_offline_causal(cfg):
    rng = np.random.default_rng(42)
    x = rng.standard_normal((4, 5000)) * 1e-5 + 0.01
    offline = preprocess(x, FS, cfg)
    sf = StreamingFilter.from_config(FS, cfg, n_channels=4)
    outs, i = [], 0
    while i < x.shape[1]:
        n = int(rng.integers(0, 97))  # includes empty chunks
        outs.append(sf.process(x[:, i : i + n]))
        i += n
    np.testing.assert_allclose(np.concatenate(outs, axis=1), offline, rtol=0, atol=1e-15)
    sf.reset()
    np.testing.assert_allclose(sf.process(x), offline, rtol=0, atol=1e-15)


def test_streaming_validation():
    with pytest.raises(ValueError):
        StreamingFilter.from_config(FS, CFG_ZP, n_channels=4)
    sf = StreamingFilter(build_filter_sos(FS, CFG), n_channels=4)
    with pytest.raises(ValueError):
        sf.process(np.zeros((3, 10)))
    with pytest.raises(ValueError):
        sf.process(np.full((4, 10), np.nan))
    with pytest.raises(ValueError):
        StreamingFilter(None, n_channels=0)
    passthrough = StreamingFilter(None, n_channels=2, remove_dc=False)
    x = np.arange(10.0).reshape(2, 5)
    np.testing.assert_array_equal(passthrough.process(x), x)


# --------------------------------------------------------------------------- #
# Stage toggles and input validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["causal", "zero_phase"])
@pytest.mark.parametrize(
    "changes",
    [
        dict(bandpass_low_hz=None),
        dict(bandpass_high_hz=None),
        dict(bandpass_low_hz=None, bandpass_high_hz=None),
        dict(notch_hz=None),
        dict(remove_dc=False),
        dict(bandpass_low_hz=None, bandpass_high_hz=None, notch_hz=None),
        dict(bandpass_low_hz=None, bandpass_high_hz=None, notch_hz=None, remove_dc=False),
    ],
)
def test_stage_toggles(mode, changes):
    x = _mixture(n_s=4.0)
    cfg = replace(CFG, mode=mode, **changes)
    y = preprocess(x, FS, cfg)
    assert y.shape == x.shape and y.dtype == np.float64
    assert np.all(np.isfinite(y))
    assert y is not x


def test_all_disabled_dc_policy():
    x = _mixture(n_s=2.0)
    off = dict(bandpass_low_hz=None, bandpass_high_hz=None, notch_hz=None)
    np.testing.assert_array_equal(preprocess(x, FS, replace(CFG, remove_dc=False, **off)), x)
    np.testing.assert_allclose(preprocess(x, FS, replace(CFG, **off)), x - x[:, :1])
    zp = preprocess(x, FS, replace(CFG_ZP, **off))
    np.testing.assert_allclose(zp, x - x.mean(axis=1, keepdims=True))


def test_preprocess_input_validation():
    with pytest.raises(ValueError):
        preprocess(np.zeros(100), FS, CFG)
    with pytest.raises(ValueError):
        preprocess(np.zeros((1, 4, 100)), FS, CFG)
    bad = np.zeros((4, 100))
    bad[2, 50] = np.inf
    with pytest.raises(ValueError):
        preprocess(bad, FS, CFG)
    bad[2, 50] = np.nan
    with pytest.raises(ValueError):
        preprocess(bad, FS, CFG)
    with pytest.raises(ValueError):
        preprocess(np.zeros((4, 100)), 80.0, CFG)  # 45 Hz edge above Nyquist 40 Hz
    with pytest.raises(ValueError):
        preprocess(np.zeros((4, 100)), 0.0, CFG)


def test_preprocess_int_input_becomes_float64():
    x = np.ones((2, 300), dtype=np.int32)
    y = preprocess(x, FS, CFG)
    assert y.dtype == np.float64
    np.testing.assert_allclose(y, 0.0, atol=1e-15)  # constant -> removed exactly


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def test_extract_window_sample_exact():
    x = np.tile(np.arange(1500.0), (4, 1))  # 6 s epoch
    w = extract_window(x, FS, 0.64, 2.5)  # benchmark: 0.5 s pre-stim + 0.14 s latency
    assert w.shape == (4, 625)
    assert w[0, 0] == 160 and w[0, -1] == 784
    w[0, 0] = -1
    assert x[0, 160] == 160  # copy, not view
    assert extract_window(x, FS, 0.0, 6.0).shape == (4, 1500)
    assert extract_window(x, FS, 3.5, 2.5)[0, -1] == 1499


@pytest.mark.parametrize(("start", "dur"), [(-0.004, 1.0), (3.6, 2.5), (0.0, 6.004), (1.0, 0.0)])
def test_extract_window_bounds(start, dur):
    x = np.zeros((4, 1500))
    with pytest.raises(ValueError):
        extract_window(x, FS, start, dur)


def test_last_window():
    x = np.tile(np.arange(1000.0), (4, 1))
    w = last_window(x, FS, 2.5)
    assert w.shape == (4, 625)
    assert w[0, 0] == 375 and w[0, -1] == 999
    assert last_window(x, FS, 4.0).shape == (4, 1000)
    with pytest.raises(ValueError):
        last_window(x, FS, 4.004)
    with pytest.raises(ValueError):
        last_window(x, FS, 0.0)
    with pytest.raises(ValueError):
        last_window(np.zeros(10), FS, 0.01)
