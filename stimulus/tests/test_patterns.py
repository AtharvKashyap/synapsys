import pytest

from synapsys_stimulus.patterns import sinusoidal_luminance


def test_luminance_remains_normalized() -> None:
    for frame in range(100):
        value = sinusoidal_luminance(frame, period_frames=9)
        assert 0.0 <= value <= 1.0


def test_pattern_repeats_after_one_period() -> None:
    period = 7
    for frame in range(period):
        assert sinusoidal_luminance(
            frame,
            period,
        ) == pytest.approx(sinusoidal_luminance(frame + period, period))


def test_invalid_period_raises_error() -> None:
    with pytest.raises(ValueError):
        sinusoidal_luminance(frame=0, period_frames=0)


def test_negative_frame_raises_error() -> None:
    with pytest.raises(ValueError):
        sinusoidal_luminance(frame=-1, period_frames=9)
