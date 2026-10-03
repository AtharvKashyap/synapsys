from pathlib import Path

import pytest

from synapsys_decoder.config import DecoderConfig, PreprocessingConfig, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_project_config_loads():
    cfg = load_config(CONFIGS / "synapsys.yaml")
    assert cfg.sample_rate_hz == 250
    assert cfg.channels == ("O1", "Oz", "O2", "POz")
    assert cfg.labels == ("YES", "NO", "HELP")
    assert cfg.targets["YES"] == pytest.approx(60 / 7)
    assert cfg.window_samples == 625
    assert cfg.decision.confidence_threshold is None
    assert cfg.decision.margin_threshold is None


def test_benchmark_config_keeps_extra_section():
    cfg = load_config(CONFIGS / "benchmark.yaml")
    assert cfg.extra["benchmark"]["stimulus_onset_s"] == 0.5
    assert cfg.preprocessing.notch_hz is None
    assert cfg.to_dict()["benchmark"]["subjects"] == [1]


def test_roundtrip_dict():
    cfg = load_config(CONFIGS / "synapsys.yaml")
    assert DecoderConfig.from_dict(cfg.to_dict()) == cfg


def test_rejects_harmonic_above_nyquist():
    with pytest.raises(ValueError, match="Nyquist"):
        DecoderConfig(
            sample_rate_hz=250,
            channels=("Oz",),
            targets={"A": 40.0, "B": 50.0},
            window_seconds=2.0,
            harmonics=4,
        )


@pytest.mark.parametrize(
    "kwargs",
    [{"mode": "acausal"}, {"bandpass_low_hz": 50.0, "bandpass_high_hz": 45.0}, {"notch_q": 0}],
)
def test_preprocessing_validation(kwargs):
    with pytest.raises(ValueError):
        PreprocessingConfig(**kwargs)


def test_unquoted_yes_no_labels_rejected():
    raw = {
        "sample_rate_hz": 250,
        "channels": ["Oz"],
        "targets": {True: 8.0, False: 12.0},
        "window_seconds": 2.0,
        "harmonics": 2,
    }
    with pytest.raises(ValueError, match="must be strings"):
        DecoderConfig.from_dict(raw)
