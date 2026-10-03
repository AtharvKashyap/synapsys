from pathlib import Path

import numpy as np
import pytest

from synapsys_decoder.classifier import (
    classify_window,
    classify_window_with_config,
    decide,
    score_targets,
)
from synapsys_decoder.config import DecisionConfig, load_config
from synapsys_decoder.types import REJECT

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
FS = 250.0
TARGETS = {"YES": 60 / 7, "NO": 12.0, "HELP": 15.0}


def _ssvep(f, seconds=2.5, n_channels=4, noise=0.5, seed=0, amp=1e-5):
    """Channel-specific weighted sin/cos mixtures at f and 2f, plus white noise (volts)."""
    rng = np.random.default_rng(seed)
    t = np.arange(round(seconds * FS)) / FS
    eeg = np.zeros((n_channels, t.size))
    for c in range(n_channels):
        for h, a in ((1, 1.0), (2, 0.4)):
            w_sin, w_cos = rng.uniform(-1, 1, size=2) * a * amp
            eeg[c] += w_sin * np.sin(2 * np.pi * h * f * t) + w_cos * np.cos(2 * np.pi * h * f * t)
    return eeg + noise * amp * rng.standard_normal(eeg.shape)


@pytest.mark.parametrize("seconds", [2.0, 2.5, 3.0])
@pytest.mark.parametrize("label", list(TARGETS))
def test_clean_synthetic_classifies(label, seconds):
    eeg = _ssvep(TARGETS[label], seconds=seconds, seed=list(TARGETS).index(label))
    res = classify_window(eeg, FS, TARGETS, 2)
    assert res.winner == label
    assert res.decision == label
    assert res.winner_frequency == TARGETS[label]
    for other, s in res.scores.items():
        if other != label:
            assert res.scores[label] > s
    assert res.gated is False and res.accepted is True


def test_score_order_matches_mapping_order():
    eeg = _ssvep(12.0, seed=1)
    for order in (["YES", "NO", "HELP"], ["HELP", "YES", "NO"]):
        targets = {k: TARGETS[k] for k in order}
        assert list(score_targets(eeg, FS, targets, 2)) == order
        assert list(classify_window(eeg, FS, targets, 2).scores) == order


def test_margin_and_confidence():
    res = decide({"YES": 0.3, "NO": 0.9, "HELP": 0.5}, TARGETS)
    assert res.winner == "NO"
    assert res.confidence == pytest.approx(0.9)
    assert res.margin == pytest.approx(0.4)


def test_single_class_margin_equals_confidence():
    res = decide({"A": 0.7}, {"A": 10.0})
    assert res.margin == pytest.approx(0.7)


def test_ungated():
    res = decide({"YES": 0.1, "NO": 0.11, "HELP": 0.1}, TARGETS)
    assert res.gated is False
    assert res.accepted is True
    assert res.decision == "NO"


SCORES = {"YES": 0.2, "NO": 0.6, "HELP": 0.5}  # winner NO, margin 0.1


@pytest.mark.parametrize(
    ("conf", "margin", "accepted"),
    [
        (0.5, None, True),
        (0.7, None, False),
        (None, 0.05, True),
        (None, 0.2, False),
        (0.5, 0.05, True),
        (0.7, 0.05, False),
        (0.5, 0.2, False),
        (0.7, 0.2, False),
        (0.6, 0.0, True),  # thresholds are inclusive (>=)
    ],
)
def test_gating(conf, margin, accepted):
    res = decide(SCORES, TARGETS, confidence_threshold=conf, margin_threshold=margin)
    assert res.gated is True
    assert res.accepted is accepted
    assert res.winner == "NO"  # winner reported even when rejected
    assert res.decision == ("NO" if accepted else REJECT)


def test_rejected_window_end_to_end():
    eeg = _ssvep(15.0, seed=2)
    res = classify_window(eeg, FS, TARGETS, 2, confidence_threshold=1.0)
    assert res.winner == "HELP"
    assert res.decision == REJECT
    assert res.gated and not res.accepted


def test_tie_break_first_in_class_order():
    tie = {"YES": 0.5, "NO": 0.5, "HELP": 0.5}
    assert decide(tie, TARGETS).winner == "YES"
    reordered = {"HELP": 15.0, "NO": 12.0, "YES": 60 / 7}
    assert decide(tie, reordered).winner == "HELP"
    assert decide({"YES": 0.1, "NO": 0.5, "HELP": 0.5}, TARGETS).winner == "NO"
    assert decide(tie, TARGETS).margin == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confidence_threshold": -0.1},
        {"confidence_threshold": 1.1},
        {"margin_threshold": -0.01},
        {"confidence_threshold": float("nan")},
    ],
)
def test_invalid_thresholds(kwargs):
    with pytest.raises(ValueError):
        decide(SCORES, TARGETS, **kwargs)


def test_decide_label_mismatch_and_nan():
    with pytest.raises(ValueError):
        decide({"YES": 0.1, "NO": 0.2}, TARGETS)
    with pytest.raises(ValueError):
        decide({"YES": 0.1, "NO": float("nan"), "HELP": 0.2}, TARGETS)


def test_classify_window_with_config():
    cfg = load_config(CONFIGS / "synapsys.yaml")
    eeg = _ssvep(cfg.targets["NO"], seconds=cfg.window_seconds, seed=3)
    assert eeg.shape == (len(cfg.channels), cfg.window_samples)
    res = classify_window_with_config(eeg, cfg)
    assert res.decision == "NO"
    assert list(res.scores) == list(cfg.labels)
    assert res.gated is False


def test_classify_window_with_config_uses_thresholds():
    cfg = load_config(CONFIGS / "synapsys.yaml")
    cfg = cfg.replace(decision=DecisionConfig(confidence_threshold=1.0))
    res = classify_window_with_config(_ssvep(cfg.targets["YES"], seed=4), cfg)
    assert res.winner == "YES" and res.decision == REJECT and res.gated


def test_channel_count_mismatch_raises():
    cfg = load_config(CONFIGS / "synapsys.yaml")
    with pytest.raises(ValueError, match="channels"):
        classify_window_with_config(_ssvep(12.0, n_channels=3), cfg)


def test_bad_eeg_shape_raises():
    with pytest.raises(ValueError):
        score_targets(np.zeros(625), FS, TARGETS, 2)
    with pytest.raises(ValueError):
        score_targets(_ssvep(12.0), FS, {}, 2)
