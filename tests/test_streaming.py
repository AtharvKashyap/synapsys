"""decode_stream reproduces the device schedule: checked against every committed fixture."""

import json
from pathlib import Path

import numpy as np
import pytest

from synapsys_decoder.classifier import decide
from synapsys_decoder.config import load_config
from synapsys_decoder.io.csv_eeg import load_csv_trial
from synapsys_decoder.streaming import decision_ends, decode_stream
from synapsys_decoder.types import REJECT, Trial

ROOT = Path(__file__).resolve().parents[1]
SYN = sorted((ROOT / "fixtures" / "synthetic").iterdir())
CFG = load_config(ROOT / "configs" / "synapsys.yaml")


@pytest.mark.parametrize("fix", SYN, ids=lambda p: p.name)
def test_decode_stream_matches_fixture(fix):
    exp = json.loads((fix / "expected.json").read_text())
    trial = load_csv_trial(fix / "raw.csv", fs=250, units="V")
    ss = decode_stream(trial, CFG, hop_samples=125)
    points = exp["results"]["synapsys_v1"]["decision_points"]
    assert [p["window_end_sample"] for p in points] == ss.window_end.tolist()
    for i, p in enumerate(points):
        if "error" in p:
            assert ss.degenerate[i]
            assert ss.decisions(0.2, 0.02)[i] == REJECT
            continue
        np.testing.assert_allclose(ss.scores[i], [p["scores"][k] for k in CFG.labels], atol=1e-12)
        assert ss.labels[ss.winner_index[i]] == p["winner"]
        assert ss.margin[i] == pytest.approx(p["margin"], abs=1e-12)
        assert ss.decisions()[i] == p["decision_ungated"]
        for g in p["gated_test_only"]:
            assert ss.decisions(g["T"], g["M"])[i] == g["decision"]


def test_gating_equals_classifier_decide():
    rng = np.random.default_rng(3)
    scores = rng.uniform(0, 1, (400, 3))
    scores[::7, 1] = scores[::7, 0]  # exact ties -> first label
    from synapsys_decoder.streaming import StreamScores

    ss = StreamScores(CFG.labels, np.arange(400) * 125 + 625, scores, 250.0)
    for t, m in [(None, None), (0.5, None), (None, 0.1), (0.4, 0.05)]:
        got = ss.decisions(t, m)
        for i in range(400):
            r = decide(dict(zip(CFG.labels, scores[i], strict=True)), CFG.targets, t, m)
            assert got[i] == r.decision


def test_decision_ends_and_short_stream():
    assert decision_ends(624, 625, 125).size == 0
    assert decision_ends(625, 625, 125).tolist() == [625]
    assert decision_ends(1000, 625, 125).tolist() == [625, 750, 875, 1000]
    short = Trial(eeg=np.zeros((4, 100)), fs=250.0, channel_names=CFG.channels)
    assert decode_stream(short, CFG).window_end.size == 0


def test_decode_stream_refuses_zero_phase_and_wrong_fs():
    trial = Trial(eeg=np.zeros((4, 700)), fs=500.0, channel_names=CFG.channels)
    with pytest.raises(ValueError, match="fs"):
        decode_stream(trial, CFG)
    zp = CFG.replace(
        preprocessing=CFG.preprocessing.__class__(
            **{**CFG.preprocessing.__dict__, "mode": "zero_phase"}
        )
    )
    with pytest.raises(ValueError, match="causal"):
        decode_stream(Trial(eeg=np.zeros((4, 700)), fs=250.0, channel_names=CFG.channels), zp)
