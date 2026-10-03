import json
import sys
from pathlib import Path

import numpy as np
import pytest

from synapsys_decoder.config import load_config
from synapsys_decoder.pipeline import decode_trial
from synapsys_decoder.runlog import create_run_dir, save_run
from synapsys_decoder.synthetic import make_synthetic_trial

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs" / "synapsys.yaml")


def _trial(label, rng, duration=6.0, channels=("POz", "O2", "Oz", "O1", "Pz"), **kw):
    return make_synthetic_trial(
        CFG.targets[label], CFG.sample_rate_hz, duration, channels, rng, label=label, **kw
    )


@pytest.mark.parametrize("mode", ["causal", "zero_phase"])
@pytest.mark.parametrize("label", ["YES", "NO", "HELP"])
def test_decode_noisy_trial_all_classes(label, mode):
    rng = np.random.default_rng(7)
    cfg = CFG.replace(preprocessing=CFG.preprocessing.__class__(mode=mode))
    t = _trial(label, rng, snr_db=-5, drift_amplitude_v=20e-6, line_amplitude_v=5e-6)
    d = decode_trial(t, cfg)
    assert d.result.decision == label
    assert d.window.shape == (len(CFG.channels), CFG.window_samples)
    assert not d.result.gated


def test_selects_config_channels_by_name_in_config_order():
    rng = np.random.default_rng(1)
    t = _trial("NO", rng)
    d = decode_trial(t, CFG)
    assert d.filtered.shape[0] == len(CFG.channels)
    # Oz is row 2 in the trial but row 1 in config order; filtering must follow config order
    from synapsys_decoder.preprocessing import preprocess

    expected = preprocess(t.select_channels(CFG.channels).eeg, t.fs, CFG.preprocessing)
    np.testing.assert_array_equal(d.filtered, expected)


def test_default_window_is_last_window_and_explicit_start_works():
    rng = np.random.default_rng(2)
    t = _trial("HELP", rng, duration=5.0)
    d = decode_trial(t, CFG)
    assert d.window_start_s == pytest.approx(5.0 - CFG.window_seconds)
    np.testing.assert_array_equal(d.window, d.filtered[:, -CFG.window_samples :])
    d0 = decode_trial(t, CFG, window_start_s=0.0)
    np.testing.assert_array_equal(d0.window, d0.filtered[:, : CFG.window_samples])


def test_fs_mismatch_and_missing_channel_raise():
    rng = np.random.default_rng(3)
    t = make_synthetic_trial(12.0, 500.0, 4.0, CFG.channels, rng)
    with pytest.raises(ValueError, match="fs"):
        decode_trial(t, CFG)
    t = make_synthetic_trial(12.0, 250.0, 4.0, ("O1", "Oz", "O2"), rng)
    with pytest.raises(KeyError):
        decode_trial(t, CFG)


def test_gating_flows_through_from_config():
    rng = np.random.default_rng(4)
    cfg = CFG.replace(decision=CFG.decision.__class__(confidence_threshold=0.99))
    d = decode_trial(_trial("NO", rng, snr_db=-15), cfg)
    assert d.result.gated and not d.result.accepted and d.result.decision == "REJECT"
    assert d.result.winner in CFG.labels


def test_save_run_writes_reproducibility_artifacts(tmp_path):
    run_dir = create_run_dir(tmp_path, "unit")
    save_run(
        run_dir,
        config=CFG.to_dict(),
        metrics={"accuracy": np.float64(0.5), "arr": np.arange(3)},
        trials=[{"a": 1, "b": 2.0}, {"a": 3, "c": "x"}],
        command=["unit-test"],
    )
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["metrics"] == {"accuracy": 0.5, "arr": [0, 1, 2]}
    assert meta["command"] == ["unit-test"]
    assert "numpy" in meta["environment"]
    assert load_config(run_dir / "config.yaml") == CFG
    assert (run_dir / "trials.csv").read_text().splitlines()[0] == "a,b,c"


def test_synthetic_validation_script_smoke(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import run_synthetic_validation as script
    finally:
        sys.path.pop(0)
    assert script.main(["--trials-per-class", "2", "--results-root", str(tmp_path)]) == 0
    (run_dir,) = tmp_path.iterdir()
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["metrics"]["n_nonfinite_scores"] == 0
    assert meta["metrics"]["clean"]["accuracy"] == 1.0
    assert {p.name for p in run_dir.glob("*.png")} >= {
        "clean_cca_scores.png",
        "accuracy_vs_snr.png",
        "psd_before_after.png",
        "gating_scatter.png",
        "score_distributions.png",
    }


def test_analyze_recording_script_smoke(tmp_path):
    from synapsys_decoder.io.csv_eeg import save_csv_trial

    rng = np.random.default_rng(5)
    t = make_synthetic_trial(12.0, 250.0, 6.0, CFG.channels, rng, snr_db=-5, label="NO")
    csv = tmp_path / "rec.csv"
    save_csv_trial(t, csv)
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import analyze_recording as script
    finally:
        sys.path.pop(0)
    out = tmp_path / "results"
    assert script.main([str(csv), "--results-root", str(out)]) == 0
    (run_dir,) = out.iterdir()
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["metrics"]["n_windows"] == 8  # (6.0 - 2.5) / 0.5 + 1
    last = (run_dir / "trials.csv").read_text().splitlines()[-1]
    assert ",NO," in last
