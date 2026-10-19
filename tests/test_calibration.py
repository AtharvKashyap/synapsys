"""Threshold calibration: guardrails, metrics and the fixed selection rule.

Synthetic data only exercises the code; it never provides thresholds for people.
"""

import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from synapsys_decoder.calibration import (
    REST_LABEL,
    check_calibration_set,
    evaluate_thresholds,
    prepare_recording,
    select_thresholds,
    sweep,
    threshold_grid,
)
from synapsys_decoder.config import load_config
from synapsys_decoder.io.csv_eeg import save_csv_trial
from synapsys_decoder.synthetic import make_noise_trial, make_synthetic_trial

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs" / "synapsys.yaml")


def _target(label, seed, seconds=12.0, snr=-12.0):
    rng = np.random.default_rng(seed)
    t = make_synthetic_trial(
        CFG.targets[label], 250.0, seconds, CFG.channels, rng, snr_db=snr, label=label
    )
    return dataclasses.replace(t, metadata={"split": "calibration"})


def _rest(seed, seconds=60.0):
    t = make_noise_trial(250.0, seconds, CFG.channels, np.random.default_rng(seed))
    return dataclasses.replace(t, true_label=REST_LABEL, metadata={"split": "calibration"})


@pytest.fixture(scope="module")
def recs():
    out = [
        prepare_recording(_target(lab, 10 + i * 3 + k), CFG, name=f"{lab}{k}")
        for i, lab in enumerate(CFG.labels)
        for k in range(3)
    ]
    out += [prepare_recording(_rest(100 + k), CFG, name=f"rest{k}") for k in range(2)]
    return out


def test_guardrails():
    with pytest.raises(ValueError, match="test data"):
        prepare_recording(
            dataclasses.replace(_target("NO", 1), metadata={"split": "test"}), CFG, name="x"
        )
    with pytest.raises(ValueError, match="true_label"):
        prepare_recording(dataclasses.replace(_target("NO", 1), true_label=None), CFG, name="x")
    with pytest.raises(ValueError, match="not a target"):
        prepare_recording(dataclasses.replace(_target("NO", 1), true_label="MAYBE"), CFG, name="x")
    with pytest.raises(ValueError, match="record at least"):
        prepare_recording(_target("NO", 1, seconds=5.0), CFG, name="x")


def test_settle_drops_early_points():
    r = prepare_recording(_target("NO", 1), CFG, name="x", settle_seconds=3.0)
    assert (r.scores.window_end - CFG.window_samples).min() >= 750
    r0 = prepare_recording(_target("NO", 1), CFG, name="x", settle_seconds=0.0)
    assert r0.scores.window_end[0] == CFG.window_samples


def test_needs_rest_and_targets(recs):
    with pytest.raises(ValueError, match="REST"):
        check_calibration_set([r for r in recs if not r.is_rest])
    with pytest.raises(ValueError, match="target"):
        check_calibration_set([r for r in recs if r.is_rest])


def test_gate_off_vs_strict(recs):
    off = evaluate_thresholds(recs, None, None)
    # Without a gate every rest point is accepted, so rest produces commands.
    assert off["rest_point_accepted"] == 1.0 and off["rest_commands"] > 0
    assert off["first_command_correct"] == 1.0  # clean-ish synthetic targets
    strict = evaluate_thresholds(recs, 1.0, None)  # nothing reaches 1.0
    assert strict["rest_commands"] == 0 and strict["no_command"] == 1.0
    assert strict["point_rejected"] == 1.0
    zero = evaluate_thresholds(recs, 0.0, 0.0)  # T=M=0 behaves like gate off
    assert {k: v for k, v in zero.items() if "threshold" not in k} == {
        k: v for k, v in off.items() if "threshold" not in k
    }


def test_selection_on_sweep(recs):
    grid = threshold_grid(np.arange(0, 1.0001, 0.05), np.arange(0, 0.2001, 0.05))
    rows = sweep(recs, grid)
    best = select_thresholds(rows, max_rest_commands_per_min=0.0)
    assert best is not None
    assert best["rest_commands"] == 0 and best["first_command_correct"] == 1.0
    # Ties broken toward the more conservative gate: no feasible equal row has larger T.
    same = [
        r
        for r in rows
        if r["rest_commands"] == 0
        and r["first_command_correct"] == 1.0
        and r["wrong_commands"] == best["wrong_commands"]
    ]
    assert best["confidence_threshold"] == max(r["confidence_threshold"] for r in same)


def test_select_rule_order():
    base = {
        "rest_commands_per_min": 0.0,
        "first_command_wrong": 0.0,
        "rest_commands": 0,
        "wrong_commands": 0,
        "first_command_correct": 0.8,
        "confidence_threshold": 0.3,
        "margin_threshold": 0.0,
    }
    rows = [
        {**base, "first_command_correct": 0.9, "rest_commands_per_min": 2.0},  # infeasible
        {**base, "first_command_correct": 0.8, "wrong_commands": 2},
        {**base, "first_command_correct": 0.8, "wrong_commands": 1, "confidence_threshold": 0.2},
        {**base, "first_command_correct": 0.8, "wrong_commands": 1, "confidence_threshold": 0.4},
    ]
    best = select_thresholds(rows, max_rest_commands_per_min=1.0)
    assert best["wrong_commands"] == 1 and best["confidence_threshold"] == 0.4
    assert select_thresholds(rows, max_rest_commands_per_min=1.0, max_wrong_fraction=-1) is None
    with pytest.raises(ValueError):
        select_thresholds(rows, max_rest_commands_per_min=float("nan"))


def test_calibrate_script_end_to_end(tmp_path):
    for i, lab in enumerate(CFG.labels):
        save_csv_trial(_target(lab, 50 + i), tmp_path / "cal" / f"{lab.lower()}.csv")
    save_csv_trial(_rest(60, seconds=40.0), tmp_path / "cal" / "rest.csv")
    spec = importlib.util.spec_from_file_location(
        "cal", ROOT / "scripts" / "calibrate_thresholds.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rc = mod.main(
        [
            str(tmp_path / "cal"),
            "--max-rest-commands-per-min",
            "0",
            "--results-root",
            str(tmp_path / "res"),
        ]
    )
    assert rc == 0
    (run,) = (tmp_path / "res").iterdir()
    sel = json.loads((run / "selection.json").read_text())
    assert sel["chosen"]["rest_commands"] == 0
    for f in ("sweep.csv", "recordings.csv", "threshold_map.png", "apply.txt", "run.json"):
        assert (run / f).is_file()
    assert "CONFIG_SYNAPSYS_CONF_THRESHOLD" in (run / "apply.txt").read_text()

    save_csv_trial(
        dataclasses.replace(_target("NO", 7), metadata={"split": "test"}),
        tmp_path / "cal" / "leak.csv",
    )
    with pytest.raises(ValueError, match="test data"):
        mod.main(
            [
                str(tmp_path / "cal"),
                "--max-rest-commands-per-min",
                "0",
                "--results-root",
                str(tmp_path / "res"),
            ]
        )


def test_evaluation_purpose_split_guard():
    cal = _target("NO", 1)
    with pytest.raises(ValueError, match="needs 'test'"):
        prepare_recording(cal, CFG, name="x", purpose="evaluation")
    test = dataclasses.replace(cal, metadata={"split": "test"})
    assert prepare_recording(test, CFG, name="x", purpose="evaluation").true_label == "NO"
    with pytest.raises(ValueError, match="purpose"):
        prepare_recording(cal, CFG, name="x", purpose="tuning")


def test_evaluate_heldout_script(tmp_path):
    for i, lab in enumerate(CFG.labels):
        t = dataclasses.replace(_target(lab, 70 + i), metadata={"split": "test"})
        save_csv_trial(t, tmp_path / "test" / f"{lab.lower()}.csv")
    rest = dataclasses.replace(_rest(80, seconds=30.0), metadata={"split": "test"})
    save_csv_trial(rest, tmp_path / "test" / "rest.csv")
    spec = importlib.util.spec_from_file_location("ev", ROOT / "scripts" / "evaluate_heldout.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = [str(tmp_path / "test"), "--results-root", str(tmp_path / "res")]
    with pytest.raises(SystemExit, match="null"):
        mod.main(args)  # frozen config has no thresholds yet
    assert mod.main([*args, "--allow-ungated"]) == 0
    (run,) = (tmp_path / "res").iterdir()
    out = json.loads((run / "held_out.json").read_text())
    assert out["accuracy_all_trials"] == 1.0 and out["thresholds"]["confidence"] is None
