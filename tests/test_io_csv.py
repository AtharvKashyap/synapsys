"""Tests for the CSV trial format."""

from __future__ import annotations

import json

import numpy as np
import pytest

from synapsys_decoder.io.csv_eeg import load_csv_trial, save_csv_trial
from synapsys_decoder.types import Trial


def _trial() -> Trial:
    rng = np.random.default_rng(0)
    return Trial(
        eeg=rng.normal(scale=20e-6, size=(3, 50)),
        fs=250.0,
        channel_names=("O1", "Oz", "O2"),
        true_frequency=12.0,
        true_label="NO",
        subject_id="s",
        block_id="1",
        trial_id="t9",
        metadata={"note": "x"},
    )


@pytest.mark.parametrize("units", ["V", "mV", "uV"])
def test_round_trip(tmp_path, units):
    t = _trial()
    save_csv_trial(t, tmp_path / "a.csv", units=units)
    r = load_csv_trial(tmp_path / "a.csv")
    np.testing.assert_allclose(r.eeg, t.eeg, rtol=1e-9)
    assert r.eeg.dtype == np.float64 and r.eeg.shape == (3, 50)
    assert (r.fs, r.channel_names, r.true_frequency, r.true_label) == (
        250.0,
        t.channel_names,
        12.0,
        "NO",
    )
    assert (r.subject_id, r.block_id, r.trial_id) == ("s", "1", "t9")
    assert r.metadata["note"] == "x"


def test_unit_conversion(tmp_path):
    (tmp_path / "b.csv").write_text("O1,Oz\n1,2\n3,4\n")
    (tmp_path / "b.json").write_text(json.dumps({"fs": 100, "units": "uV"}))
    r = load_csv_trial(tmp_path / "b.csv")
    np.testing.assert_array_equal(r.eeg, np.array([[1e-6, 3e-6], [2e-6, 4e-6]]))
    r2 = load_csv_trial(tmp_path / "b.csv", units="mV")
    np.testing.assert_array_equal(r2.eeg, np.array([[1e-3, 3e-3], [2e-3, 4e-3]]))


def test_channel_selection(tmp_path):
    save_csv_trial(_trial(), tmp_path / "a.csv")
    r = load_csv_trial(tmp_path / "a.csv", channels=["O2", "O1"])
    t = _trial()
    assert r.channel_names == ("O2", "O1")
    np.testing.assert_allclose(r.eeg, t.eeg[[2, 0]], rtol=1e-9)
    with pytest.raises(KeyError):
        load_csv_trial(tmp_path / "a.csv", channels=["Pz"])


def test_no_sidecar_with_explicit_args(tmp_path):
    (tmp_path / "c.csv").write_text("time_s,O1\n0,1\n0.01,2\n")
    r = load_csv_trial(tmp_path / "c.csv", fs=100.0, units="V")
    assert r.channel_names == ("O1",) and r.fs == 100.0
    np.testing.assert_array_equal(r.eeg, [[1.0, 2.0]])


def test_missing_fs_or_units_raises(tmp_path):
    (tmp_path / "c.csv").write_text("O1\n1\n2\n")
    with pytest.raises(ValueError, match="fs"):
        load_csv_trial(tmp_path / "c.csv", units="V")
    with pytest.raises(ValueError, match="units"):
        load_csv_trial(tmp_path / "c.csv", fs=100.0)
    with pytest.raises(ValueError, match="units"):
        load_csv_trial(tmp_path / "c.csv", fs=100.0, units="kV")


def test_non_monotonic_time_raises(tmp_path):
    (tmp_path / "d.csv").write_text("time_s,O1\n0,1\n0.02,2\n0.01,3\n")
    with pytest.raises(ValueError, match="increasing"):
        load_csv_trial(tmp_path / "d.csv", fs=100.0, units="V")


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_csv_trial(tmp_path / "nope.csv")


def test_save_without_time(tmp_path):
    save_csv_trial(_trial(), tmp_path / "e.csv", include_time=False)
    assert (tmp_path / "e.csv").read_text().splitlines()[0] == "O1,Oz,O2"
