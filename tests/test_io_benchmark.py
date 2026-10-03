"""Tests for the Wang 2017 Benchmark loader using a small fake dataset."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.io import savemat

from synapsys_decoder.io.benchmark import (
    benchmark_files_present,
    load_benchmark_subject,
    normalize_channel_name,
    read_channel_names,
    read_freq_phase,
)

RAW_NAMES = [
    "FP1",
    "FPZ",
    "FP2",
    "AF3",
    "AF4",
    "F7",
    "F5",
    "F3",
    "F1",
    "FZ",
    "F2",
    "F4",
    "F6",
    "F8",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCZ",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "T7",
    "C5",
    "C3",
    "C1",
    "CZ",
    "C2",
    "C4",
    "C6",
    "T8",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPZ",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "P7",
    "P5",
    "P3",
    "P1",
    "PZ",
    "P2",
    "P4",
    "P6",
    "P8",
    "PO7",
    "PO5",
    "PO3",
    "POZ",
    "PO4",
    "PO6",
    "PO8",
    "CB1",
    "O1",
    "OZ",
    "O2",
    "CB2",
    "M1",
    "M2",
]
N_SAMP = 300


@pytest.fixture
def fake_dir(tmp_path):
    assert len(RAW_NAMES) == 64
    rows = [
        f"{i + 1}\t{-18.0 * i}\t0.5\t{n}." if i % 2 else f"{i + 1} 0 0.5 {n} "
        for i, n in enumerate(RAW_NAMES)
    ]
    (tmp_path / "64-channels.loc").write_text("\n".join(rows) + "\n")
    # Non-sorted order (column-major 8x5 style).
    freqs = np.array([8.0 + 0.2 * k for k in range(40)]).reshape(5, 8).T.ravel()
    freqs = freqs + 1e-9  # floating noise
    phases = (np.arange(40) % 4) * 0.5 * np.pi
    savemat(
        tmp_path / "Freq_Phase.mat",
        {"freqs": freqs.reshape(1, 40), "phases": phases.reshape(1, 40)},
    )
    c, t, b = np.meshgrid(np.arange(64), np.arange(40), np.arange(2), indexing="ij")
    ident = c + 100 * t + 10000 * b
    data = (ident[:, None, :, :] + 0.25 * np.arange(N_SAMP)[None, :, None, None]).astype(np.float32)
    data = data.astype(np.float32)
    # data[c, s, t, b] shape (64, N_SAMP, 40, 2)
    savemat(tmp_path / "S1.mat", {"data": data})
    return tmp_path, data, freqs


@pytest.fixture
def patched(monkeypatch):
    import synapsys_decoder.io.benchmark as bm

    monkeypatch.setattr(bm, "N_SAMPLES", N_SAMP)
    monkeypatch.setattr(bm, "N_BLOCKS", 2)


def test_normalize():
    assert [
        normalize_channel_name(x)
        for x in ["OZ", "POZ", "FPZ", "FP1", "CZ", "CB1", "PO7", "FCZ", "Oz. "]
    ] == [
        "Oz",
        "POz",
        "Fpz",
        "Fp1",
        "Cz",
        "CB1",
        "PO7",
        "FCz",
        "Oz",
    ]


def test_channel_names_and_freq_phase(fake_dir):
    d, _, freqs = fake_dir
    names = read_channel_names(d / "64-channels.loc")
    assert len(names) == 64 and "Oz" in names and "POz" in names and "Fp1" in names
    f, p = read_freq_phase(d / "Freq_Phase.mat")
    assert f.shape == (40,) and p.shape == (40,)
    np.testing.assert_array_equal(f, freqs)
    assert not np.all(np.diff(f) > 0)


def test_channel_selection_case_insensitive_ordered(fake_dir, patched):
    d, data, _ = fake_dir
    trials = load_benchmark_subject(d, 1, channels=["oz", "O1", "POZ", "o2"], blocks=[0])
    assert len(trials) == 40
    t = trials[0]
    assert t.channel_names == ("Oz", "O1", "POz", "O2")
    idx = [RAW_NAMES.index(n) for n in ["OZ", "O1", "POZ", "O2"]]
    np.testing.assert_array_equal(t.eeg, data[idx, :, 0, 0].astype(np.float64) * 1e-6)
    assert t.metadata["channel_names_raw"] == ("OZ.", "O1", "POZ.", "O2")


def test_missing_channel_raises(fake_dir, patched):
    with pytest.raises(KeyError, match="Nope"):
        load_benchmark_subject(fake_dir[0], 1, channels=["Oz", "Nope"])


def test_no_filtering_exact_scaling_and_dtype(fake_dir, patched):
    d, data, _ = fake_dir
    trials = load_benchmark_subject(d, 1, channels=["Oz"], blocks=[1], frequencies=[8.2])
    (t,) = trials
    assert t.eeg.dtype == np.float64 and t.eeg.shape == (1, N_SAMP)
    ti = t.metadata["target_index"]
    c = RAW_NAMES.index("OZ")
    np.testing.assert_array_equal(t.eeg[0], data[c, :, ti, 1].astype(np.float64) * 1e-6)


def test_frequency_selection_uses_freq_phase_not_index(fake_dir, patched):
    d, _, freqs = fake_dir
    (t,) = load_benchmark_subject(d, 1, channels=["Oz"], blocks=[0], frequencies=[12.0])
    assert t.true_frequency == pytest.approx(12.0, abs=1e-6)
    assert t.true_frequency == freqs[t.metadata["target_index"]]
    assert t.metadata["target_index"] != round((12.0 - 8.0) / 0.2)
    assert t.trial_id == f"S1-b0-t{t.metadata['target_index']}"


def test_off_grid_frequency_raises(fake_dir, patched):
    with pytest.raises(ValueError, match=r"0\.2 Hz grid"):
        load_benchmark_subject(fake_dir[0], 1, frequencies=[60 / 7])


def test_block_selection_and_invalid(fake_dir, patched):
    d, _, _ = fake_dir
    ts = load_benchmark_subject(d, 1, channels=["Oz"], frequencies=[8.0], blocks=[1])
    assert [t.block_id for t in ts] == ["1"]
    with pytest.raises(ValueError, match="0-based"):
        load_benchmark_subject(d, 1, blocks=[7])


def test_metadata(fake_dir, patched):
    (t,) = load_benchmark_subject(fake_dir[0], 1, channels=["Oz"], blocks=[0], frequencies=[8.0])
    m = t.metadata
    assert t.fs == 250.0 and t.true_label is None and t.subject_id == "S1" and t.block_id == "0"
    assert m["stimulus_onset_sample"] == 125 and m["stimulus_onset_s"] == 0.5
    assert m["stimulation_duration_s"] == 5.0 and m["units_source"] == "uV"
    assert "notch" in m["upstream_processing"]
    assert m["phase_rad"] == pytest.approx((m["target_index"] % 4) * 0.5 * np.pi)
    assert "S1.mat" in m["source_file"]


def test_missing_files(tmp_path):
    assert not benchmark_files_present(tmp_path)
    with pytest.raises(FileNotFoundError, match=r"data/README\.md"):
        load_benchmark_subject(tmp_path, 1)
    with pytest.raises(ValueError, match="subject"):
        load_benchmark_subject(tmp_path, 99)


def test_wrong_data_shape_raises(fake_dir):
    d, _, _ = fake_dir
    savemat(d / "S2.mat", {"data": np.zeros((64, 10, 40, 6), dtype=np.float32)})
    with pytest.raises(ValueError, match="shape"):
        load_benchmark_subject(d, 2)


def test_files_present(fake_dir):
    assert benchmark_files_present(fake_dir[0])


@pytest.mark.skipif(not benchmark_files_present("data/benchmark"), reason="real data not present")
def test_real_s1():
    trials = load_benchmark_subject(
        "data/benchmark", 1, channels=["O1", "Oz", "O2", "POz"], blocks=[0]
    )
    assert len(trials) == 40
    t = trials[0]
    assert t.eeg.shape == (4, 1500) and t.fs == 250.0
    assert t.metadata["stimulus_onset_sample"] == 125
