"""Smoke test of scripts/run_benchmark_validation.py on a small fake Benchmark dataset."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[1]
NAMES = [f"X{i}" for i in range(64)]
OCC = ["PZ", "PO5", "PO3", "POZ", "PO4", "PO6", "O1", "OZ", "O2"]
NAMES[-len(OCC) :] = OCC
N_SAMP, N_BLOCKS = 1000, 2  # 4 s epochs: room for onset 0.64 s + 3 s window


@pytest.fixture
def fake_benchmark(tmp_path, monkeypatch):
    import synapsys_decoder.io.benchmark as bm

    monkeypatch.setattr(bm, "N_SAMPLES", N_SAMP)
    monkeypatch.setattr(bm, "N_BLOCKS", N_BLOCKS)
    (tmp_path / "64-channels.loc").write_text(
        "".join(f"{i + 1}\t0\t0.5\t{n}\n" for i, n in enumerate(NAMES))
    )
    freqs = np.array([8.0 + 0.2 * k for k in range(40)]).reshape(5, 8).T.ravel()
    savemat(tmp_path / "Freq_Phase.mat", {"freqs": freqs[None], "phases": np.zeros((1, 40))})
    rng = np.random.default_rng(0)
    t = np.arange(N_SAMP) / 250.0
    data = rng.normal(0, 5.0, (64, N_SAMP, 40, N_BLOCKS)).astype(np.float32)  # uV
    stim = t >= 0.5
    for k, f in enumerate(freqs):
        for c in range(64 - len(OCC), 64):
            phase = rng.uniform(0, 2 * np.pi)
            data[c, :, k, :] += (3.0 * np.sin(2 * np.pi * f * t + phase) * stim)[:, None]
    savemat(tmp_path / "S1.mat", {"data": data})
    return tmp_path


def test_benchmark_script_runs_end_to_end(fake_benchmark, tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import run_benchmark_validation as script
    finally:
        sys.path.pop(0)
    out = tmp_path / "results"
    rc = script.main(
        ["--data-dir", str(fake_benchmark), "--subjects", "1", "--results-root", str(out)]
    )
    assert rc == 0
    (run_dir,) = out.iterdir()
    meta = json.loads((run_dir / "run.json").read_text())["metrics"]
    assert meta["n_trials"] == 3 * N_BLOCKS
    assert meta["experiment1_smoke"]["causal"]["accuracy"] == 1.0  # strong fake SSVEP
    assert len(meta["experiment2_window_sweep"]["accuracy"]["causal"]) == 5
    assert "disclaimer" in meta
    for name in [
        "smoke_confusion",
        "accuracy_vs_window",
        "accuracy_channels",
        "accuracy_harmonics",
        "psd_example",
        "smoke_score_distributions",
    ]:
        assert (run_dir / f"{name}.png").exists()


def test_missing_data_exits_with_hint(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import run_benchmark_validation as script
    finally:
        sys.path.pop(0)
    with pytest.raises(SystemExit, match=r"data/README\.md"):
        script.main(["--data-dir", str(tmp_path), "--results-root", str(tmp_path / "r")])


def test_walkthrough_runs_all_stages_on_fake_data(fake_benchmark, tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import benchmark_walkthrough as wt
    finally:
        sys.path.pop(0)
    out = tmp_path / "walk"
    rc = wt.main(["--data-dir", str(fake_benchmark), "--subjects", "1", "--out", str(out)])
    status = json.loads((out / "status.json").read_text())
    assert status["finished"] is True
    assert [s["status"] for s in status["stages"]] == ["pass"] * 7, [
        (s["id"], [c for c in s["checks"] if c["ok"] is False]) for s in status["stages"]
    ]
    assert rc == 0
    assert (out / "index.html").exists()
    for s in status["stages"]:
        for f in s["figures"]:
            assert (out / f["src"]).exists()
