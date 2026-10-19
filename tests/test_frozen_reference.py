"""Freeze guard for the v1 Python reference that the ESP32 port must reproduce.

If any of these fail, the reference behaviour changed. That is allowed only as a
deliberate, documented version bump (docs/reference_v1.md): regenerate fixtures with
scripts/make_fixtures.py, re-run the C++ equivalence suite, and record why.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from synapsys_decoder.cca import DegenerateSignalError, canonical_correlations
from synapsys_decoder.classifier import decide
from synapsys_decoder.config import load_config
from synapsys_decoder.preprocessing import build_filter_sos, preprocess
from synapsys_decoder.references import harmonic_references

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "fixtures"
CFG_FILES = {"synapsys_v1": "configs/synapsys.yaml", "benchmark_v1": "configs/benchmark.yaml"}


def test_frozen_v1_settings():
    cfg = load_config(ROOT / "configs" / "synapsys.yaml")
    assert cfg.sample_rate_hz == 250
    assert cfg.channels == ("O1", "Oz", "O2", "POz")
    assert cfg.targets == pytest.approx({"YES": 60 / 7, "NO": 12.0, "HELP": 15.0}, abs=1e-9)
    assert list(cfg.targets) == ["YES", "NO", "HELP"]
    assert (cfg.window_seconds, cfg.window_samples, cfg.harmonics) == (2.5, 625, 2)
    pp = cfg.preprocessing
    assert (pp.remove_dc, pp.bandpass_low_hz, pp.bandpass_high_hz, pp.bandpass_order) == (
        True,
        0.5,
        45.0,
        4,
    )
    assert (pp.notch_hz, pp.notch_q, pp.mode) == (60.0, 30.0, "causal")
    assert cfg.decision.confidence_threshold is None
    assert cfg.decision.margin_threshold is None


@pytest.mark.parametrize("name", CFG_FILES)
def test_exported_config_matches_code(name):
    exported = json.loads((FIX / "config" / f"{name}.json").read_text())
    cfg = load_config(ROOT / CFG_FILES[name])
    sos = build_filter_sos(cfg.sample_rate_hz, cfg.preprocessing)
    np.testing.assert_allclose(np.array(exported["preprocessing"]["sos"]), sos, rtol=0, atol=0)
    assert exported["window_samples"] == cfg.window_samples
    assert [t["frequency_hz"] for t in exported["targets"]] == list(cfg.frequencies)


def _fixture_dirs():
    return sorted(p.parent for p in FIX.glob("*/*/expected.json"))


def _read(path):
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2).T


@pytest.mark.parametrize("fdir", _fixture_dirs(), ids=lambda p: p.name)
def test_fixture_reproduces(fdir):
    exp = json.loads((fdir / "expected.json").read_text())
    raw = _read(fdir / "raw.csv")
    for cname, res in exp["results"].items():
        cfg = load_config(ROOT / CFG_FILES[cname])
        filtered = preprocess(raw, cfg.sample_rate_hz, cfg.preprocessing)
        np.testing.assert_allclose(
            filtered, _read(fdir / res["filtered_file"]), rtol=1e-9, atol=1e-15
        )
        w = cfg.window_samples
        for pt in res["decision_points"]:
            win = filtered[:, pt["window_start_sample"] : pt["window_end_sample"]]
            if "error" in pt:
                with pytest.raises(DegenerateSignalError):
                    canonical_correlations(
                        win,
                        harmonic_references(
                            cfg.frequencies[0], cfg.sample_rate_hz, w, cfg.harmonics
                        ),
                    )
                continue
            for lab, f in cfg.targets.items():
                rho = canonical_correlations(
                    win, harmonic_references(f, cfg.sample_rate_hz, w, cfg.harmonics)
                )
                np.testing.assert_allclose(rho, pt["all_correlations"][lab], atol=1e-9)
            r = decide(pt["scores"], cfg.targets)
            assert (r.winner, r.decision) == (pt["winner"], pt["decision_ungated"])
            for g in pt["gated_test_only"]:
                assert decide(pt["scores"], cfg.targets, g["T"], g["M"]).decision == g["decision"]


def test_committed_fixture_files_match_manifest():
    manifest = json.loads((FIX / "MANIFEST.json").read_text())["files"]
    for rel, digest in manifest.items():
        p = FIX / rel
        if rel.startswith("wang/") and not p.exists():
            continue  # generated locally from data/benchmark (not redistributed)
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, rel
