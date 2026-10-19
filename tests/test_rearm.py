"""Tests for the command re-arming state machine (hand-written sequences)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from synapsys_decoder.rearm import RearmConfig, RearmStateMachine
from synapsys_decoder.types import REJECT

Y, N_, H, X = "YES", "NO", "HELP", REJECT


def run(seq, cfg=None, t0=625, hop=125):
    sm = RearmStateMachine(cfg or RearmConfig())
    return [sm.update(d, t0 + hop * i) for i, d in enumerate(seq)]


def test_defaults():
    c = RearmConfig()
    assert (c.confirm_count, c.refractory_samples, c.release_count) == (2, 625, 2)


def test_k2_basic_emit():
    assert run([Y, Y]) == [None, "YES"]
    assert run([Y, N_, Y]) == [None, None, None]


def test_no_repeat_while_staring():
    out = run([Y] * 40)
    assert out.count("YES") == 1 and out[1] == "YES"


def test_repeat_after_release_via_reject():
    # emit at i=1; X,X release (R=625 = 5 hops, reached at i=6)
    seq = [Y, Y, X, X, X, X, X, Y, Y]
    out = run(seq)
    assert out == [None, "YES", None, None, None, None, None, None, "YES"]


def test_repeat_after_release_via_other_label():
    seq = [Y, Y, N_, H, N_, N_, N_, Y, Y]
    out = run(seq, RearmConfig(refractory_samples=250))
    # t - emitted: i=3 -> 250, released=2 -> ARMED at i=3; i=4 NO starts cand,
    # i=5 NO emits (refractory ended) -> then Y,Y blocked until released again
    assert out[1] == "YES"
    assert out[5] == "NO"
    assert out.count("YES") == 1


def test_repeat_after_R_elapsed_same_label_with_gap():
    cfg = RearmConfig(refractory_samples=250, release_count=2)
    out = run([Y, Y, X, X, Y, Y], cfg)
    assert out == [None, "YES", None, None, None, "YES"]


def test_release_before_R_waits_for_R():
    cfg = RearmConfig(refractory_samples=625, release_count=2)
    sm = RearmStateMachine(cfg)
    sm.update(Y, 0)
    assert sm.update(Y, 125) == "YES"  # emitted_at=125
    sm.update(X, 250)
    sm.update(X, 375)  # released=2 but only 250 samples elapsed
    assert sm.state == "REFRACTORY"
    sm.update(X, 700)  # 575 < 625
    assert sm.state == "REFRACTORY"
    assert sm.update(X, 750) is None  # 625 elapsed -> completes re-arm; no candidate
    assert sm.state == "ARMED"
    assert sm.update(Y, 875) is None and sm.state == "CANDIDATE"
    assert sm.update(Y, 1000) == "YES"


def test_release_completing_point_does_not_start_candidate():
    cfg = RearmConfig(refractory_samples=0, release_count=1)
    sm = RearmStateMachine(cfg)
    sm.update(Y, 0)
    assert sm.update(Y, 125) == "YES"
    assert sm.update(N_, 250) is None
    assert sm.state == "ARMED"  # NO was consumed by the release


def test_staring_resets_release_counter():
    cfg = RearmConfig(refractory_samples=0, release_count=2)
    sm = RearmStateMachine(cfg)
    sm.update(Y, 0)
    sm.update(Y, 125)
    sm.update(X, 250)
    sm.update(Y, 375)  # back to L resets
    sm.update(X, 500)
    assert sm.state == "REFRACTORY"
    sm.update(X, 625)
    assert sm.state == "ARMED"


def test_candidate_interrupted_by_reject():
    sm = RearmStateMachine()
    assert sm.update(Y, 0) is None and sm.state == "CANDIDATE"
    assert sm.update(X, 125) is None and sm.state == "ARMED"
    assert run([Y, X, Y]) == [None, None, None]


def test_candidate_switching_label():
    assert run([Y, N_, N_]) == [None, None, "NO"]
    assert run([Y, N_, H, H]) == [None, None, None, "HELP"]


def test_k1_emits_immediately():
    cfg = RearmConfig(confirm_count=1, refractory_samples=0, release_count=1)
    # emit, blocked, release (X completes it, R=0), then Y re-arms->emit needs next point
    assert run([Y, Y, X, Y, Y], cfg) == ["YES", None, None, "YES", None]
    sm = RearmStateMachine(cfg)
    assert sm.update(Y, 0) == "YES" and sm.state == "REFRACTORY"


def test_n0_only_R_gates():
    cfg = RearmConfig(confirm_count=1, refractory_samples=250, release_count=0)
    sm = RearmStateMachine(cfg)
    assert sm.update(Y, 0) == "YES"
    assert sm.update(Y, 125) is None and sm.state == "REFRACTORY"
    assert sm.update(Y, 250) is None and sm.state == "ARMED"  # R reached, no release needed
    assert sm.update(Y, 375) == "YES"


def test_refractory_blocks_other_label():
    cfg = RearmConfig(refractory_samples=10_000, release_count=0)
    out = run([Y, Y, N_, N_, N_, H, H], cfg)
    assert out == [None, "YES", None, None, None, None, None]


def test_invalid_config():
    for kw in ({"confirm_count": 0}, {"refractory_samples": -1}, {"release_count": -1}):
        with pytest.raises(ValueError):
            RearmConfig(**kw)
    with pytest.raises(ValueError):
        RearmConfig(confirm_count=1.5)  # type: ignore[arg-type]


def test_non_monotonic_time_raises_equal_ok():
    sm = RearmStateMachine()
    sm.update(Y, 500)
    sm.update(Y, 500)
    with pytest.raises(ValueError):
        sm.update(Y, 499)


def test_reset():
    sm = RearmStateMachine()
    sm.update(Y, 1000)
    assert sm.update(Y, 1125) == "YES"
    sm.reset()
    assert sm.state == "ARMED"
    assert sm.update(Y, 0) is None  # time memory cleared too
    assert sm.update(Y, 125) == "YES"


def _load_make_script():
    p = Path(__file__).resolve().parents[1] / "scripts" / "make_rearm_fixtures.py"
    spec = importlib.util.spec_from_file_location("make_rearm_fixtures", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fixtures_deterministic_and_match_machine(tmp_path):
    mod = _load_make_script()
    out = tmp_path / "sequences.csv"
    mod.write_fixtures(out)
    committed = Path(__file__).resolve().parents[1] / "fixtures" / "rearm" / "sequences.csv"
    assert out.read_bytes() == committed.read_bytes()
    # independent replay of the committed file
    import csv

    sms: dict[str, RearmStateMachine] = {}
    with committed.open() as f:
        for row in csv.DictReader(f):
            sid = row["sequence_id"]
            if row["step"] == "0":
                sms[sid] = RearmStateMachine(
                    RearmConfig(
                        int(row["confirm_count"]),
                        int(row["refractory_samples"]),
                        int(row["release_count"]),
                    )
                )
            got = sms[sid].update(row["decision"], int(row["window_end_sample"]))
            assert (got or "") == row["expected_emit"]
            assert sms[sid].state == row["expected_state"]
