"""Laptop feeder: scenarios, protocol formatting, pacing, and full runs against the fake board."""

from __future__ import annotations

import zlib

import numpy as np
import pytest

from feeder import runner as runner_mod
from feeder.board_link import BoardLink, LinkError, parse_line, s_line
from feeder.fake_board import FakeBoard
from feeder.runner import Runner, samples_due
from feeder.scenarios import CACHE, SPECS, load_scenarios

needs_cache = pytest.mark.skipif(
    not CACHE.exists(), reason="run `python -m feeder.scenarios` first (needs data/benchmark)"
)


# ---- pure pieces


def test_s_line_format_keeps_float32_data_exact():
    v = np.array([1.2345678901234567e-06, -3e-05, 0.0, 7.000000000000001e-07])
    assert s_line(v) == "S 1.23456789e-06 -3e-05 0 7e-07"
    # Wang samples are float32 microvolts: 9 digits give back exactly the same float32 value
    rng = np.random.default_rng(0)
    uv = rng.normal(0, 30, size=(4, 500)).astype(np.float32)
    for k in range(uv.shape[1]):
        line = s_line(uv[:, k].astype(np.float64) * 1e-6)
        back = np.array([float(x) for x in line.split()[1:]]) * 1e6
        assert np.array_equal(back.astype(np.float32), uv[:, k])


def test_parse_line_ignores_logs_and_bad_json():
    assert parse_line("I (123) app: hello") is None
    assert parse_line('SSVEP {"event":"ack"') is None
    assert parse_line('noise SSVEP {"event":"ack","n":3}') == {"event": "ack", "n": 3}


def test_pacer_rate_independent_of_ui():
    # A 0.7 s stall (e.g. a dropped frame burst) is caught up in blocks; total = fs * elapsed.
    sent, t = 0, 0.0
    for t in [0.1 * k for k in range(1, 31)] + [3.8] + [3.8 + 0.01 * k for k in range(1, 300)]:
        while (n := samples_due(t, sent, 1500)) > 0:
            assert n <= runner_mod.BLOCK
            sent += n
        assert sent == min(1500, int(t * 250))
    assert sent == 1500


def test_scenario_specs_match_plan():
    assert [(s.tile, s.freq_hz, s.expected) for s in SPECS] == [
        ("YES", 8.0, "YES"),
        ("NO", 12.0, "NO"),
        ("HELP", 15.0, "HELP"),
        ("", 10.0, None),
        ("", 14.0, None),
    ]


# ---- against the fake board (real pty, real threads, Python reference decoder)


@pytest.fixture
def board():
    fb = FakeBoard()
    link = BoardLink(fb.start(), baud=115200, ack_timeout_s=5.0)  # a pty takes no custom baud
    yield fb, link
    link.close()
    fb.stop()


def fast_runner(link):
    """Runner on a virtual clock that advances 0.1 s per sleep: a 6 s trial runs in ~1 s."""
    clock = {"t": 0.0}

    def now():
        return clock["t"]

    def sleep(dt):
        clock["t"] += max(dt, 0.1)

    return Runner(link, clock=now, sleep=sleep)


def run(link, eeg, expected, r=None):
    r = r or fast_runner(link)
    st = r.start(eeg, expected, tail_s=0.5)
    r._thread.join(timeout=60)
    assert st.done
    return st


def test_ack_crc_roundtrip_with_fake_board(board):
    _, link = board
    ack = link.handshake(5.0)
    assert ack["config"] == "benchmark_v1"
    ack = link.send_block(np.zeros((4, 10)), begin_stream=1)
    expect = "B 1\n" + "".join(s_line(np.zeros(4)) + "\n" for _ in range(10))
    assert ack["crc"] == zlib.crc32(expect.encode())


def test_crc_mismatch_aborts_scenario(board):
    fb, link = board
    link.handshake(5.0)
    fb.corrupt_next_crc = True
    st = run(link, np.zeros((4, 300)), "YES")
    assert st.error is not None and "CRC" in st.error
    assert st.outcome_ok is False
    link.handshake(5.0)  # resync, then a corrupted block raises directly
    fb.corrupt_next_crc = True
    with pytest.raises(LinkError):
        link.send_block(np.zeros((4, 5)), begin_stream=9)


@needs_cache
@pytest.mark.parametrize("number", [1, 2, 3, 4, 5])
def test_each_scenario_gives_expected_outcome(board, number):
    _, link = board
    link.handshake(5.0)
    sc = load_scenarios()[number - 1]
    st = run(link, sc.eeg, sc.expected)
    assert st.error is None
    assert st.sent == 1500
    assert [d["t"] for d in st.decisions] == list(range(625, 1501, 125))
    assert st.command == sc.expected
    assert st.outcome_ok is True


@needs_cache
def test_same_label_twice_both_command(board):
    _, link = board
    link.handshake(5.0)
    yes = load_scenarios()[0]
    r = fast_runner(link)
    first = run(link, yes.eeg, "YES", r)
    second = run(link, yes.eeg, "YES", r)
    assert first.command == "YES" and second.command == "YES"
    assert first.stream_id != second.stream_id


def test_handshake_fails_cleanly_without_board():
    fb = FakeBoard()
    path = fb.start()
    fb.stop()  # nothing answers any more
    with pytest.raises((LinkError, OSError)):  # no answer, or the OS refusing a closed pty
        link = BoardLink(path, baud=115200)
        try:
            link.handshake(1.0)
        finally:
            link.close()


@needs_cache
def test_link_recovers_after_crc_error(board):
    fb, link = board
    link.handshake(5.0)
    r = fast_runner(link)
    fb.corrupt_next_crc = True
    broken = run(link, np.zeros((4, 300)), "YES", r)
    assert broken.error is not None
    assert link.needs_resync
    no = load_scenarios()[1]
    st = run(link, no.eeg, "NO", r)  # resyncs by itself, then plays normally
    assert st.error is None and st.command == "NO"
    assert not link.needs_resync


@needs_cache
def test_lag_is_reported_when_link_is_slow(board):
    _, link = board
    link.handshake(5.0)
    clock = {"t": 0.0}
    real_send = link.send_block

    def slow_send(samples, begin_stream=None):  # each block "takes" 0.2 s: half real time
        clock["t"] += 0.2
        return real_send(samples, begin_stream)

    link.send_block = slow_send
    r = Runner(
        link, clock=lambda: clock["t"], sleep=lambda dt: clock.__setitem__("t", clock["t"] + dt)
    )
    st = r.start(load_scenarios()[0].eeg, "YES", tail_s=0.2)
    r._thread.join(timeout=60)
    assert st.max_lag > 2 * runner_mod.BLOCK  # the UI warns above this


@needs_cache
def test_stage_app_scripted_run_headless(monkeypatch):
    """The real pygame app, no window (SDL dummy driver), fake board: 1 command + 1 quiet."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from feeder.app import main

    assert main(["--fake", "--script", "3,4"]) == 0


@needs_cache
def test_unplugged_board_is_an_error_never_a_quiet_tick():
    fb = FakeBoard()
    link = BoardLink(fb.start(), baud=115200, ack_timeout_s=1.0)
    link.handshake(5.0)
    quiet = load_scenarios()[3]  # a quiet scenario: the dangerous case for a false tick
    fb.stop()  # board gone
    link.needs_resync = True  # as after an earlier link error
    st = run(link, quiet.eeg, None)
    assert st.error is not None
    assert st.outcome_ok is False
    link.close()


@needs_cache
def test_new_scenario_never_blocks_the_caller(board):
    import time as _time

    _, link = board
    link.handshake(5.0)
    r = Runner(link)  # real clock: the first run is mid-stream when the second starts
    sc = load_scenarios()
    first = r.start(sc[0].eeg, "YES")
    _time.sleep(0.5)
    t = _time.monotonic()
    second = r.start(sc[1].eeg, "NO")
    assert _time.monotonic() - t < 0.05  # start() returns immediately
    r._thread.join(timeout=30)
    assert first.done and first.command is None  # cancelled before its command
    assert second.error is None and second.command == "NO"


# ---- laptop Bluetooth relay (board radio can't run): forwards the board's own lines only
def test_relay_lines_use_the_firmware_ble_formats():
    from feeder.relay import relay_line

    assert relay_line({"event": "command", "label": "F8_0", "t": 750, "stream": "h1"}) == "C F8_0"
    dec = {
        "t": 625,
        "stream": "h1",
        "status": "OK",
        "scores": [0.450502, 0.228937, 0.1392],
        "winner": "F8_0",
        "margin": 0.221565,
        "decision": "F8_0",
        "state": "CANDIDATE",
    }
    assert relay_line(dec) == "D 0.45,0.23,0.14 m0.22 F8_0"
    assert relay_line({**dec, "status": "DEGENERATE"}) == "D t=625 DEGENERATE"
    assert relay_line({"event": "health", "ble": "off"}) is None
    assert relay_line({"event": "ack", "n": 3}) is None


def test_relay_status_text():
    from feeder.relay import Relay

    r = Relay()
    for line in ("state poweredOn", "advertising"):
        r.handle_output(line)
    assert r.summary == "waiting for iPhone…"
    r.handle_output("phone 1")
    assert r.summary == "iPhone connected"
    r.handle_output("state poweredOff")
    assert r.phones == 0 and "poweredOff" in r.summary
    missing = Relay(binary=CACHE.parent / "no_such_relay")
    missing.start()
    assert "not built" in missing.summary


@needs_cache
def test_relay_gets_exactly_the_boards_lines_for_its_stream(board, tmp_path):
    """A stand-in relay binary records its stdin: one D per decision, then the board's C."""
    from feeder.relay import Relay

    _, link = board
    link.handshake(5.0)
    got = tmp_path / "stdin.txt"
    fake = tmp_path / "relay.sh"
    fake.write_text(f'#!/bin/sh\necho advertising\necho "phone 1"\ncat > "{got}"\n')
    fake.chmod(0o755)
    relay = Relay(binary=fake)
    relay.start()
    r = fast_runner(link)
    r.on_event = relay.forward
    link.events.put({"event": "command", "label": "F15_0", "stream": "h999999999"})  # stale
    st = run(link, load_scenarios()[0].eeg, "YES", r)
    relay.close()
    lines = got.read_text().splitlines()
    assert [ln[0] for ln in lines] == ["D"] * len(st.decisions[:2]) + ["C"] + ["D"] * (
        len(st.decisions) - 2
    )
    assert lines[2] == "C F8_0" and st.command == "YES"
    assert "F15_0" not in got.read_text()


def test_relay_refuses_the_fake_board():
    from feeder.app import main

    with pytest.raises(SystemExit):
        main(["--fake", "--relay"])
