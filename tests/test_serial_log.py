"""Device serial log -> recordings; record/verify scripts end to end on a simulated log."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from synapsys_decoder.config import load_config
from synapsys_decoder.io.csv_eeg import load_csv_trial
from synapsys_decoder.io.serial_log import parse_serial_lines
from synapsys_decoder.streaming import decode_stream

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs" / "synapsys.yaml")
FIX = ROOT / "fixtures" / "synthetic"


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _info(stream):
    return "SSVEP " + json.dumps(
        {
            "event": "raw_info",
            "fs": 250,
            "units": "uV",
            "channels": list(CFG.channels),
            "source": "replay",
            "stream": stream,
        }
    )


def _raw_lines(eeg_v, start_seq=0):
    return [
        f"RAW {start_seq + i} " + " ".join(f"{v * 1e6:.3f}" for v in eeg_v[:, i])
        for i in range(eeg_v.shape[1])
    ]


def _device_lines(fix_id, *, perturb=0.0):
    """Simulated firmware output for one fixture: raw_info, RAW lines, decision lines."""
    trial = load_csv_trial(FIX / fix_id / "raw.csv", fs=250, units="V")
    ss = decode_stream(trial, CFG)
    lines = [_info(fix_id), *_raw_lines(trial.eeg)]
    for i, t in enumerate(ss.window_end):
        if ss.degenerate[i]:
            lines.append(
                "SSVEP "
                + json.dumps(
                    {
                        "t": int(t),
                        "stream": fix_id,
                        "status": "DEGENERATE",
                        "scores": [0, 0, 0],
                        "winner": "NONE",
                        "margin": 0,
                        "decision": "REJECT",
                    }
                )
            )
            continue
        sc = ss.scores[i] + perturb
        lines.append(
            "SSVEP "
            + json.dumps(
                {
                    "t": int(t),
                    "stream": fix_id,
                    "status": "OK",
                    "scores": sc.tolist(),
                    "winner": ss.labels[ss.winner_index[i]],
                    "margin": float(ss.margin[i]),
                    "decision": ss.decisions()[i],
                    "state": "ARMED",
                    "ms": 1.0,
                }
            )
        )
    return trial, lines


def test_parse_roundtrip_and_noise_lines():
    trial, lines = _device_lines("syn_large_dc_offset")
    noisy = [
        "I (123) boot: junk",
        "SSVEP {not json",
        'I (177) app: raw stream ON: "RAW <seq> <O1>"',
        *lines[:50],
        "W (9) acq: hello",
        *lines[50:],
        'SSVEP {"event":"health","fs":250.0}',
    ]
    log = parse_serial_lines(noisy)
    assert len(log.streams) == 1
    st = log.streams[0]
    assert st.contiguous and st.n_samples == trial.n_samples
    back = st.to_trial()
    assert back.channel_names == CFG.channels
    # %.3f uV: <= 0.0005 uV rounding even on the +-300 mV offsets
    assert np.max(np.abs(back.eeg - trial.eeg)) <= 0.5e-9 + 1e-15
    assert len(st.decisions) == 8
    assert any(e.get("event") == "health" for e in log.events)


def test_gap_detection_and_longest_run():
    _, lines = _device_lines("syn_noise_only")
    body = lines[1:]  # RAW lines start at index 0 of body
    raw = [ln for ln in body if ln.startswith("RAW")]
    gappy = [lines[0], *raw[:300], *raw[310:]]
    st = parse_serial_lines(gappy).streams[0]
    assert not st.contiguous and st.gaps[0]["after_seq"] == 299
    with pytest.raises(ValueError, match="not contiguous"):
        st.to_trial()
    kept = st.to_trial(allow_gaps=True)
    assert kept.n_samples == len(raw) - 310
    assert kept.metadata["first_seq"] == 310

    ev = 'SSVEP {"event":"raw_gap","stream":"live","lost":3,"next_seq":5}'
    st2 = parse_serial_lines([lines[0], *raw[:5], ev, *raw[5:20]]).streams[0]
    assert not st2.contiguous


def test_malformed_and_missing_info():
    st = parse_serial_lines(["RAW 7 1 2 3 4", "RAW 8 1 2 3", "RAW 9 1 2 3 4"]).streams[0]
    assert st.malformed == 1 and not st.contiguous
    st = parse_serial_lines(["RAW 7 1 2 3 4", "RAW 8 1 2 3 4"]).streams[0]
    with pytest.raises(ValueError, match="raw_info"):
        st.to_trial()
    t = st.to_trial(fs=250.0, channels=CFG.channels)
    assert t.eeg[0, 0] == pytest.approx(1e-6) and t.metadata["first_seq"] == 7


def test_late_decision_attributed_by_stream_name():
    _, a = _device_lines("syn_yes_clean" if (FIX / "syn_yes_clean").exists() else "syn_flat")
    _, b = _device_lines("syn_noise_only")
    a_dec = [ln for ln in a if '"t":' in ln]
    a_other = [ln for ln in a if '"t":' not in ln]
    log = parse_serial_lines([*a_other, *b[:5], *a_dec[-1:], *b[5:]])
    assert len(log.streams[0].decisions) == 1 and log.streams[1].decisions[0]["t"] == 625


def test_verify_device_log_pass_and_fail():
    verify = _script("verify_device_log").verify
    lines = []
    for f in ("syn_large_dc_offset", "syn_noise_only", "syn_flat"):
        lines += _device_lines(f)[1]
    rep = verify(lines, CFG)
    assert rep["failed"] == 0 and rep["points"] > 0 and len(rep["streams"]) == 3
    _, bad = _device_lines("syn_noise_only", perturb=5e-3)
    assert verify(bad, CFG)["failed"] > 0


def test_record_from_log(tmp_path):
    rec = _script("record_serial")
    _, lines = _device_lines("syn_noise_only")
    log = tmp_path / "serial.log"
    log.write_text("\n".join(lines))
    rc = rec.main(
        [
            "--from-log",
            str(log),
            "--out-dir",
            str(tmp_path / "out"),
            "--label",
            "REST",
            "--split",
            "calibration",
        ]
    )
    assert rc == 0
    (csv,) = (tmp_path / "out").glob("*.csv")
    t = load_csv_trial(csv)
    assert t.true_label == "REST" and t.metadata["split"] == "calibration"
    assert t.channel_names == CFG.channels
