"""esp32_benchmark.py against a simulated host-feed board (protocol of main/host_feed.hpp).

The fake board decodes with the Python reference, so this checks the script's protocol
handling and comparison logic, not the firmware (that is the QEMU / real-board run).
"""

import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from synapsys_decoder.config import load_config
from synapsys_decoder.streaming import decode_stream
from synapsys_decoder.synthetic import make_synthetic_trial
from synapsys_decoder.types import Trial

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs" / "benchmark.yaml")


class FakeBoard:
    def __init__(self, config="benchmark_v1", perturb=0.0, drop_eval=False):
        self.out: list[bytes] = []
        self.inbuf = b""
        self.config, self.perturb, self.drop_eval = config, perturb, drop_eval
        self.stream, self.samples = 0, []

    def _emit(self, obj):
        self.out.append(("SSVEP " + json.dumps(obj) + "\n").encode())

    def _scores(self):
        x = np.asarray(self.samples).T
        ss = decode_stream(Trial(eeg=x, fs=250.0, channel_names=CFG.channels), CFG)
        return ss

    def write(self, data: bytes):
        self.inbuf += data
        *lines, self.inbuf = self.inbuf.split(b"\n")
        for raw in lines:
            line = raw.decode()
            if line.startswith("B "):
                self.stream, self.samples = int(line[2:]), []
            elif line.startswith("S "):
                self.samples.append([float(v) for v in line[2:].split()])
                n = len(self.samples)
                if n >= 625 and (n - 625) % 125 == 0:
                    ss = self._scores()
                    sc = (ss.scores[-1] + self.perturb).tolist()
                    self._emit(
                        {
                            "t": n,
                            "stream": f"h{self.stream}",
                            "status": "OK",
                            "scores": sc,
                            "winner": ss.labels[ss.winner_index[-1]],
                            "margin": float(ss.margin[-1]),
                            "decision": "X",
                            "ms": 3.0,
                        }
                    )
            elif line == "E" and not self.drop_eval:
                full = np.asarray(self.samples).T  # window [n-625, n), filtered from start
                from synapsys_decoder.pipeline import decode_trial

                tr = Trial(eeg=full, fs=250.0, channel_names=CFG.channels)
                r = decode_trial(tr, CFG, window_start_s=(full.shape[1] - 625) / 250).result
                self._emit(
                    {
                        "event": "eval",
                        "stream": f"h{self.stream}",
                        "t": len(self.samples),
                        "status": "OK",
                        "scores": [r.scores[k] + self.perturb for k in CFG.labels],
                        "winner": r.winner,
                        "margin": r.margin,
                        "decision": r.decision,
                        "ms": 3.0,
                    }
                )
            elif line.startswith("Y "):
                self._emit(
                    {
                        "event": "ack",
                        "n": int(line[2:]),
                        "config": self.config,
                        "samples": 0,
                        "bad_lines": 0,
                    }
                )

    def readline(self):
        return self.out.pop(0) if self.out else b""


def _load(monkeypatch, board):
    spec = importlib.util.spec_from_file_location("eb", ROOT / "scripts" / "esp32_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def trials(_d, s, **_k):
        out = []
        for i, f in enumerate(CFG.targets.values()):
            t = make_synthetic_trial(
                f, 250.0, 6.0, CFG.channels, np.random.default_rng(10 * s + i), snr_db=-10.0
            )
            out.append(dataclasses.replace(t, subject_id=f"S{s}", true_label=None))
        return out

    monkeypatch.setattr(mod, "load_benchmark_subject", trials)
    monkeypatch.setattr(mod, "benchmark_files_present", lambda *_a: True)
    import serial

    monkeypatch.setattr(serial, "serial_for_url", lambda *_a, **_k: board)
    return mod


def test_pass_against_reference_board(monkeypatch, tmp_path, capsys):
    mod = _load(monkeypatch, FakeBoard())
    assert mod.main(["--port", "fake", "--subjects", "1", "--results-root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "same winner on 3/3" in out and "24 hop points, 0 failed" in out
    (run,) = tmp_path.iterdir()
    assert json.loads((run / "run.json").read_text())["metrics"]["all_pass"] is True


def test_fails_on_score_error_and_missing_eval(monkeypatch, tmp_path):
    mod = _load(monkeypatch, FakeBoard(perturb=5e-3))
    assert mod.main(["--port", "f", "--subjects", "1", "--results-root", str(tmp_path / "a")]) == 1
    mod = _load(monkeypatch, FakeBoard(drop_eval=True))
    assert mod.main(["--port", "f", "--subjects", "1", "--results-root", str(tmp_path / "b")]) == 1
    (run,) = (tmp_path / "b").iterdir()
    assert json.loads((run / "run.json").read_text())["metrics"]["missing_evals"] == 3


def test_refuses_wrong_board_config(monkeypatch, tmp_path):
    mod = _load(monkeypatch, FakeBoard(config="synapsys_v1"))
    with pytest.raises(SystemExit, match="benchmark_v1"):
        mod.main(["--port", "f", "--subjects", "1", "--results-root", str(tmp_path)])
