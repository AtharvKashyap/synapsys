"""DEV ONLY: a stand-in for the demo firmware on a pseudo-terminal (no ESP32 needed).

Speaks the host-feed protocol and answers like the board: decisions every 125 samples from
625 with the demo confidence gate, re-arm K=2 R=625 N=2, one command line per emitted command,
acks with the CRC of the received bytes. It runs the Python reference decoder, never the C++,
so it is for building and testing the laptop app only. The stage app marks it on screen.

    cd live_demo && ../.venv/bin/python -m feeder.fake_board   # prints the tty path for --port
"""

from __future__ import annotations

import json
import os
import select
import threading
import tty
import zlib

import numpy as np

from synapsys_decoder.config import load_config
from synapsys_decoder.rearm import RearmStateMachine
from synapsys_decoder.streaming import decode_stream
from synapsys_decoder.types import Trial

from .scenarios import CONFIG_PATH, REARM, demo_threshold

WINDOW = 625
HOP = 125


class FakeBoard:
    def __init__(self, threshold: float | None = None, corrupt_next_crc: bool = False):
        self.cfg = load_config(CONFIG_PATH)
        self.threshold = demo_threshold() if threshold is None else threshold
        self.corrupt_next_crc = corrupt_next_crc  # test hook: report a wrong CRC once
        self._master: int | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> str:
        master, slave = os.openpty()
        tty.setraw(slave)  # no echo / line discipline: bytes pass through unchanged
        self._master = master
        self._slave = slave
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return os.ttyname(slave)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)  # the loop polls _stop every 50 ms
        for fd in (self._master, self._slave):
            try:
                os.close(fd)
            except OSError:
                pass

    def _emit(self, obj: dict) -> None:
        os.write(self._master, ("SSVEP " + json.dumps(obj, separators=(",", ":")) + "\n").encode())

    def _decide(self, stream: int, buf: list[list[float]], rearm: RearmStateMachine) -> None:
        eeg = np.asarray(buf, dtype=float).T
        ss = decode_stream(
            Trial(eeg=eeg, fs=250.0, channel_names=tuple(self.cfg.channels)), self.cfg
        )
        k = len(ss.window_end) - 1
        t = int(ss.window_end[k])
        decision = ss.decisions(self.threshold, None)[k]
        winner = ss.labels[ss.winner_index[k]]
        cmd = rearm.update(decision, t)
        self._emit(
            {
                "t": t,
                "stream": f"h{stream}",
                "status": "OK",
                "scores": [round(float(v), 6) for v in ss.scores[k]],
                "winner": winner,
                "margin": round(float(ss.margin[k]), 6),
                "decision": decision,
                "state": rearm.state,
                "ms": 0.0,
            }
        )
        if cmd is not None:
            self._emit({"event": "command", "label": cmd, "t": t, "stream": f"h{stream}"})

    def _loop(self) -> None:
        crc, bad, total, pending = 0, 0, 0, b""
        stream, buf, rearm = None, [], RearmStateMachine(REARM)
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([self._master], [], [], 0.05)
                if not ready:
                    continue
                chunk = os.read(self._master, 65536)
            except OSError:
                return
            pending += chunk
            *lines, pending = pending.split(b"\n")
            for raw in lines:
                line = raw + b"\n"
                text = raw.decode(errors="replace").strip()
                if not text.startswith("Y"):
                    crc = zlib.crc32(line, crc)
                if not text:
                    continue
                cmd, rest = text[0], text[1:].split()
                if cmd == "B":
                    stream, buf, rearm = int(rest[0]), [], RearmStateMachine(REARM)
                elif cmd == "S":
                    if stream is None or len(rest) != 4:
                        bad += 1
                        continue
                    buf.append([float(v) for v in rest])
                    total += 1
                    n = len(buf)
                    if n >= WINDOW and (n - WINDOW) % HOP == 0:
                        self._decide(stream, buf, rearm)
                elif cmd == "Y":
                    reported = crc ^ 1 if self.corrupt_next_crc else crc
                    self.corrupt_next_crc = False
                    self._emit(
                        {
                            "event": "ack",
                            "n": int(rest[0]),
                            "config": "benchmark_v1",
                            "samples": total,
                            "bad_lines": bad,
                            "crc": reported,
                        }
                    )
                    crc = 0
                else:
                    bad += 1


def main() -> None:
    fb = FakeBoard()
    print(fb.start(), flush=True)
    print("fake board running (Ctrl-C to stop)", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        fb.stop()


if __name__ == "__main__":
    main()
