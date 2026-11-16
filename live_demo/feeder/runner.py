"""Play one scenario trial to the board in real time and collect what the board decided.

Streaming is paced by the clock, not by the UI: the runner thread sends whatever is due at
250 samples/s (in blocks of up to BLOCK samples, one block in flight), so a slow frame on the
laptop never changes what the board receives.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .board_link import BoardLink, LinkError
from .scenarios import COMMAND_OF_BOARD_LABEL

FS = 250.0
BLOCK = 25  # 0.1 s of samples per block (~2.5 kB at 921600 baud = ~27 ms on the wire)


def samples_due(elapsed_s: float, sent: int, total: int, fs: float = FS, block: int = BLOCK) -> int:
    """How many samples to send now: catch up to real time, at most one block at a time."""
    target = min(total, int(elapsed_s * fs))
    return max(0, min(block, target - sent))


@dataclass
class RunState:
    """Shared with the UI (read-only there). Times are time.monotonic() seconds."""

    stream_id: int
    expected: str | None
    started_at: float = 0.0
    sent: int = 0
    total: int = 0
    done: bool = False
    error: str | None = None
    command: str | None = None  # audience label of the first command the board emitted
    command_at: float | None = None
    max_lag: int = 0  # worst samples-behind-real-time seen (link slower than 250 samples/s)
    decisions: list[dict] = field(default_factory=list)  # board decision lines, in order

    @property
    def outcome_ok(self) -> bool | None:
        """None while running; then whether the board did what the screen said to expect."""
        if self.error is not None:
            return False
        if self.command is not None:
            return self.command == self.expected
        return (self.expected is None) if self.done else None


class Runner:
    def __init__(self, link: BoardLink, clock=time.monotonic, sleep=time.sleep):
        self.link = link
        self.clock = clock
        self.sleep = sleep
        # Stream ids start from the clock so a restarted app never reuses the board's last id.
        self._next_id = int(time.time()) % 1_000_000
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self.state: RunState | None = None
        self.on_event = None  # optional callable(ev): every board event of the current stream
        self.board_ble: str | None = None  # "off" / "advertising" / "connected" (health lines)

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, eeg: np.ndarray, expected: str | None, tail_s: float = 0.6) -> RunState:
        """Play eeg (4, n) as a new stream. Never blocks: a running scenario is told to stop,
        and the new run's thread waits for it before touching the link."""
        prev = self._thread
        self._cancel.set()
        cancel = threading.Event()
        self._cancel = cancel
        st = RunState(stream_id=self._next_id, expected=expected, total=eeg.shape[1])
        self._next_id += 1
        self.state = st
        self._thread = threading.Thread(
            target=self._run, args=(st, eeg, tail_s, prev, cancel), daemon=True
        )
        self._thread.start()
        return st

    def cancel(self) -> None:
        self._cancel.set()

    def _collect(self, st: RunState) -> None:
        tag = f"h{st.stream_id}"
        for ev in self.link.drain_events():
            if ev.get("event") == "health":
                self.board_ble = ev.get("ble")
            if ev.get("stream") != tag:
                continue
            if self.on_event is not None:
                self.on_event(ev)
            if ev.get("event") == "command":
                if st.command is None:
                    st.command_at = self.clock()  # set first: the UI reads command, then this
                    st.command = COMMAND_OF_BOARD_LABEL.get(ev["label"], ev["label"])
            elif "scores" in ev and ev.get("event") is None:
                st.decisions.append(ev)

    def _run(self, st, eeg, tail_s, prev: threading.Thread | None, cancel: threading.Event) -> None:
        if prev is not None:
            prev.join()  # at most one block + ack timeout; this is not the UI thread
        st.started_at = self.clock()
        try:
            while st.sent < st.total and not cancel.is_set():
                elapsed = self.clock() - st.started_at
                st.max_lag = max(st.max_lag, min(st.total, int(elapsed * FS)) - st.sent)
                n = samples_due(elapsed, st.sent, st.total)
                if n == 0:
                    self.sleep(0.005)
                else:
                    begin = st.stream_id if st.sent == 0 else None
                    self.link.send_block(eeg[:, st.sent : st.sent + n], begin_stream=begin)
                    st.sent += n
                self._collect(st)
            # The last decision/command lines follow the last ack closely; give them a moment.
            t_end = self.clock() + tail_s
            while self.clock() < t_end and not cancel.is_set():
                self._collect(st)
                self.sleep(0.02)
            self._collect(st)
        except LinkError as e:
            st.error = str(e)
        except Exception as e:  # never let a dead thread look like a quiet result
            st.error = f"{type(e).__name__}: {e}"
        finally:
            st.done = True
