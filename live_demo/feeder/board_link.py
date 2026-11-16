"""Laptop <-> ESP32-S3 host-feed link (same protocol as scripts/esp32_benchmark.py).

host -> board, one ASCII line each:  B <id> | S <v0> <v1> <v2> <v3> (volts, %.9g) | Y <n>
board -> host: lines "SSVEP {json}": acks {"event":"ack","n","config","bad_lines","crc"},
decisions {"t","stream":"h<id>","scores","winner","margin","decision",...} and commands
{"event":"command","label","t","stream":"h<id>"}. The ack's crc is the CRC-32 (zlib) of every
byte sent since the previous Y line, so lost or corrupted bytes are detected.

A reader thread parses everything the board prints; send_block() writes one block, then
waits for its ack (at most one block in flight, far below the board's 16 KB RX buffer).
"""

from __future__ import annotations

import json
import queue
import threading
import time
import zlib

import numpy as np

DEFAULT_BAUD = 921600


class LinkError(Exception):
    """Bytes lost or corrupted, the board stopped answering, or the port went away."""


def s_line(sample_v: np.ndarray) -> str:
    """One sample (4 values, volts) as the board's S line. 9 significant digits: the Wang data
    is float32 (which round-trips at 9 digits), and the board parses input one byte at a time,
    so %.17g lines (~92 B) could not keep up with 250 samples/s; these are ~55 B."""
    return "S " + " ".join(f"{float(v):.9g}" for v in sample_v)


def parse_line(line: str) -> dict | None:
    p = line.find("SSVEP ")
    if p == -1:
        return None
    try:
        ev = json.loads(line[p + 6 :])
    except json.JSONDecodeError:
        return None
    return ev if isinstance(ev, dict) else None


class BoardLink:
    def __init__(self, url: str, baud: int = DEFAULT_BAUD, ack_timeout_s: float = 3.0):
        import serial  # pyserial; also handles socket:// (QEMU) and pty paths (fake board)

        self.ser = serial.serial_for_url(url, baudrate=baud, timeout=0.05, exclusive=True)
        self.needs_resync = False  # set by any LinkError; the next send handshakes first
        self.ack_timeout_s = ack_timeout_s
        self.events: queue.Queue[dict] = queue.Queue()  # everything except acks
        self.log: list[str] = []
        self._acks: dict[int, dict] = {}
        self._cv = threading.Condition()
        self._sync = 0
        self._bad_lines = 0
        self._closed = False
        self._error: str | None = None
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    # ---- reader thread
    def _read_loop(self) -> None:
        buf = b""
        while not self._closed:
            try:
                chunk = self.ser.read(4096)
            except Exception as e:  # port unplugged
                with self._cv:
                    self._error = f"port lost: {e}"
                    self._cv.notify_all()
                return
            if not chunk:
                continue
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for raw in lines:
                line = raw.decode("utf-8", errors="replace").strip()
                self.log.append(line)
                ev = parse_line(line)
                if ev is None:
                    continue
                if ev.get("event") == "ack":
                    with self._cv:
                        self._acks[int(ev.get("n", -1))] = ev
                        self._cv.notify_all()
                else:
                    self.events.put(ev)

    # ---- sending
    def _wait_ack(self, n: int, timeout_s: float) -> dict | None:
        t_end = time.monotonic() + timeout_s
        with self._cv:
            while n not in self._acks:
                if self._error:
                    raise LinkError(self._error)
                left = t_end - time.monotonic()
                if left <= 0:
                    return None
                self._cv.wait(left)
            return self._acks.pop(n)

    def handshake(self, wait_s: float = 10.0) -> dict:
        """Repeat Y until the board answers; returns the ack (it carries the board's config)."""
        t_end = time.monotonic() + wait_s
        with self._cv:
            self._acks.clear()  # stale acks from a broken block must not be matched later
        while time.monotonic() < t_end:
            self._sync += 1
            # The leading newline ends any half-received line; Y resets the board's CRC.
            try:
                self.ser.write(f"\nY {self._sync}\n".encode())
            except Exception as e:  # port unplugged
                raise LinkError(f"write failed: {e}") from e
            ack = self._wait_ack(self._sync, 0.5)
            if ack is not None:
                self._bad_lines = int(ack.get("bad_lines", 0))
                self.needs_resync = False
                return ack
        raise LinkError(
            "no answer from the board. Use the COM connector's port (the native USB port also "
            "shows up as /dev/cu.usbmodem* but ignores input); demo firmware flashed? press RST"
        )

    def send_block(self, samples_v: np.ndarray, begin_stream: int | None = None) -> dict:
        """Send samples (4, n) volts, optionally starting stream <id> first; wait for the ack.
        After a LinkError the link resynchronises (handshake) before the next block."""
        if self.needs_resync:
            self.handshake(5.0)
        try:
            return self._send_block(samples_v, begin_stream)
        except LinkError:
            self.needs_resync = True
            raise

    def _send_block(self, samples_v: np.ndarray, begin_stream: int | None) -> dict:
        lines = [] if begin_stream is None else [f"B {begin_stream}"]
        lines += [s_line(samples_v[:, k]) for k in range(samples_v.shape[1])]
        data = "".join(f"{ln}\n" for ln in lines).encode()
        self._sync += 1
        n = self._sync
        try:
            self.ser.write(data + f"Y {n}\n".encode())
        except Exception as e:
            raise LinkError(f"write failed: {e}") from e
        ack = self._wait_ack(n, self.ack_timeout_s)
        if ack is None:
            raise LinkError(f"no ack for block {n}")
        bad = int(ack.get("bad_lines", 0)) - self._bad_lines
        self._bad_lines = int(ack.get("bad_lines", 0))
        if bad:
            raise LinkError(f"board could not parse {bad} line(s) in block {n}")
        if int(ack.get("crc", -1)) != zlib.crc32(data):
            raise LinkError(f"CRC mismatch in block {n} (bytes lost on the link)")
        return ack

    def drain_events(self) -> list[dict]:
        out = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                return out

    def close(self) -> None:
        self._closed = True
        try:
            self.ser.close()
        except Exception:
            pass
