"""Laptop Bluetooth relay: forwards the board's own lines to the iPhone app.

Demo fallback for a board whose radio cannot run (brownout when it starts). The path becomes
ESP32-S3 (decodes) --COM--> laptop --BLE--> iPhone. The relay decides nothing: it passes on
the label in the board's `command` line and a debug text built from the board's decision
line, in the firmware's own BLE formats (firmware/esp32/main/serial_output.cpp), so the
phone app works unchanged. The Bluetooth side is relay/ble_relay.swift (build/ble_relay).
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1]
BINARY = DEMO / "build" / "ble_relay"


def debug_text(ev: dict) -> str:
    """The firmware's BLE debug text for a decision line: '%.2f,%.2f,%.2f m%.2f <decision>'."""
    if ev.get("status") == "DEGENERATE":
        return f"t={ev.get('t')} DEGENERATE"
    s = ev["scores"]
    return f"{s[0]:.2f},{s[1]:.2f},{s[2]:.2f} m{ev['margin']:.2f} {ev['decision']}"


def relay_line(ev: dict) -> str | None:
    """Board event -> relay stdin line, or None if the phone does not get it."""
    if ev.get("event") == "command":
        return f"C {ev['label']}"
    if ev.get("event") is None and "scores" in ev:
        return f"D {debug_text(ev)}"
    return None


class Relay:
    """Runs build/ble_relay; `forward(ev)` sends a board event; `phones` = subscribed phones."""

    def __init__(self, binary: Path = BINARY):
        self.binary = binary
        self.state = "starting"
        self.phones = 0
        self.error: str | None = None
        self.delivered: list[str] = []  # "<label> <phones subscribed>" per command sent
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None

    def start(self) -> None:
        if not self.binary.exists():
            self.error = f"{self.binary.name} not built: run relay/build_relay.sh"
            return
        self._proc = subprocess.Popen(
            [str(self.binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            self.handle_output(line.strip())
        self.state, self.phones = "stopped", 0

    def handle_output(self, line: str) -> None:
        word, _, rest = line.partition(" ")
        if word == "state":
            self.state = rest
            if rest != "poweredOn":
                self.phones = 0
        elif word == "advertising":
            self.state = "advertising"
        elif word == "phone":
            self.phones = int(rest)
        elif word == "sent":
            self.delivered.append(rest.removeprefix("C "))
        elif word == "error":
            self.error = rest

    def forward(self, ev: dict) -> None:
        line = relay_line(ev)
        if line is None or self._proc is None or self._proc.poll() is not None:
            return
        with self._lock:
            try:
                self._proc.stdin.write(line + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                self.state = "stopped"

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.stdin.close()  # the relay exits when its stdin closes
                self._proc.wait(timeout=2)
            except Exception:
                self._proc.kill()

    @property
    def summary(self) -> str:
        if self.error:
            return f"phone relay: {self.error}"
        if self.state == "stopped":
            return "phone relay stopped"
        if self.state not in ("advertising", "poweredOn"):
            return f"phone relay: Bluetooth {self.state}"
        return "iPhone connected" if self.phones else "waiting for iPhone…"
