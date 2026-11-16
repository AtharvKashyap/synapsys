"""Stage app: flickering targets on the laptop, recorded EEG streamed to the ESP32-S3.

    cd live_demo
    ../.venv/bin/python -m feeder.app --port /dev/cu.usbmodemXXXX   # the board's COM port
    ../.venv/bin/python -m feeder.app --port ... --relay            # phone via the laptop
    ../.venv/bin/python -m feeder.app --fake                        # no board (dev only)

Keys: 1-5 play the tile at that position, left to right (Shift+1-5: its next backup trial),
Space = next scenario in story order (YES, NO, HELP, 10 Hz, 14 Hz),
X = flicker on/off, F = full screen, R = reconnect the board, Esc = quit.

Each scenario replays one 6 s Wang recording exactly as it was recorded: 0.5 s before the
stimulus, 5 s of flicker, 0.5 s after. The tiles flicker during those 5 s with the
highlighted tile marking the one the volunteer looked at. The board's answer comes back over
the same COM port; the phone gets it over BLE: from the board itself, or with --relay from
the laptop, which forwards the board's own command line (for a board whose radio can't run).
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .board_link import DEFAULT_BAUD, BoardLink, LinkError
from .runner import BLOCK, FS, Runner, RunState
from .scenarios import Scenario, load_scenarios

ONSET_S = 0.5  # stimulus onset inside every Wang epoch
STIM_S = 5.0  # stimulus duration
LAG_WARN = 2 * BLOCK  # samples behind real time before the screen warns

# Colours: plain, high contrast.
BG = (14, 15, 17)
PANEL = (24, 26, 30)
TEXT = (232, 232, 232)
DIM = (140, 142, 148)
FAINT = (70, 72, 78)
GREEN = (30, 155, 72)
RED = (205, 35, 35)
ORANGE = (242, 128, 6)
ERROR = (230, 70, 60)
CMD_COLOUR = {"YES": GREEN, "NO": RED, "HELP": ORANGE}
TRACE = (120, 190, 255)
CHANNELS = ("O1", "Oz", "O2", "POz")


def candidate_ports() -> list[str]:
    pats = ["/dev/cu.usbmodem*", "/dev/cu.wchusbserial*", "/dev/cu.usbserial*", "/dev/cu.SLAB*"]
    return sorted({p for pat in pats for p in glob.glob(pat)})


class Board:
    """Owns the link; connects in the background so the window never freezes."""

    def __init__(self, port: str | None, fake: bool):
        self.port_arg = port
        self.fake = fake
        self.fake_board = None
        self.link: BoardLink | None = None
        self.runner: Runner | None = None
        self.status = "connecting…"
        self.ok = False
        self.device = "fake board" if fake else "ESP32-S3"  # what decodes, for every label
        self.on_event = None  # given to every Runner (the phone relay)
        self._thread: threading.Thread | None = None

    def connect(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.ok = False
        self.status = "connecting…"
        self._thread = threading.Thread(target=self._connect, daemon=True)
        self._thread.start()

    def _connect(self) -> None:
        if self.link is not None:
            self.link.close()
            self.link = None
        if self.fake:
            from .fake_board import FakeBoard

            if self.fake_board is None:
                self.fake_board = FakeBoard()
                self._fake_path = self.fake_board.start()
            ports, baud = [self._fake_path], 115200
        else:
            ports = [self.port_arg] if self.port_arg else candidate_ports()
            baud = DEFAULT_BAUD
        if not ports:
            self.status = "no serial port: plug in the COM connector, press R"
            return
        errors = []
        for port in ports:  # the native USB port also looks like usbmodem*: try each
            try:
                link = BoardLink(port, baud=baud)
                ack = link.handshake(4.0)
            except (LinkError, OSError) as e:
                errors.append(f"{Path(port).name}: {e}")
                try:
                    link.close()
                except Exception:
                    pass
                continue
            if ack.get("config") != "benchmark_v1":
                link.close()
                errors.append(f"{Path(port).name}: firmware config {ack.get('config')!r}")
                continue
            runner = Runner(link)
            runner.on_event = self.on_event
            self.link, self.runner, self.ok = link, runner, True
            self.status = f"{'FAKE BOARD' if self.fake else 'ESP32-S3'} on {Path(port).name}"
            return
        self.status = "board not answering: use the COM port, then press R"
        print("connect failed:", "; ".join(errors), file=sys.stderr)

    @property
    def lost(self) -> bool:
        return self.link is not None and self.link._error is not None


class App:
    def __init__(self, args: argparse.Namespace):
        import pygame

        self.pg = pygame
        self.args = args
        self.scenarios: list[Scenario] = load_scenarios()  # story order (scenario numbers)
        self.tiles = sorted(self.scenarios, key=lambda s: s.spec.freq_hz)  # left to right
        self.trial_index = [0] * len(self.scenarios)  # 0 = primary, k = k-th backup
        self.board = Board(args.port, args.fake)
        self.relay = None
        if args.relay:
            from .relay import Relay

            self.relay = Relay()
            self.relay.start()
            self.board.on_event = self.relay.forward
        self.flicker = not args.no_flicker
        self.current: Scenario | None = None
        self.state: RunState | None = None
        self.eeg: np.ndarray | None = None
        self.source = ""
        self.next_number = 1
        self.results: list[bool] = []

        pygame.init()
        flags = pygame.SCALED | pygame.RESIZABLE
        self.screen = pygame.display.set_mode((1280, 720), flags, vsync=1)
        pygame.display.set_caption("SynapSys live demo")
        face = "helveticaneue,helvetica,arial"
        self.f_small = pygame.font.SysFont(face, 18)
        self.f_med = pygame.font.SysFont(face, 26)
        self.f_tile = pygame.font.SysFont(face, 30, bold=True)
        self.f_big = pygame.font.SysFont(face, 46, bold=True)
        self.clock = pygame.time.Clock()
        self.board.connect()

    # ---- actions
    def play_tile(self, position: int, backup: bool = False) -> None:
        """Key 1-5: the tile at that position, left to right."""
        self.play(self.tiles[position - 1].number, backup)

    def play(self, number: int, backup: bool = False) -> None:
        """Scenario number (story order: 1 YES, 2 NO, 3 HELP, 4 10 Hz, 5 14 Hz)."""
        if not self.board.ok or self.board.runner is None or self.board.lost:
            return
        if not 1 <= number <= len(self.scenarios):
            return
        sc = self.scenarios[number - 1]
        if backup:
            self.trial_index[number - 1] = (self.trial_index[number - 1] + 1) % (
                1 + len(sc.alternates)
            )
        k = self.trial_index[number - 1]
        src, eeg = (sc.source, sc.eeg) if k == 0 else sc.alternates[k - 1]
        self.state = self.board.runner.start(eeg, sc.expected)
        self.current, self.eeg, self.source = sc, eeg, src
        self.next_number = number % len(self.scenarios) + 1

    # ---- drawing helpers
    def text(self, s, font, colour, pos, anchor="topleft"):
        surf = font.render(s, True, colour)
        rect = surf.get_rect(**{anchor: pos})
        self.screen.blit(surf, rect)
        return rect

    def mark(self, ok: bool, centre: tuple[int, int], colour, size: int = 30):
        """Check mark (ok) or cross, drawn as lines (no font glyph needed)."""
        pg, (cx, cy), h = self.pg, centre, size // 2
        if ok:
            pts = [(cx - h, cy), (cx - h // 3, cy + h * 2 // 3), (cx + h, cy - h * 2 // 3)]
            pg.draw.lines(self.screen, colour, False, pts, 6)
        else:
            pg.draw.line(self.screen, colour, (cx - h, cy - h), (cx + h, cy + h), 6)
            pg.draw.line(self.screen, colour, (cx - h, cy + h), (cx + h, cy - h), 6)
        return pg.Rect(cx - h, cy - h, 2 * h, 2 * h)

    def stream_time(self, now: float) -> float | None:
        st = self.state
        if st is None or st.started_at == 0.0:
            return None
        return now - st.started_at

    def draw_header(self) -> None:
        pg = self.pg
        self.text("SynapSys · live decoding demo", self.f_small, DIM, (32, 22))
        b = self.board
        colour = GREEN if b.ok and not b.lost else (ORANGE if not b.ok else ERROR)
        status = "board lost (unplugged?) — press R" if b.lost else b.status
        r = self.text(status, self.f_small, TEXT, (1248, 22), "topright")
        pg.draw.circle(self.screen, colour, (r.left - 14, r.centery), 6)
        if self.relay is not None:
            rl = self.relay
            c = GREEN if rl.phones else (ERROR if rl.error or rl.state == "stopped" else ORANGE)
            r2 = self.text(rl.summary, self.f_small, TEXT, (1248, 46), "topright")
            pg.draw.circle(self.screen, c, (r2.left - 14, r2.centery), 6)
        if self.args.fake:
            band = pg.Rect(0, 52, 1280, 30)
            pg.draw.rect(self.screen, ERROR, band)
            self.text(
                "SIMULATED BOARD (Python on the laptop) — NOT THE ESP32",
                self.f_small,
                (255, 255, 255),
                band.center,
                "center",
            )

    def draw_tiles(self, now: float) -> None:
        pg = self.pg
        size, gap, top = 170, 44, 112
        left = (1280 - (5 * size + 4 * gap)) // 2
        t = self.stream_time(now)
        flickering = (
            self.flicker
            and self.state is not None
            and not self.state.done
            and t is not None
            and ONSET_S <= t < ONSET_S + STIM_S
        )
        for i, sc in enumerate(self.tiles):
            x = left + i * (size + gap)
            rect = pg.Rect(x, top, size, size)
            if flickering:
                # Sampled sinusoidal luminance (as in the Wang recordings), moderate contrast.
                lum = 0.5 * (1 + math.sin(2 * math.pi * sc.spec.freq_hz * (t - ONSET_S)))
                v = int(45 + 170 * lum)
                colour = (v, v, v)
            else:
                colour = (78, 80, 86)
            pg.draw.rect(self.screen, colour, rect, border_radius=6)
            attended = self.current is sc and self.state is not None and not self.state.done
            if attended:
                pg.draw.rect(
                    self.screen, (255, 255, 255), rect.inflate(22, 22), 6, border_radius=10
                )
            name = sc.spec.tile or "—"
            self.text(
                name,
                self.f_tile,
                TEXT if sc.spec.tile else DIM,
                (rect.centerx, rect.bottom + 22),
                "midtop",
            )
            self.text(
                f"{sc.spec.freq_hz:g} Hz",
                self.f_small,
                DIM,
                (rect.centerx, rect.bottom + 58),
                "midtop",
            )
            self.text(str(i + 1), self.f_small, DIM, (rect.left + 8, rect.top + 6))

    def draw_eeg(self) -> None:
        pg = self.pg
        box = pg.Rect(32, 400, 1216, 150)
        pg.draw.rect(self.screen, PANEL, box, border_radius=6)
        label = f"Recorded EEG, streaming to the {self.board.device} over COM"
        if self.source:
            label += f"   ·   Wang 2017 benchmark, {self.source.replace('block', 'block')}"
        self.text(label, self.f_small, DIM, (box.left + 12, box.top - 26))
        st, eeg = self.state, self.eeg
        if st is None or eeg is None or st.sent < 2:
            self.text("waiting for a scenario", self.f_small, FAINT, box.center, "center")
            return
        span = int(4 * FS)  # last 4 s
        hi = st.sent
        lo = max(0, hi - span)
        seg = eeg[:, lo:hi]
        row_h = box.height / 4
        x0, w = box.left + 60, box.width - 76
        for c in range(4):
            y_mid = box.top + row_h * (c + 0.5)
            self.text(CHANNELS[c], self.f_small, DIM, (box.left + 12, y_mid), "midleft")
            x = seg[c] - np.median(seg[c])
            scale = max(np.percentile(np.abs(x), 98), 1e-9)
            ys = y_mid - (x / scale) * (row_h * 0.42)
            xs = x0 + (np.arange(seg.shape[1]) / span) * w
            if len(xs) > 1:
                pg.draw.lines(self.screen, TRACE, False, np.column_stack([xs, ys]).tolist(), 1)
        rate_ok = st.max_lag <= LAG_WARN
        msg = (
            "250 samples/s, real time"
            if rate_ok
            else f"link slower than real time (behind {st.max_lag / FS:.1f} s)"
        )
        self.text(
            msg, self.f_small, DIM if rate_ok else ERROR, (box.right - 12, box.top - 26), "topright"
        )

    def draw_result(self, now: float) -> None:
        pg = self.pg
        box = pg.Rect(32, 580, 1216, 104)
        pg.draw.rect(self.screen, PANEL, box, border_radius=6)
        mid = box.centery
        sc, st = self.current, self.state
        if sc is None or st is None:
            pos = (box.left + 24, mid - 14)
            self.text("Press 1–5 to play a scenario", self.f_med, DIM, pos, "midleft")
            if self.flicker:
                warn, colour = "The tiles will flicker (8–15 Hz). Press X to turn it off.", ORANGE
            else:
                warn, colour = "Flicker is off (X turns it on).", DIM
            self.text(warn, self.f_small, colour, (box.left + 24, mid + 22), "midleft")
            return
        exp = sc.expected
        exp_text = (
            f"Expected: {exp}"
            if exp
            else f"Expected: quiet  ({sc.spec.freq_hz:g} Hz is not a command)"
        )
        self.text(
            exp_text,
            self.f_big if exp else self.f_med,
            CMD_COLOUR.get(exp, TEXT),
            (box.left + 24, mid),
            "midleft",
        )

        right = box.right - 24
        if st.error:
            msg = (
                "Board lost — plug it in, press R"
                if self.board.lost
                else "Link error — press the key again"
            )
            self.text(msg, self.f_med, ERROR, (right, mid - 14), "midright")
            self.text(st.error[:80], self.f_small, DIM, (right, mid + 18), "midright")
            return
        if st.command is not None:
            ok = st.command == exp
            secs = (st.command_at - st.started_at) if st.command_at else 0.0
            colour = CMD_COLOUR.get(st.command, TEXT) if ok else ERROR
            mark = self.mark(ok, (right - 18, mid - 12), colour)
            self.text(
                f"{self.board.device} sent {st.command}",
                self.f_big,
                colour,
                (mark.left - 16, mid - 12),
                "midright",
            )
            self.text(
                f"{secs:.1f} s after the recording started (flicker began at 0.5 s)"
                + self.phone_path(),
                self.f_small,
                DIM,
                (right, mid + 26),
                "midright",
            )
            return
        if st.done:
            ok = exp is None
            colour = TEXT if ok else ERROR
            mark = self.mark(ok, (right - 14, mid), colour, size=22)
            dev = self.board.device
            msg = f"{dev}: no command (quiet)" if ok else f"{dev}: no command (expected {exp})"
            self.text(msg, self.f_med, colour, (mark.left - 14, mid), "midright")
            return
        # Running: one dot per decision point (8 per trial), coloured by the board's decision.
        self.text(f"{self.board.device} decoding…", self.f_med, DIM, (right, mid - 18), "midright")
        dots = st.decisions[-8:]
        for k in range(8):
            cx = right - (7 - k) * 26 - 6
            if k < len(dots):
                lab = dots[k].get("decision", "REJECT")
                from .scenarios import COMMAND_OF_BOARD_LABEL

                c = CMD_COLOUR.get(COMMAND_OF_BOARD_LABEL.get(lab, ""), FAINT)
                pg.draw.circle(self.screen, c, (cx, mid + 22), 8)
            else:
                pg.draw.circle(self.screen, FAINT, (cx, mid + 22), 8, 2)

    def phone_path(self) -> str:
        """How the answer reaches the phone, only as far as we know it to be true."""
        if self.relay is not None:
            return " · sent to iPhone"
        runner = self.board.runner
        if self.args.fake or runner is None or runner.board_ble in (None, "off"):
            return ""
        return " · sent over BLE by the board"

    def draw_footer(self) -> None:
        hint = (
            "1–5 tile (left to right) · Shift+1–5 backup trial · Space next · "
            "X flicker · F full screen · R reconnect · Esc quit"
        )
        if not self.flicker:
            hint = "FLICKER OFF · " + hint
        self.text(hint, self.f_small, FAINT, (640, 704), "center")

    # ---- main loop
    def handle_key(self, ev) -> bool:
        pg = self.pg
        if ev.key in (pg.K_ESCAPE, pg.K_q):
            return False
        if pg.K_1 <= ev.key <= pg.K_5:
            self.play_tile(ev.key - pg.K_0, backup=bool(ev.mod & pg.KMOD_SHIFT))
        elif ev.key in (pg.K_SPACE, pg.K_PAGEDOWN, pg.K_RIGHT, pg.K_DOWN, pg.K_RETURN):  # clickers
            self.play(self.next_number)
        elif ev.key == pg.K_x:
            self.flicker = not self.flicker
        elif ev.key == pg.K_f:
            pg.display.toggle_fullscreen()
        elif ev.key == pg.K_r:
            self.board.connect()
        return True

    def frame(self) -> None:
        now = time.monotonic()
        self.screen.fill(BG)
        self.draw_header()
        self.draw_tiles(now)
        self.draw_eeg()
        self.draw_result(now)
        self.draw_footer()
        self.pg.display.flip()

    def run(self) -> int:
        pg = self.pg
        script = [int(c) for c in self.args.script.split(",")] if self.args.script else []
        if any(not 1 <= n <= len(self.scenarios) for n in script):
            raise SystemExit(f"--script numbers must be 1..{len(self.scenarios)}")
        t_launch = time.monotonic()
        shots = Path(self.args.shots) if self.args.shots else None
        running, waiting_since, shot_n, recorded = True, None, 0, set()
        while running:
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    running = False
                elif ev.type == pg.KEYDOWN:
                    running = self.handle_key(ev)
            if self.args.script and not self.board.ok and time.monotonic() - t_launch > 30:
                print("board never connected:", self.board.status, file=sys.stderr)
                running = False
            phone_ready = self.relay is None or self.relay.phones > 0
            if self.args.script and not phone_ready and time.monotonic() - t_launch > 30:
                print("no iPhone connected to the relay:", self.relay.summary, file=sys.stderr)
                running = False
            # Scripted run (tests / rehearsal): play each scenario after the previous one ends.
            if self.args.script and self.board.ok and phone_ready:
                st = self.state
                if st is not None and st.done and st.stream_id not in recorded:
                    recorded.add(st.stream_id)
                    self.results.append(bool(st.outcome_ok))
                if st is None or st.done:
                    if not script:
                        if shots and self.current is not None and shot_n < 2:
                            self.frame()
                            pg.image.save(
                                self.screen, str(shots / f"s{self.current.number}_result.png")
                            )
                        running = False
                    elif waiting_since is None:
                        waiting_since = time.monotonic()
                    elif time.monotonic() - waiting_since > 1.0:
                        self.play(script.pop(0))
                        waiting_since, shot_n = None, 0
            self.frame()
            if shots and self.state is not None and self.current is not None:
                t = self.stream_time(time.monotonic()) or 0.0
                if shot_n == 0 and t > 2.0:
                    pg.image.save(self.screen, str(shots / f"s{self.current.number}_flicker.png"))
                    shot_n = 1
                elif shot_n == 1 and self.state.done:
                    pg.image.save(self.screen, str(shots / f"s{self.current.number}_result.png"))
                    shot_n = 2
            self.clock.tick(60)
        pg.quit()
        if self.relay is not None:
            if self.args.script:
                print("relay delivered (label, phones):", self.relay.delivered)
            self.relay.close()
        if self.args.script:
            print("scenario results:", self.results)
            return 0 if self.results and all(self.results) else 1
        return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SynapSys live demo (laptop side)")
    ap.add_argument(
        "--port", help="the board's COM port (default: try every usbmodem/usbserial port)"
    )
    ap.add_argument("--fake", action="store_true", help="simulated board (development only)")
    ap.add_argument("--no-flicker", action="store_true", help="start with the flicker off")
    ap.add_argument(
        "--relay",
        action="store_true",
        help="forward the board's commands to the iPhone over the laptop's Bluetooth",
    )
    ap.add_argument(
        "--script", help="play these scenarios and exit, e.g. 1,2,3,4,5 (exit 0 = all as expected)"
    )
    ap.add_argument("--shots", help="directory: save screenshots during --script")
    args = ap.parse_args(argv)
    if args.relay and args.fake:
        ap.error("--relay needs the real board: it must never send simulated answers to the phone")
    if args.shots:
        os.makedirs(args.shots, exist_ok=True)
    return App(args).run()


if __name__ == "__main__":
    sys.exit(main())
