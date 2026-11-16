"""PsychoPy renderer for the SynapSys SSVEP interface."""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path

from psychopy import core, event, visual

from synapsys_stimulus.config import RendererConfig
from synapsys_stimulus.patterns import sinusoidal_luminance

LOGGER = logging.getLogger(__name__)


def interpolate_color(
    inactive: tuple[float, float, float],
    active: tuple[float, float, float],
    luminance: float,
) -> tuple[float, float, float]:
    """Interpolate between inactive and active PsychoPy RGB colors."""
    return tuple(low + luminance * (high - low) for low, high in zip(inactive, active, strict=True))


class StimulusRenderer:
    """Render the three-command SynapSys SSVEP interface."""

    def __init__(self) -> None:
        self.frame_log: list[tuple[int, float]] = []

        self.window = visual.Window(
            size=RendererConfig.WINDOW_SIZE,
            fullscr=RendererConfig.FULLSCREEN,
            units="height",
            color=RendererConfig.BACKGROUND_COLOR,
            waitBlanking=True,
        )

        self.glows: list[visual.Rect] = []
        self.shadows: list[visual.Rect] = []
        self.tiles: list[visual.Rect] = []
        self.underlines: list[visual.Rect] = []
        self.captions: list[visual.TextStim] = []
        for position, accent_color, stimulus in zip(
            RendererConfig.TILE_POSITIONS,
            RendererConfig.ACCENT_COLORS,
            RendererConfig.STIMULI,
            strict=True,
        ):
            glow = visual.Rect(
                self.window,
                width=RendererConfig.TILE_WIDTH + RendererConfig.GLOW_MARGIN,
                height=RendererConfig.TILE_HEIGHT + RendererConfig.GLOW_MARGIN,
                pos=position,
                fillColor=RendererConfig.GLOW_COLOR,
                lineColor=None,
            )

            shadow_x = position[0] + RendererConfig.SHADOW_OFFSET[0]
            shadow_y = position[1] + RendererConfig.SHADOW_OFFSET[1]
            shadow = visual.Rect(
                self.window,
                width=RendererConfig.TILE_WIDTH,
                height=RendererConfig.TILE_HEIGHT,
                pos=(shadow_x, shadow_y),
                fillColor=RendererConfig.SHADOW_COLOR,
                lineColor=None,
            )

            tile = visual.Rect(
                self.window,
                width=RendererConfig.TILE_WIDTH,
                height=RendererConfig.TILE_HEIGHT,
                pos=position,
                fillColor=RendererConfig.INACTIVE_COLOR,
                lineColor=None,
            )

            tile_bottom = position[1] - RendererConfig.TILE_HEIGHT / 2
            underline = visual.Rect(
                self.window,
                width=RendererConfig.UNDERLINE_WIDTH,
                height=RendererConfig.UNDERLINE_HEIGHT,
                pos=(position[0], tile_bottom - RendererConfig.UNDERLINE_OFFSET),
                fillColor=accent_color,
                lineColor=None,
            )

            caption = visual.TextStim(
                self.window,
                text=" ".join(stimulus.label),
                pos=(position[0], tile_bottom - RendererConfig.CAPTION_OFFSET),
                height=RendererConfig.CAPTION_HEIGHT,
                color=RendererConfig.CAPTION_COLOR,
                bold=True,
            )

            self.glows.append(glow)
            self.shadows.append(shadow)
            self.tiles.append(tile)
            self.underlines.append(underline)
            self.captions.append(caption)

        self.fixation = visual.Circle(
            self.window,
            radius=RendererConfig.FIXATION_RADIUS,
            pos=RendererConfig.FIXATION_POSITION,
            fillColor=RendererConfig.FIXATION_COLOR,
            lineColor=RendererConfig.FIXATION_COLOR,
        )

        self.title = visual.TextStim(
            self.window,
            text=RendererConfig.TITLE_TEXT,
            pos=RendererConfig.TITLE_POSITION,
            height=RendererConfig.TITLE_HEIGHT,
            color=RendererConfig.TITLE_COLOR,
        )

        self.instructions = visual.TextStim(
            self.window,
            text="ESC to exit",
            pos=(0, -0.485),
            height=0.02,
            color=(0.45, 0.45, 0.45),
        )

    def _draw_frame(self, frame_number: int) -> None:
        """Draw one complete stimulus frame.

        Only tile fillColor varies with frame_number. Every other element
        (glows, shadows, underlines, captions, the central fixation point, title,
        instructions) is drawn at a fixed appearance so it cannot
        contribute unintended temporal modulation.
        """
        self.title.draw()
        for glow, shadow, tile, underline, caption, stimulus in zip(
            self.glows,
            self.shadows,
            self.tiles,
            self.underlines,
            self.captions,
            RendererConfig.STIMULI,
            strict=True,
        ):
            luminance = sinusoidal_luminance(
                frame=frame_number,
                period_frames=stimulus.period_frames,
            )
            tile.fillColor = interpolate_color(
                RendererConfig.INACTIVE_COLOR,
                RendererConfig.ACTIVE_COLOR,
                luminance,
            )
            glow.draw()
            shadow.draw()
            tile.draw()
            underline.draw()
            caption.draw()
        self.fixation.draw()
        self.instructions.draw()

    def _log_refresh_rate(self) -> None:
        """Measure and log the display's actual refresh rate.

        Programmed stimulus frequencies assume REFRESH_RATE_HZ exactly. If
        the display is actually running at a different rate (e.g. a
        ProMotion/120 Hz panel), those frequencies would be silently wrong.
        This measures the real rate and logs what the resulting frequencies
        actually are, without changing behavior.
        """
        measured_hz = self.window.getActualFrameRate()
        if measured_hz is None:
            LOGGER.warning(
                "Could not measure actual display refresh rate; frequencies below assume %.1f Hz.",
                RendererConfig.REFRESH_RATE_HZ,
            )
            return

        LOGGER.info(
            "Measured refresh rate: %.2f Hz (assumed %.1f Hz)",
            measured_hz,
            RendererConfig.REFRESH_RATE_HZ,
        )
        if abs(measured_hz - RendererConfig.REFRESH_RATE_HZ) > RendererConfig.REFRESH_TOLERANCE_HZ:
            LOGGER.warning(
                "Measured refresh rate %.2f Hz differs from the assumed %.1f Hz "
                "by more than %.1f Hz. Stimulus frequencies for this run are not "
                "what the configuration intends.",
                measured_hz,
                RendererConfig.REFRESH_RATE_HZ,
                RendererConfig.REFRESH_TOLERANCE_HZ,
            )

        for stimulus in RendererConfig.STIMULI:
            actual_hz = measured_hz / stimulus.period_frames
            LOGGER.info(
                "%s: nominal %.3f Hz, actual %.3f Hz (%d-frame period)",
                stimulus.label,
                stimulus.nominal_frequency_hz,
                actual_hz,
                stimulus.period_frames,
            )

    def run(self) -> None:
        """Run the stimulus renderer until the user presses Escape."""
        frame_number = 0
        LOGGER.info("Starting SynapSys stimulus renderer.")
        for stimulus in RendererConfig.STIMULI:
            LOGGER.info(
                "%s: %.3f Hz (%d-frame period)",
                stimulus.label,
                stimulus.nominal_frequency_hz,
                stimulus.period_frames,
            )
        self._log_refresh_rate()

        try:
            while True:
                self._draw_frame(frame_number)
                timestamp = core.getTime()
                self.window.flip()
                self.frame_log.append((frame_number, timestamp))

                if "escape" in event.getKeys():
                    break
                frame_number += 1
        finally:
            self.window.close()
            LOGGER.info("Renderer stopped after %d frames.", frame_number)

    def save_frame_log(self, output_directory: Path) -> Path:
        """Save recorded frame timestamps as CSV."""
        output_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
        output_path = output_directory / f"frames_{timestamp}.csv"
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(("frame", "timestamp_s"))
            writer.writerows(self.frame_log)
        return output_path


def configure_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def main() -> None:
    """Application entry point."""
    configure_logging()
    renderer = StimulusRenderer()
    try:
        renderer.run()
    finally:
        output_path = renderer.save_frame_log(Path("logs"))
        LOGGER.info("Frame log written to %s", output_path)


if __name__ == "__main__":
    main()
