"""Configuration for the SynapSys SSVEP stimulus renderer."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StimulusConfig:
    """Configuration for one SSVEP target."""

    label: str
    period_frames: int

    @property
    def nominal_frequency_hz(self) -> float:
        """Return frequency assuming the configured display refresh rate."""
        return RendererConfig.REFRESH_RATE_HZ / self.period_frames


class RendererConfig:
    """Global renderer configuration."""

    REFRESH_RATE_HZ = 60.0

    # Measured refresh must fall within this many Hz of REFRESH_RATE_HZ, or
    # the logged nominal frequencies are no longer trustworthy for this run.
    REFRESH_TOLERANCE_HZ = 1.0

    WINDOW_SIZE = (1200, 700)
    FULLSCREEN = False

    BACKGROUND_COLOR = (-0.88, -0.88, -0.88)
    ACTIVE_COLOR = (1.0, 1.0, 1.0)
    INACTIVE_COLOR = (-0.6, -0.6, -0.6)

    # Static decorative elements only. None of these may vary frame-to-frame
    # -- only tile luminance is allowed to carry temporal modulation.
    TITLE_TEXT = "S Y N A P S Y S"
    TITLE_COLOR = (0.55, 0.55, 0.55)
    TITLE_POSITION = (0.0, 0.46)
    TITLE_HEIGHT = 0.04

    # A larger, very low-contrast rect behind the shadow card, for a soft
    # layered-depth look.
    GLOW_COLOR = (-0.82, -0.82, -0.82)
    GLOW_MARGIN = 0.12

    SHADOW_COLOR = (-0.75, -0.75, -0.75)
    SHADOW_OFFSET = (0.015, -0.015)

    # One static fixation point in the middle of the layout (not on any tile):
    # the viewer rests their gaze here, then looks at the flickering target
    # they want to select. (0, 0) would touch the HELP tile's glow card, so the
    # point sits at x = 0 in the empty space between the three tiles.
    FIXATION_COLOR = (0.85, 0.2, 0.2)
    FIXATION_RADIUS = 0.018
    FIXATION_POSITION = (0.0, 0.08)

    # Captions live below each tile on the fixed dark background rather than
    # on top of the flickering tile, so contrast never depends on the
    # tile's current luminance.
    CAPTION_COLOR = (0.92, 0.92, 0.92)
    CAPTION_HEIGHT = 0.038
    CAPTION_OFFSET = 0.08

    UNDERLINE_OFFSET = 0.04
    UNDERLINE_WIDTH = 0.16
    UNDERLINE_HEIGHT = 0.006

    # One accent color per target (matched to STIMULI order), used only for
    # the static underline beneath each caption -- a fixed color, not a
    # frame-varying one.
    ACCENT_COLORS = (
        (0.2, 0.85, 0.6),
        (0.95, 0.45, 0.2),
        (0.55, 0.45, 0.95),
    )

    TILE_WIDTH = 0.30
    TILE_HEIGHT = 0.30

    # Triangular layout (top-left / top-right / bottom-center) spaces the
    # three targets further apart in the visual field than a horizontal row,
    # so a viewer fixating on one target keeps the other two further into
    # the periphery.
    TILE_POSITIONS = (
        (-0.55, 0.28),
        (0.55, 0.28),
        (0.0, -0.20),
    )

    # Frame period is the source of truth, not the frequency. Only
    # frequencies of the form 60 / N for integer N are exactly representable
    # as a short repeating frame pattern on a 60 Hz display -- e.g. 8, 11,
    # and 14 Hz are not exactly representable as short integer-period
    # sequences at 60 Hz. (Sampled sinusoidal stimulation at an arbitrary
    # frequency, giving every frame its own luminance value rather than
    # repeating a short pattern, is a valid alternative used elsewhere in
    # SSVEP research; this renderer intentionally does not use it yet,
    # favoring the simpler, more debuggable integer-period model until
    # our own EEG data says otherwise.)
    #
    # At 60 Hz:
    # 7-frame period -> 8.571 Hz
    # 5-frame period -> 12.000 Hz
    # 4-frame period -> 15.000 Hz
    # Chosen for being mutually non-harmonic (no frequency is an integer
    # multiple of another), which avoids classifier crosstalk.
    STIMULI = (
        StimulusConfig("YES", 7),
        StimulusConfig("NO", 5),
        StimulusConfig("HELP", 4),
    )
