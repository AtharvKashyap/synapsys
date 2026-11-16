"""Frame-based luminance patterns for SSVEP stimulation."""

import math


def sinusoidal_luminance(frame: int, period_frames: int) -> float:
    """Return normalized luminance in [0, 1] for a given display frame.

    Using frame number rather than wall-clock time prevents OS scheduling
    jitter from directly changing the intended stimulus phase.
    """
    if frame < 0:
        raise ValueError("frame must be non-negative")
    if period_frames <= 0:
        raise ValueError("period_frames must be positive")

    phase = 2.0 * math.pi * frame / period_frames
    return 0.5 * (1.0 + math.sin(phase))
