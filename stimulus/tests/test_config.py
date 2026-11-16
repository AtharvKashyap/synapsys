import pytest

from synapsys_stimulus.config import RendererConfig


def test_nominal_frequencies() -> None:
    frequencies = [stimulus.nominal_frequency_hz for stimulus in RendererConfig.STIMULI]
    assert frequencies[0] == pytest.approx(8.5714, rel=1e-3)
    assert frequencies[1] == pytest.approx(12.0)
    assert frequencies[2] == pytest.approx(15.0)


def test_fixation_point_clear_of_every_tile_and_caption() -> None:
    """The single central fixation point must not sit on any tile, glow card or caption."""
    fx, fy = RendererConfig.FIXATION_POSITION
    r = RendererConfig.FIXATION_RADIUS
    half_w = (RendererConfig.TILE_WIDTH + RendererConfig.GLOW_MARGIN) / 2
    half_h = (RendererConfig.TILE_HEIGHT + RendererConfig.GLOW_MARGIN) / 2
    for x, y in RendererConfig.TILE_POSITIONS:
        caption_y = y - RendererConfig.TILE_HEIGHT / 2 - RendererConfig.CAPTION_OFFSET
        bottom = min(y - half_h, caption_y - RendererConfig.CAPTION_HEIGHT)
        overlaps_x = abs(fx - x) < half_w + r
        overlaps_y = bottom - r < fy < y + half_h + r
        assert not (overlaps_x and overlaps_y), (x, y)
    assert fx == 0.0  # horizontally centred
