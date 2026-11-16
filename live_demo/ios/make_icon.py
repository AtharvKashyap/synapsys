"""Draw the app icon (1024x1024, no transparency): the three stimulus tiles in their
triangle layout, in the app's command colours, around the central fixation dot.

    ../../.venv/bin/python make_icon.py

writes SynapSysDemo/Assets.xcassets/AppIcon.appiconset/AppIcon.png.
"""

from pathlib import Path

from PIL import Image, ImageDraw

OUT = (
    Path(__file__).parent
    / "SynapSysDemo"
    / "Assets.xcassets"
    / "AppIcon.appiconset"
    / "AppIcon.png"
)
SIZE, SS = 1024, 4  # draw 4x larger, then downsample: smooth edges
BG = (18, 18, 20)
YES, NO, HELP = (26, 153, 69), (204, 33, 33), (242, 128, 5)  # Palette in ContentView.swift
DOT = (232, 150, 150)  # fixation colour in the stimulus renderer
TILE, RADIUS = 250, 40
CENTRES = {YES: (322, 360), NO: (702, 360), HELP: (512, 690)}  # top-left, top-right, bottom

img = Image.new("RGB", (SIZE * SS, SIZE * SS), BG)
d = ImageDraw.Draw(img)
for colour, (cx, cy) in CENTRES.items():
    h = TILE // 2
    d.rounded_rectangle(
        [(cx - h) * SS, (cy - h) * SS, (cx + h) * SS, (cy + h) * SS], RADIUS * SS, fill=colour
    )
r, (cx, cy) = 22, (512, 525)
d.ellipse([(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS], fill=DOT)
OUT.parent.mkdir(parents=True, exist_ok=True)
img.resize((SIZE, SIZE), Image.LANCZOS).save(OUT)
print(OUT)
