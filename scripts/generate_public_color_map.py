from __future__ import annotations

import colorsys
import math
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "public_affective_map.png"
WIDTH, HEIGHT = 919, 849
LEFT, TOP, RIGHT, BOTTOM = 94, 63, 830, 791
BACKGROUND = (247, 242, 231)


def hue_for(theta: float) -> float:
    """Interpolate a valence/arousal wheel without third-party artwork."""
    anchors = [
        (0.0, 120.0),
        (90.0, 0.0),
        (180.0, -60.0),
        (270.0, -180.0),
        (360.0, -240.0),
    ]
    for (start_angle, start_hue), (end_angle, end_hue) in zip(anchors, anchors[1:]):
        if start_angle <= theta <= end_angle:
            fraction = (theta - start_angle) / (end_angle - start_angle)
            return (start_hue + fraction * (end_hue - start_hue)) % 360.0
    return 120.0


def main() -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    pixels = image.load()
    center_x = (LEFT + RIGHT - 1) / 2
    center_y = (TOP + BOTTOM - 1) / 2
    radius_x = (RIGHT - LEFT) / 2
    radius_y = (BOTTOM - TOP) / 2
    for y in range(TOP, BOTTOM):
        arousal = (center_y - y) / radius_y
        for x in range(LEFT, RIGHT):
            valence = (x - center_x) / radius_x
            radius = math.hypot(valence, arousal)
            if radius > 1.0:
                continue
            theta = math.degrees(math.atan2(arousal, valence)) % 360.0
            hue = hue_for(theta) / 360.0
            saturation = min(0.9, 0.1 + radius * 0.8)
            lightness = 0.94 - radius * 0.36
            red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
            pixels[x, y] = tuple(round(channel * 255) for channel in (red, green, blue))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)
    print(f"Generated {OUTPUT}")


if __name__ == "__main__":
    main()
