from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np
from PIL import Image


class ColorMap:
    """Robustly samples the supplied labeled circumplex texture."""

    # Measured square containing the circumplex in the 919x849 source image.
    SOURCE_CROP = (94, 63, 830, 791)
    NEUTRAL = np.array([247.0, 242.0, 231.0])

    def __init__(self, image_path: Path) -> None:
        image = Image.open(image_path).convert("RGB")
        self.image = np.asarray(image.crop(self.SOURCE_CROP), dtype=np.float32)

    def raw_rgb(self, valence: float, arousal: float) -> tuple[int, int, int]:
        height, width, _ = self.image.shape
        x = int(np.clip((valence + 1) * 0.5 * (width - 1), 0, width - 1))
        y = int(np.clip((1 - (arousal + 1) * 0.5) * (height - 1), 0, height - 1))
        # Labels and axes are dark pixels. Take a robust colorful neighborhood median.
        radius = 6
        patch = self.image[max(0, y-radius):min(height, y+radius+1), max(0, x-radius):min(width, x+radius+1)]
        flat = patch.reshape(-1, 3)
        chroma = flat.max(axis=1) - flat.min(axis=1)
        usable = flat[(flat.mean(axis=1) > 55) & (chroma > 18)]
        sample = np.median(usable if len(usable) else flat, axis=0)
        return tuple(int(round(value)) for value in sample)

    def display_color(self, valence: float, arousal: float, confidence: float) -> dict:
        raw = np.array(self.raw_rgb(valence, arousal), dtype=np.float32)
        confidence = float(np.clip(confidence, 0, 1))
        blend = 0.18 + 0.55 * confidence
        mixed = self.NEUTRAL * (1 - blend) + raw * blend
        # A final saturation cap keeps the transcript calm and readable.
        r, g, b = mixed / 255
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        r, g, b = colorsys.hsv_to_rgb(h, min(s, 0.38), max(v, 0.72))
        final = tuple(int(round(channel * 255)) for channel in (r, g, b))
        return {"raw_rgb": list(map(int, raw)), "hex": "#%02x%02x%02x" % final}

