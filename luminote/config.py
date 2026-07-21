from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    model_name: str = os.getenv("LUMINOTE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    default_layer: int = int(os.getenv("LUMINOTE_LAYER", "24"))
    max_new_tokens: int = int(os.getenv("LUMINOTE_MAX_NEW_TOKENS", "220"))
    prototype_path: Path = ROOT / "data" / "prototypes" / "qwen2.5-1.5b.npz"
    prototype_meta_path: Path = ROOT / "data" / "prototypes" / "qwen2.5-1.5b.json"
    coordinates_path: Path = ROOT / "data" / "emotion_coordinates.json"
    color_map_path: Path = ROOT / "data" / "public_affective_map.png"
    log_path: Path = ROOT / "logs" / "turns.jsonl"
    generation_log_path: Path = ROOT / "logs" / "generation_integrity.jsonl"


settings = Settings()
