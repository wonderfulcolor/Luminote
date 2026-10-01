from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    model_name: str = os.getenv("LUMINOTE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    default_layer: int = int(os.getenv("LUMINOTE_LAYER", "24"))
    # Layer for per-token readings of the user message and reply; see scripts/benchmark_layers.py.
    span_layer: int = int(os.getenv("LUMINOTE_SPAN_LAYER", "16"))
    # "boundary" colours the weather from the planned response; "reply" from the reply as written.
    weather_source: str = os.getenv("LUMINOTE_WEATHER_SOURCE", "boundary")
    max_new_tokens: int = int(os.getenv("LUMINOTE_MAX_NEW_TOKENS", "220"))
    prototype_path: Path = ROOT / "data" / "prototypes" / "qwen2.5-1.5b.npz"
    prototype_meta_path: Path = ROOT / "data" / "prototypes" / "qwen2.5-1.5b.json"
    coordinates_path: Path = ROOT / "data" / "emotion_coordinates.json"
    examples_path: Path = ROOT / "data" / "calibration_examples.jsonl"
    color_map_path: Path = ROOT / "data" / "public_affective_map.png"
    reaction_index_path: Path = ROOT / "luminote-reactions" / "metadata" / "reaction_embeddings.npz"
    log_path: Path = ROOT / "logs" / "turns.jsonl"
    generation_log_path: Path = ROOT / "logs" / "generation_integrity.jsonl"


settings = Settings()
