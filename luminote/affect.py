from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class Match:
    concept: str
    similarity: float


@dataclass(frozen=True)
class AffectEstimate:
    valence: float
    arousal: float
    confidence: float
    matches: list[Match]

    def to_dict(self) -> dict:
        return asdict(self)


class PrototypeEstimator:
    def __init__(self, vectors_path: Path, metadata_path: Path) -> None:
        archive = np.load(vectors_path)
        metadata = json.loads(metadata_path.read_text())
        self.prototypes = torch.from_numpy(archive["prototypes"]).float()
        self.baselines = torch.from_numpy(archive["baselines"]).float()
        self.concepts: list[str] = metadata["concepts"]
        self.coordinates = torch.tensor(metadata["coordinates"], dtype=torch.float32)
        self.model_name: str = metadata["model_name"]
        self.layer_count = int(self.prototypes.shape[0])

    def estimate(self, activation: torch.Tensor, layer: int) -> AffectEstimate:
        layer = max(0, min(layer, self.layer_count - 1))
        centered = activation.float().cpu() - self.baselines[layer]
        prototypes = self.prototypes[layer]
        centered = centered / centered.norm().clamp_min(1e-8)
        normalized = prototypes / prototypes.norm(dim=1, keepdim=True).clamp_min(1e-8)
        similarities = normalized @ centered
        count = min(5, len(self.concepts))
        values, indices = torch.topk(similarities, count)
        # Only evidence above zero influences the coordinate; softmax keeps it smooth.
        evidence = torch.relu(values)
        weights = torch.softmax(evidence * 6.0, dim=0)
        coordinate = (self.coordinates[indices] * weights[:, None]).sum(dim=0)
        strength = torch.sigmoid((values[0] - 0.12) * 8.0)
        margin = torch.sigmoid((values[0] - values[min(2, count - 1)]) * 10.0)
        agreement = 1.0 - torch.clamp(
            ((self.coordinates[indices] - coordinate) ** 2).sum(dim=1).sqrt().mean() / 1.5,
            0.0,
            1.0,
        )
        confidence = float(torch.clamp(strength * (0.55 + 0.45 * margin) * agreement, 0, 1))
        matches = [
            Match(self.concepts[int(index)], round(float(value), 4))
            for value, index in zip(values, indices, strict=True)
        ]
        return AffectEstimate(
            valence=round(float(coordinate[0]), 4),
            arousal=round(float(coordinate[1]), 4),
            confidence=round(confidence, 4),
            matches=matches,
        )

