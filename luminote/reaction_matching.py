"""Match a reply to reaction images in activation space rather than on coordinates.

Valence/arousal/dominance keep only a sliver of what distinguishes reaction memes: on a set of
62 well-known memes, nearest-coordinate selection retrieved the intended meme at about chance,
while comparing whitened activations of the reply and of each meme's description did far better
(see scripts/benchmark_reaction_matching.py). Coordinates remain for colour and explanation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class ReactionIndex:
    ids: list[str]
    embeddings: np.ndarray  # [asset, dimension], unit rows
    layer: int
    model_name: str

    @classmethod
    def load(cls, path: Path) -> ReactionIndex | None:
        if not path.exists():
            return None
        archive = np.load(path)
        return cls(
            ids=[str(value) for value in archive["ids"]],
            embeddings=archive["embeddings"].astype(np.float32),
            layer=int(archive["layer"]),
            model_name=str(archive["model_name"]),
        )

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            ids=np.array(self.ids),
            embeddings=self.embeddings.astype(np.float32),
            layer=np.array(self.layer),
            model_name=np.array(self.model_name),
        )

    def rank(self, embedding: torch.Tensor, limit: int = 8) -> list[dict]:
        similarities = self.embeddings @ embedding.numpy()
        order = np.argsort(-similarities, kind="stable")[:limit]
        return [{"id": self.ids[index], "similarity": round(float(similarities[index]), 4)} for index in order]
