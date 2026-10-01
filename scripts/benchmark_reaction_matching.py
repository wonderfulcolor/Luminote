"""How often does reaction selection pick the reaction a chat moment calls for?

data/reaction_eval/ holds 62 well-known reaction memes, each described two ways (the image from
outside, and the reaction in its own voice), and 186 chat moments (two assistant replies and one
user message per meme) where that meme would be the right reaction. No moment names its meme or
the feeling. Only text is included; no images.

Methods compared at the calibrated match layer:
- coordinates: nearest valence/arousal/dominance, the original selection
- raw:         cosine similarity of mean activations
- whitened:    cosine after whitening by within-feeling variation (what the app uses)

The descriptions and moments were written by Claude, so shared style may flatter raw similarity;
whitening is meant to discount that, but treat absolute numbers as optimistic.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from luminote.affect import PrototypeEstimator
from luminote.calibration import pool_texts
from luminote.config import ROOT, settings
from luminote.model import LocalModel

EVAL = ROOT / "data" / "reaction_eval"


def retrieval(similarity: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    ranks = (similarity > similarity[np.arange(len(truth)), truth][:, None]).sum(axis=1)
    return {"top1": float(np.mean(ranks == 0)), "top5": float(np.mean(ranks < 5)), "mrr": float(np.mean(1 / (ranks + 1)))}


def unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    reactions = json.loads((EVAL / "reactions.json").read_text())
    names = [reaction["name"] for reaction in reactions]
    queries = [json.loads(line) for line in (EVAL / "queries.jsonl").read_text().splitlines() if line.strip()]
    truth = np.array([names.index(query["meme"]) for query in queries])

    model = LocalModel(settings.model_name)
    estimator = PrototypeEstimator(settings.prototype_path, settings.prototype_meta_path)
    layer = estimator.match_layer
    if layer is None:
        raise SystemExit("Calibration has no reaction-matching whitener. Re-run ./scripts/calibrate.sh.")

    def pooled(texts: list[str]) -> torch.Tensor:
        return torch.from_numpy(pool_texts(model, texts)[:, layer])

    moments = pooled([query["text"] for query in queries])
    outside = pooled([reaction["description"] for reaction in reactions])
    voice = pooled([reaction["voice"] for reaction in reactions])
    references = {"description": [outside], "voice": [voice], "both": [outside, voice]}

    def coordinates(vectors: torch.Tensor) -> np.ndarray:
        estimates = [estimator.estimate_pooled(vector[None], layer) for vector in vectors]
        return np.array([[e.valence, e.arousal, e.dominance or 0.0] for e in estimates])

    def whitened(vectors: torch.Tensor) -> np.ndarray:
        return np.stack([estimator.match_embedding(vector[None], layer).numpy() for vector in vectors])

    query_points, query_whitened = coordinates(moments), whitened(moments)
    origin = estimator.match_origin.numpy()
    print(f"{len(queries)} moments over {len(names)} reactions at layer {layer}; "
          f"chance top1 {1 / len(names):.1%}, top5 {5 / len(names):.1%}\n")
    print(f"{'method':<12} {'reactions as':<12} {'top1':>6} {'top5':>6} {'mrr':>6}")
    results = {}
    for name, sets in references.items():
        mean = torch.stack(sets).mean(dim=0)
        # Each reaction's vector is the mean over its descriptions, as place_reactions.py builds it.
        points = coordinates(mean)
        rows = {
            "coordinates": -np.linalg.norm(query_points[:, None] - points[None], axis=2),
            "raw": unit(moments.numpy() - origin) @ unit(mean.numpy() - origin).T,
            "whitened": query_whitened @ whitened(mean).T,
        }
        for method, similarity in rows.items():
            score = retrieval(similarity, truth)
            results[f"{method}/{name}"] = score
            print(f"{method:<12} {name:<12} {score['top1']:>6.1%} {score['top5']:>6.1%} {score['mrr']:>6.3f}")

    output = ROOT / "logs" / "benchmark_reaction_matching.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"model": settings.model_name, "layer": layer, "results": results}, indent=2) + "\n")
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
