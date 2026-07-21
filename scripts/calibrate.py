from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from luminote.config import settings
from luminote.model import LocalModel


def capture(model: LocalModel, prompt: str) -> np.ndarray:
    messages = [
        {"role": "system", "content": "You are a concise, natural conversational partner."},
        {"role": "user", "content": prompt},
    ]
    boundary = model.inspect_boundary(messages)
    return np.stack([vector.numpy() for vector in boundary.by_layer])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Luminote emotion prototypes.")
    parser.add_argument("--model", default=settings.model_name)
    parser.add_argument("--output", type=Path, default=settings.prototype_path)
    parser.add_argument("--metadata", type=Path, default=settings.prototype_meta_path)
    args = parser.parse_args()

    data = json.loads(settings.coordinates_path.read_text())
    model = LocalModel(args.model)
    neutral_runs = [capture(model, prompt) for prompt in data["neutral_templates"]]
    baseline = np.mean(neutral_runs, axis=0).astype(np.float32)
    prototypes = []
    concepts = []
    coordinates = []

    for concept in data["concepts"]:
        name = concept["name"]
        print(f"Calibrating {name}…", flush=True)
        runs = [capture(model, template.format(concept=name)) for template in data["prompt_templates"]]
        prototypes.append(np.mean(runs, axis=0) - baseline)
        concepts.append(name)
        coordinates.append([concept["valence"], concept["arousal"]])

    # Stored as [layer, concept, hidden] for cheap runtime layer selection.
    prototype_array = np.stack(prototypes, axis=1).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, prototypes=prototype_array, baselines=baseline)
    args.metadata.write_text(json.dumps({
        "version": 1,
        "model_name": args.model,
        "layer_count": model.layer_count,
        "hidden_size": int(prototype_array.shape[-1]),
        "concepts": concepts,
        "coordinates": coordinates,
        "response_start_anchor": "final token emitted by apply_chat_template(add_generation_prompt=True)",
    }, indent=2) + "\n")
    print(f"Saved {args.output} and {args.metadata}")


if __name__ == "__main__":
    main()

