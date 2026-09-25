from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from luminote.calibration import class_prototypes, fit_axes, pool_examples, within_class_whitener
from luminote.config import settings
from luminote.examples import TARGETS, load_examples
from luminote.model import LocalModel


def capture(model: LocalModel, prompt: str) -> np.ndarray:
    messages = [
        {"role": "system", "content": "You are a concise, natural conversational partner."},
        {"role": "user", "content": prompt},
    ]
    boundary = model.inspect_boundary(messages)
    return np.stack([vector.numpy() for vector in boundary.by_layer])


def concept_coordinates(concept: dict) -> list[float]:
    point = [concept["valence"], concept["arousal"]]
    if "dominance" in concept:
        point.append(concept["dominance"])
    return point


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Luminote emotion prototypes.")
    parser.add_argument("--model", default=settings.model_name)
    parser.add_argument("--output", type=Path, default=settings.prototype_path)
    parser.add_argument("--metadata", type=Path, default=settings.prototype_meta_path)
    parser.add_argument("--examples", type=Path, default=settings.examples_path)
    args = parser.parse_args()

    data = json.loads(settings.coordinates_path.read_text())
    concept_names = [concept["name"] for concept in data["concepts"]]
    model = LocalModel(args.model)

    # Boundary prototypes: the original prompted method, read at the response-start token.
    neutral_runs = [capture(model, prompt) for prompt in data["neutral_templates"]]
    baseline = np.mean(neutral_runs, axis=0).astype(np.float32)
    prototypes = []
    for name in concept_names:
        print(f"Calibrating boundary prototype for {name}…", flush=True)
        runs = [capture(model, template.format(concept=name)) for template in data["prompt_templates"]]
        prototypes.append(np.mean(runs, axis=0) - baseline)
    arrays = {
        # Stored as [layer, concept, hidden] for cheap runtime layer selection.
        "prototypes": np.stack(prototypes, axis=1).astype(np.float32),
        "baselines": baseline,
    }

    # Span prototypes: mean activation over example displays that never name the feeling.
    examples = []
    if args.examples.exists():
        examples = load_examples(args.examples, set(concept_names))
        print(f"Pooling {len(examples)} calibration examples…", flush=True)
        pooled = pool_examples(model, examples, progress=True)
        span_prototypes, span_baseline = class_prototypes(
            pooled, [example.concept for example in examples], concept_names
        )
        # Targets share the concept baseline so the estimator can read both from one origin.
        target_prototypes, _ = class_prototypes(
            pooled, [example.target for example in examples], TARGETS, baseline=span_baseline
        )
        coordinates = {concept["name"]: concept_coordinates(concept) for concept in data["concepts"]}
        axis_weights, axis_means, axis_intercepts = fit_axes(
            pooled, np.array([coordinates[example.concept] for example in examples])
        )
        # Reaction matching runs at one layer; a full whitener per layer would be too large to store.
        match_layer = min(settings.span_layer, model.layer_count - 1)
        arrays |= {
            "match_whitener": within_class_whitener(pooled[:, match_layer], [example.concept for example in examples]),
            "match_origin": pooled[:, match_layer].mean(axis=0),
            "match_layer": np.array(match_layer),
            "span_prototypes": span_prototypes,
            "span_baselines": span_baseline,
            "target_prototypes": target_prototypes,
            "axis_weights": axis_weights,
            "axis_means": axis_means,
            "axis_intercepts": axis_intercepts,
        }
    else:
        print(f"No calibration examples at {args.examples}; building boundary prototypes only.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    args.metadata.write_text(json.dumps({
        "version": 2,
        "model_name": args.model,
        "layer_count": model.layer_count,
        "hidden_size": int(arrays["prototypes"].shape[-1]),
        "concepts": concept_names,
        "coordinates": [concept_coordinates(concept) for concept in data["concepts"]],
        "targets": list(TARGETS) if examples else [],
        "example_count": len(examples),
        "response_start_anchor": "final token emitted by apply_chat_template(add_generation_prompt=True)",
    }, indent=2) + "\n")
    print(f"Saved {args.output} and {args.metadata}")


if __name__ == "__main__":
    main()
