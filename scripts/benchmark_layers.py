"""Profile every layer on how well example-built prototypes recognise displayed emotion.

The original benchmark scored layers on held-out prompts that still contained the concept word,
which rewards layers that recognise the word. This one separates three questions per layer:

- display:   leave-one-format-out accuracy on calibration examples (generalising across voices)
- named:     do prototypes built from sentences that only *name* the feeling recognise displays?
             High transfer means the layer links the word to what the feeling looks like.
- external:  human-written text from other sources (EmoBank valence/arousal/dominance ratings,
             GoEmotions labels), downloaded on first use and never redistributed.

External coordinates are scored twice: from prototype matching, and from the linear axis probe
fitted on the same examples, which the app uses for span readings when it is calibrated.

Layers that hold up out of source are the ones reading what is shown rather than what is named.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import urllib.request
from pathlib import Path

import numpy as np
import torch

from luminote.affect import PrototypeEstimator, summarize
from luminote.calibration import class_prototypes, fit_axes, pool_examples, pool_texts
from luminote.config import ROOT, settings
from luminote.examples import FORMATS, load_examples
from luminote.model import LocalModel

EXTERNAL = ROOT / "data" / "external"
EMOBANK_URL = "https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus/emobank.csv"
GOEMOTIONS_URL = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/{}"
# GoEmotions labels with a clear counterpart among Luminote's concepts.
GOEMOTIONS_MAP = {
    "joy": "joyful", "excitement": "excited", "amusement": "delighted",
    "anger": "angry", "annoyance": "angry", "fear": "afraid", "nervousness": "tense",
    "sadness": "sad", "disappointment": "sad", "grief": "miserable", "relief": "relaxed",
    "confusion": "uncertain", "curiosity": "attentive", "neutral": "neutral",
}
NAMED_TEMPLATES = ("I feel {concept}.", "{concept}.", "Right now I am {concept}.", "That makes me {concept}.")


def fetch(url: str, name: str) -> str:
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    path = EXTERNAL / name
    if not path.exists():
        print(f"Downloading {url}", flush=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    return path.read_text(encoding="utf-8")


def load_emobank(limit: int, rng: random.Random) -> list[tuple[str, float, float, float]]:
    rows = [
        (row["text"], float(row["V"]), float(row["A"]), float(row["D"]))
        for row in csv.DictReader(io.StringIO(fetch(EMOBANK_URL, "emobank.csv")))
        if row["split"] == "test" and row["text"].strip()
    ]
    rng.shuffle(rows)
    return rows[:limit]


def load_goemotions(limit: int, rng: random.Random) -> list[tuple[str, str]]:
    names = fetch(GOEMOTIONS_URL.format("emotions.txt"), "goemotions_emotions.txt").split()
    by_concept: dict[str, list[str]] = {}
    for line in fetch(GOEMOTIONS_URL.format("test.tsv"), "goemotions_test.tsv").splitlines():
        text, labels, _ = line.split("\t")
        ids = labels.split(",")
        if len(ids) != 1:
            continue
        concept = GOEMOTIONS_MAP.get(names[int(ids[0])])
        if concept:
            by_concept.setdefault(concept, []).append(text)
    # Balance concepts so neutral, which dominates GoEmotions, does not dominate the score.
    per_concept = max(1, limit // len(by_concept))
    rows = []
    for concept, texts in sorted(by_concept.items()):
        rng.shuffle(texts)
        rows.extend((text, concept) for text in texts[:per_concept])
    return rows


def cosine(vectors: np.ndarray, prototypes: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    centered = vectors - baseline
    centered /= np.linalg.norm(centered, axis=-1, keepdims=True).clip(1e-8)
    normalized = prototypes / np.linalg.norm(prototypes, axis=-1, keepdims=True).clip(1e-8)
    return centered @ normalized.T


def pearson(left: list[float], right: list[float]) -> float:
    if np.std(left) < 1e-9 or np.std(right) < 1e-9:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def classification(scores: np.ndarray, truth: np.ndarray, coordinates: np.ndarray) -> dict:
    predicted = scores.argmax(axis=1)
    distance = np.linalg.norm(coordinates[predicted, :2] - coordinates[truth, :2], axis=1)
    return {
        "accuracy": float(np.mean(predicted == truth)),
        # A neighbouring feeling (within the estimator's mode radius) counts as a near miss.
        "neighbourhood": float(np.mean(distance <= 0.6)),
        "valence_sign": float(np.mean(np.sign(coordinates[predicted, 0]) == np.sign(coordinates[truth, 0]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--emobank", type=int, default=600, help="EmoBank test sentences (0 to skip)")
    parser.add_argument("--goemotions", type=int, default=560, help="GoEmotions test comments (0 to skip)")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", type=Path, default=ROOT / "logs" / "benchmark_layers.json")
    args = parser.parse_args()
    rng = random.Random(args.seed)

    data = json.loads(settings.coordinates_path.read_text())
    concepts = [concept["name"] for concept in data["concepts"]]
    coordinates = np.array(
        [[c["valence"], c["arousal"], c.get("dominance", 0.0)] for c in data["concepts"]], dtype=np.float32
    )
    index = {name: position for position, name in enumerate(concepts)}
    model = LocalModel(settings.model_name)
    layers = range(model.layer_count)
    results: dict[int, dict] = {layer: {} for layer in layers}

    examples = load_examples(settings.examples_path, set(concepts))
    print(f"Pooling {len(examples)} calibration examples…", flush=True)
    pooled = pool_examples(model, examples, progress=True)
    labels = np.array([example.concept for example in examples])
    truth = np.array([index[label] for label in labels])
    formats = np.array([example.format for example in examples])
    prototypes, baseline = class_prototypes(pooled, labels, concepts)
    axis_weights, axis_means, axis_intercepts = fit_axes(pooled, coordinates[truth])

    def axis_predict(vectors: np.ndarray, layer: int) -> np.ndarray:
        return np.clip((vectors - axis_means[layer]) @ axis_weights[layer] + axis_intercepts[layer], -1, 1)

    print("Leave-one-format-out…", flush=True)
    for layer in layers:
        scores = np.zeros((len(examples), len(concepts)), dtype=np.float32)
        for held_out in FORMATS:
            test = formats == held_out
            if not test.any():
                continue
            fold, fold_baseline = class_prototypes(pooled[~test], labels[~test], concepts)
            scores[test] = cosine(pooled[test, layer], fold[layer], fold_baseline[layer])
        results[layer]["display"] = classification(scores, truth, coordinates)

    print("Named-word transfer…", flush=True)
    named_texts = [template.format(concept=name) for name in concepts for template in NAMED_TEMPLATES]
    named_labels = [name for name in concepts for _ in NAMED_TEMPLATES]
    named = pool_texts(model, named_texts)
    named_prototypes, named_baseline = class_prototypes(named, named_labels, concepts)
    for layer in layers:
        scores = cosine(pooled[:, layer], named_prototypes[layer], named_baseline[layer])
        results[layer]["named_to_display"] = classification(scores, truth, coordinates)["accuracy"]

    if args.emobank:
        rows = load_emobank(args.emobank, rng)
        print(f"EmoBank: pooling {len(rows)} sentences…", flush=True)
        vectors = pool_texts(model, (row[0] for row in rows), progress=True)
        gold = np.array([row[1:] for row in rows])
        tensor_coordinates = torch.from_numpy(coordinates)
        for layer in layers:
            scores = torch.from_numpy(cosine(vectors[:, layer], prototypes[layer], baseline[layer]))
            estimates = [summarize(score, concepts, tensor_coordinates) for score in scores]
            predicted = np.array([[e.valence, e.arousal, e.dominance] for e in estimates])
            linear = axis_predict(vectors[:, layer], layer)
            results[layer]["emobank"] = {
                axis: round(pearson(predicted[:, position].tolist(), gold[:, position].tolist()), 4)
                for position, axis in enumerate(("valence", "arousal", "dominance"))
            }
            results[layer]["emobank_axis"] = {
                axis: round(pearson(linear[:, position].tolist(), gold[:, position].tolist()), 4)
                for position, axis in enumerate(("valence", "arousal", "dominance"))
            }

    if args.goemotions:
        rows = load_goemotions(args.goemotions, rng)
        print(f"GoEmotions: pooling {len(rows)} comments…", flush=True)
        vectors = pool_texts(model, (row[0] for row in rows), progress=True)
        go_truth = np.array([index[row[1]] for row in rows])
        for layer in layers:
            scores = cosine(vectors[:, layer], prototypes[layer], baseline[layer])
            results[layer]["goemotions"] = classification(scores, go_truth, coordinates)
            signed = go_truth[coordinates[go_truth, 0] != 0]
            linear = axis_predict(vectors[coordinates[go_truth, 0] != 0, layer], layer)
            results[layer]["goemotions"]["axis_valence_sign"] = float(
                np.mean(np.sign(linear[:, 0]) == np.sign(coordinates[signed, 0]))
            )

    # The original benchmark, for comparison: named held-out prompts read at the response boundary.
    if settings.prototype_path.exists():
        estimator = PrototypeEstimator(settings.prototype_path, settings.prototype_meta_path)
        print("Original boundary benchmark…", flush=True)
        for layer in layers:
            results[layer]["boundary_named_quadrant"] = 0
        count = 0
        for concept in data["concepts"]:
            for template in data["held_out_templates"]:
                boundary = model.inspect_boundary([{"role": "user", "content": template.format(concept=concept["name"])}])
                count += 1
                for layer in layers:
                    estimate = estimator.estimate(boundary.by_layer[layer], layer)
                    results[layer]["boundary_named_quadrant"] += int(
                        estimate.valence * concept["valence"] >= 0 and estimate.arousal * concept["arousal"] >= 0
                    )
        for layer in layers:
            results[layer]["boundary_named_quadrant"] /= count

    header = f"{'layer':>5}  {'display':>7}  {'nbhd':>5}  {'named→disp':>10}"
    if args.emobank:
        header += f"  {'EB r(V)':>7}  {'EB r(A)':>7}  {'EB r(D)':>7}  {'ax r(V)':>7}  {'ax r(A)':>7}  {'ax r(D)':>7}"
    if args.goemotions:
        header += f"  {'GoE acc':>7}  {'GoE val':>7}  {'ax val':>7}"
    header += f"  {'old bnd':>7}"
    print("\n" + header)
    for layer in layers:
        row = results[layer]
        line = f"{layer:>5}  {row['display']['accuracy']:>7.1%}  {row['display']['neighbourhood']:>5.0%}  {row['named_to_display']:>10.1%}"
        if args.emobank:
            line += "".join(f"  {row[key][axis]:>7.3f}" for key in ("emobank", "emobank_axis") for axis in ("valence", "arousal", "dominance"))
        if args.goemotions:
            goemotions = row["goemotions"]
            line += f"  {goemotions['accuracy']:>7.1%}  {goemotions['valence_sign']:>7.1%}  {goemotions['axis_valence_sign']:>7.1%}"
        line += f"  {row.get('boundary_named_quadrant', float('nan')):>7.1%}"
        print(line)

    def out_of_source(layer: int) -> float:
        row = results[layer]
        parts = [row["display"]["accuracy"]]
        if args.emobank:
            parts.append((row["emobank_axis"]["valence"] + row["emobank_axis"]["arousal"]) / 2)
        if args.goemotions:
            parts.append(row["goemotions"]["accuracy"])
        return float(np.mean(parts))

    best = max(layers, key=out_of_source)
    print(f"\nRecommended layer for span readings: {best}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "model": settings.model_name,
        "example_count": len(examples),
        "seed": args.seed,
        "recommended_layer": best,
        "layers": results,
    }, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
