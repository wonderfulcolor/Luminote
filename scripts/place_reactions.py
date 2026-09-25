"""Place and index reaction images by what the model reads in descriptions of them.

Each reaction gets one or more short descriptions in its own file, so they survive rebuilding
the manifest from the PSD:

    luminote-reactions/metadata/reaction_descriptions.json
    {
      "this_is_fine": [
        "Everything around me is going wrong and I am going to keep smiling and act like it is under control.",
        "Dog sitting calmly in a burning room: forced calm while everything falls apart."
      ]
    }

Descriptions written in the reaction's own voice ("I lost, in public, and everyone saw") match
far better than descriptions of the image, which mostly read as "this is a funny meme".

This script writes two things:
- reaction_embeddings.npz, which the server uses to rank reactions against each reply; and
- probe coordinates in the manifest, for the circumplex display. Without --apply they sit
  alongside the existing coordinates; with --apply they replace them (originals kept as chart*).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from luminote.affect import PrototypeEstimator
from luminote.calibration import pool_texts
from luminote.config import ROOT, settings
from luminote.model import LocalModel
from luminote.reaction_matching import ReactionIndex

METADATA = ROOT / "luminote-reactions" / "metadata"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=METADATA / "reaction_manifest.json")
    parser.add_argument("--descriptions", type=Path, default=METADATA / "reaction_descriptions.json")
    parser.add_argument("--index", type=Path, default=settings.reaction_index_path)
    parser.add_argument("--apply", action="store_true", help="use probe coordinates on the circumplex")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    raw = json.loads(args.descriptions.read_text()) if args.descriptions.exists() else {}
    descriptions = {key: [value] if isinstance(value, str) else list(value) for key, value in raw.items()}
    assets = [asset for asset in manifest["assets"] if any(text.strip() for text in descriptions.get(asset["id"], []))]
    if not assets:
        print(f"No assets in {args.manifest.name} have a description in {args.descriptions.name}.")
        return

    model = LocalModel(settings.model_name)
    estimator = PrototypeEstimator(settings.prototype_path, settings.prototype_meta_path)
    if estimator.match_layer is None:
        raise SystemExit("Calibration has no reaction-matching whitener. Re-run ./scripts/calibrate.sh.")
    layer = estimator.match_layer

    embeddings = []
    for asset in assets:
        texts = [text for text in descriptions[asset["id"]] if text.strip()]
        pooled = torch.from_numpy(pool_texts(model, texts)[:, layer])
        embeddings.append(estimator.match_embedding(pooled, layer).numpy())
        estimate = estimator.estimate_pooled(pooled, layer)
        asset["descriptions"] = texts
        asset["probeValence"], asset["probeArousal"] = estimate.valence, estimate.arousal
        asset["probeDominance"] = estimate.dominance
        asset["probeLayer"] = layer
        if args.apply:
            asset.setdefault("chartValence", asset["valence"])
            asset.setdefault("chartArousal", asset["arousal"])
            asset["valence"], asset["arousal"] = estimate.valence, estimate.arousal
            asset["dominance"] = estimate.dominance
            asset["coordinateSource"] = "probe_description"
        print(f"{asset['id']:<32} {estimate.valence:+.3f} {estimate.arousal:+.3f} {estimate.dominance or 0:+.3f}  {estimate.matches[0].concept}")

    ReactionIndex([asset["id"] for asset in assets], np.stack(embeddings), layer, model.model_name).save(args.index)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Indexed {len(assets)} reactions at layer {layer} in {args.index}")
    print("Restart the server to use the new index.")


if __name__ == "__main__":
    main()
