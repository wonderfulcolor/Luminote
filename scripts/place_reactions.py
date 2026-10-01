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
scripts/draft_reaction_descriptions.py can draft candidates from the images for you to review.

Assets without their own descriptions borrow a starter from reaction_descriptions.starter.json
for the feeling their label names (`worried_kermit` uses "worried"). Starters make matching work
out of the box but know nothing about the particular image, so replace them as you go.

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
from luminote.reactions import resolve_descriptions

METADATA = ROOT / "luminote-reactions" / "metadata"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=METADATA / "reaction_manifest.json")
    parser.add_argument("--descriptions", type=Path, default=METADATA / "reaction_descriptions.json")
    parser.add_argument("--starters", type=Path, default=METADATA / "reaction_descriptions.starter.json")
    parser.add_argument("--index", type=Path, default=settings.reaction_index_path)
    parser.add_argument("--apply", action="store_true", help="use probe coordinates on the circumplex")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    raw = json.loads(args.descriptions.read_text()) if args.descriptions.exists() else {}
    own = {key: [value] if isinstance(value, str) else list(value) for key, value in raw.items()}
    starters = json.loads(args.starters.read_text()) if args.starters.exists() else {}
    resolved = {asset["id"]: resolve_descriptions(asset, own, starters) for asset in manifest["assets"]}
    unmatched = [asset_id for asset_id, (texts, _) in resolved.items() if not texts]
    if unmatched:
        print(f"No description or starter for: {', '.join(unmatched)}")
    assets = [asset for asset in manifest["assets"] if resolved[asset["id"]][0]]
    if not assets:
        print(f"No assets in {args.manifest.name} could be described.")
        return
    borrowed = sum(source != "own" for _, source in resolved.values() if source)
    print(f"{len(assets) - borrowed} assets with their own descriptions, {borrowed} using starters.")

    model = LocalModel(settings.model_name)
    estimator = PrototypeEstimator(settings.prototype_path, settings.prototype_meta_path)
    if estimator.match_layer is None:
        raise SystemExit("Calibration has no reaction-matching whitener. Re-run ./scripts/calibrate.sh.")
    layer = estimator.match_layer

    embeddings = []
    for asset in assets:
        texts, source = resolved[asset["id"]]
        pooled = torch.from_numpy(pool_texts(model, texts)[:, layer])
        embeddings.append(estimator.match_embedding(pooled, layer).numpy())
        estimate = estimator.estimate_pooled(pooled, layer)
        asset["descriptions"] = texts
        asset["descriptionSource"] = source
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
