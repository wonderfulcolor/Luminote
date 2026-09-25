"""Place reaction images by what the model reads in a description of each one.

Hand-placed coordinates say where a person thinks a meme sits. Reading a short description of
the meme through the same probe as the conversation puts it where the *model* reads it, so
meme and reply are compared in one space. Descriptions live in their own file so they survive
rebuilding the manifest from the PSD:

    luminote-reactions/metadata/reaction_descriptions.json
    {"this_is_fine": "Dog sitting calmly in a burning room, captioned 'this is fine': forced calm while everything falls apart."}

Without --apply the probe coordinates are added alongside the existing ones for comparison.
With --apply they replace valence/arousal/dominance and the originals are kept as chart*.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from luminote.affect import PrototypeEstimator
from luminote.calibration import pool_texts
from luminote.config import ROOT, settings
from luminote.model import LocalModel

METADATA = ROOT / "luminote-reactions" / "metadata"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=METADATA / "reaction_manifest.json")
    parser.add_argument("--descriptions", type=Path, default=METADATA / "reaction_descriptions.json")
    parser.add_argument("--layer", type=int, default=settings.span_layer)
    parser.add_argument("--apply", action="store_true", help="use probe coordinates for selection")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    descriptions: dict[str, str] = json.loads(args.descriptions.read_text()) if args.descriptions.exists() else {}
    assets = [asset for asset in manifest["assets"] if descriptions.get(asset["id"], "").strip()]
    if not assets:
        print(f"No assets in {args.manifest.name} have a description in {args.descriptions.name}.")
        return

    model = LocalModel(settings.model_name)
    estimator = PrototypeEstimator(settings.prototype_path, settings.prototype_meta_path)
    pooled = pool_texts(model, (descriptions[asset["id"]] for asset in assets))
    for asset, vectors in zip(assets, pooled, strict=True):
        estimate = estimator.estimate_pooled(torch.from_numpy(vectors[args.layer])[None], args.layer)
        asset["description"] = descriptions[asset["id"]]
        asset["probeValence"], asset["probeArousal"] = estimate.valence, estimate.arousal
        asset["probeDominance"] = estimate.dominance
        asset["probeLayer"] = args.layer
        if args.apply:
            asset.setdefault("chartValence", asset["valence"])
            asset.setdefault("chartArousal", asset["arousal"])
            asset["valence"], asset["arousal"] = estimate.valence, estimate.arousal
            asset["dominance"] = estimate.dominance
            asset["coordinateSource"] = "probe_description"
        print(f"{asset['id']:<32} {estimate.valence:+.3f} {estimate.arousal:+.3f} {estimate.dominance or 0:+.3f}  {estimate.matches[0].concept}")

    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Updated {len(assets)} assets in {args.manifest}")


if __name__ == "__main__":
    main()
