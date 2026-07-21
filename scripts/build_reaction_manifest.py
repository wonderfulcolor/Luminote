from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from luminote.config import ROOT
from luminote.reactions import build_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local Luminote reaction-asset manifest.")
    parser.add_argument("--reaction-root", type=Path, default=ROOT / "luminote-reactions")
    parser.add_argument("--coordinates", type=Path, default=ROOT / "data" / "emotion_coordinates.json")
    args = parser.parse_args()
    manifest = build_manifest(args.reaction_root, args.coordinates)
    output = write_manifest(args.reaction_root, manifest)
    print(f"Wrote {output}")
    print(f"Active assets: {len(manifest['assets'])}")
    print(f"Skipped assets: {len(manifest['skippedAssets'])}")
    print(f"Unmatched supported assets: {len(manifest['unmatchedAssets'])}")


if __name__ == "__main__":
    main()
