# Public asset guidance

The repository intentionally contains no reaction images, source PSD, third-party memes, or copied circumplex artwork.

## Adding licensed reaction images

Place public-safe browser images in `luminote-reactions/app-assets/`. Supported browser formats are PNG, JPEG, GIF, and WebP. Then edit `luminote-reactions/metadata/reaction_manifest.json` and add entries like:

```json
{
  "id": "my_reaction",
  "label": "My reaction",
  "filename": "my-reaction.png",
  "publicPath": "/reaction-assets/my-reaction.png",
  "format": "png",
  "animated": false,
  "hasAlphaChannel": true,
  "hasTransparentPixels": true,
  "coordinateSource": "author_assigned",
  "valence": 0.4,
  "arousal": 0.2
}
```

Coordinates must be finite numbers from `-1` to `1`. Selection is deterministic Euclidean nearest neighbor with asset-ID alphabetical tie-breaking.

Only add an asset when you can document its creator, source, and license. Do not assume that a meme, reaction GIF, television frame, celebrity image, cartoon character, or search-engine result is safe to redistribute.

## Optional PSD builder

`scripts/build_reaction_manifest.py` remains available for a collection you own. It expects:

```text
luminote-reactions/
  originals/
  maps/
    one-master-file.psd
```

Install development dependencies with `./scripts/setup.sh`, then run:

```bash
.venv/bin/python scripts/build_reaction_manifest.py
```

The repository ignore rules deliberately prevent `originals/`, `maps/`, and generated browser assets from being committed. Review licenses and intentionally stage only distributable files if you choose to publish assets.

## Color map

`scripts/generate_public_color_map.py` creates the public build's unlabeled procedural wheel. It does not reproduce or distribute the private reference chart. The generated PNG is ignored because setup can reproduce it deterministically.
