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

Coordinates must be finite numbers from `-1` to `1`. An optional `"dominance"` coordinate (sense of control, `-1` to `1`) separates reactions that share valence and arousal.

## How a reaction is chosen

1. **Activation match (preferred).** If reactions have been indexed from descriptions (below), the server compares the reply with each reaction's descriptions in the model's activation space, whitened so that differences in topic and wording count for less than differences in stance. The frontend takes the most similar reaction, nudging back ones shown in the last few turns.
2. **Coordinates (fallback).** Otherwise, the nearest reaction by valence and arousal, with dominance at half weight when both sides have it, the same recency nudge, and asset-ID alphabetical tie-breaking.

Coordinates alone keep very little of what separates one reaction meme from another. On the evaluation set in `data/reaction_eval/`, coordinate selection picked the intended meme about as often as chance, while activation matching found it first roughly 40% of the time and in its top five roughly two thirds of the time (`scripts/benchmark_reaction_matching.py`).

## Indexing reactions by description

Add `luminote-reactions/metadata/reaction_descriptions.json`, with one or more descriptions per asset ID:

```json
{
  "my_reaction": [
    "Everything around me is going wrong and I am going to keep smiling and act like it is under control.",
    "Dog sitting calmly in a burning room: forced calm while everything falls apart."
  ]
}
```

**Write at least one description in the reaction's own voice**, as the inner monologue of someone posting it. Descriptions of the image alone ("a dog in a burning room") mostly read to the model as "this is a funny meme" and match poorly. Say what distinguishes the reaction from its neighbours: surprised Pikachu is feigned shock at a predictable result of your own actions, not genuine surprise.

### Starters

`luminote-reactions/metadata/reaction_descriptions.starter.json` holds two voice descriptions for each of the 58 feelings on the chart. An asset without its own descriptions borrows the starter for the feeling its label names, exactly (`worried`, `neutral_02`) or as a word within it (`worried_kermit`). Starters make matching work with no writing at all, but they describe the feeling, not the image, so the gains come from replacing them with descriptions of each reaction.

### Drafting descriptions from the images

`scripts/draft_reaction_descriptions.py` shows each image to Claude and writes two candidates per image (one in the reaction's voice, one of the image) to `reaction_descriptions.candidates.json` for you to review. It never edits `reaction_descriptions.json`; copy across what you keep.

**It sends your images to the Anthropic API.** Only run it on images you are comfortable sharing there. It needs the optional dependency and an Anthropic credential:

```bash
.venv/bin/pip install -e '.[draft]'
PYTHONPATH=. .venv/bin/python scripts/draft_reaction_descriptions.py --dry-run   # see what would be sent
PYTHONPATH=. .venv/bin/python scripts/draft_reaction_descriptions.py --only my_reaction
```

Animated images are sent as their first frame, so describe the ending yourself when the punchline is in the motion.

### Indexing

Then run:

```bash
PYTHONPATH=. .venv/bin/python scripts/place_reactions.py          # index, and add probe coordinates alongside
PYTHONPATH=. .venv/bin/python scripts/place_reactions.py --apply  # also use probe coordinates on the circumplex
```

This writes `reaction_embeddings.npz` (model- and calibration-specific, ignored by Git) next to the manifest. Restart the server afterwards, and re-run it after recalibrating. Descriptions are kept apart from the manifest so they survive rebuilding it from a PSD.

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
