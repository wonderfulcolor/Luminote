"""Draft candidate descriptions of your reaction images with Claude, for you to review.

Reaction matching compares each reply with descriptions of each reaction image. This script
shows each image in the manifest to Claude and asks for two candidates: a short description of
what the image means, and the reaction in its own voice. It writes them to

    luminote-reactions/metadata/reaction_descriptions.candidates.json

and never touches reaction_descriptions.json. Review the candidates, edit what is off, copy the
ones you want into reaction_descriptions.json, then run scripts/place_reactions.py.

This sends your images to the Anthropic API, so only run it on images you are comfortable
sharing there. It needs `pip install -e '.[draft]'` and an Anthropic credential (ANTHROPIC_API_KEY,
or a profile from `ant auth login`). Use --dry-run to see what would be sent without calling it.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

from PIL import Image

from luminote.config import ROOT
from luminote.reactions import base_variant_label

METADATA = ROOT / "luminote-reactions" / "metadata"
ASSETS = ROOT / "luminote-reactions" / "app-assets"
MODEL = "claude-opus-5"
# Opus 5's safety classifiers can occasionally decline; "default" re-runs a declined request on
# Anthropic's recommended fallback model instead of returning the refusal.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_SIDE = 1024

SCHEMA = {
    "type": "object",
    "properties": {
        "voice": {"type": "string"},
        "image": {"type": "string"},
    },
    "required": ["voice", "image"],
    "additionalProperties": False,
}

INSTRUCTIONS = """This image is a reaction image someone would post in a chat. A system picks reaction images by comparing \
a chat reply with short descriptions of each image, so the descriptions must capture what posting this image *says*.

Write two descriptions:
- "voice": 12-35 words, first person, the inner monologue of someone posting this reaction. Express its stance \
directly. Say what distinguishes it from similar reactions (for example, feigned shock at a predictable outcome is \
different from genuine surprise). No references to the image, characters, or memes.
- "image": 12-35 words, what the image shows and what it means when posted, in the form "<what it shows>: <what it \
means>".

The file is labelled "{label}", which may hint at the intended feeling but may be wrong or vague; trust the image.

Examples of the style, for other reactions:
{examples}"""


def image_block(path: Path) -> dict:
    """First frame, downscaled and re-encoded as PNG so every format and size is accepted."""
    with Image.open(path) as image:
        image.seek(0)
        frame = image.convert("RGBA")
    frame.thumbnail((MAX_SIDE, MAX_SIDE))
    buffer = io.BytesIO()
    frame.save(buffer, format="PNG")
    data = base64.standard_b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def style_examples(count: int = 3) -> str:
    reactions = json.loads((ROOT / "data" / "reaction_eval" / "reactions.json").read_text())
    picked = [reaction for reaction in reactions if reaction["name"] in {"this is fine", "surprised pikachu", "success kid"}]
    return "\n".join(
        json.dumps({"voice": reaction["voice"], "image": reaction["description"]}) for reaction in picked[:count]
    )


def asset_path(asset: dict) -> Path | None:
    name = Path(asset.get("publicPath") or "").name
    path = ASSETS / name
    return path if name and path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=METADATA / "reaction_manifest.json")
    parser.add_argument("--output", type=Path, default=METADATA / "reaction_descriptions.candidates.json")
    parser.add_argument("--only", nargs="*", help="asset IDs to draft (default: all)")
    parser.add_argument("--overwrite", action="store_true", help="redraft assets that already have candidates")
    parser.add_argument("--dry-run", action="store_true", help="list what would be sent without calling the API")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    candidates = json.loads(args.output.read_text()) if args.output.exists() else {}
    assets = [asset for asset in manifest["assets"] if not args.only or asset["id"] in args.only]
    todo = [asset for asset in assets if args.overwrite or asset["id"] not in candidates]
    missing = [asset["id"] for asset in todo if asset_path(asset) is None]
    todo = [asset for asset in todo if asset_path(asset) is not None]
    if missing:
        print(f"No local image for: {', '.join(missing)}")
    print(f"{len(todo)} images to draft with {MODEL}.")
    if args.dry_run or not todo:
        for asset in todo:
            label = base_variant_label(asset.get("label") or asset["id"])
            print(f"  would send {asset_path(asset).relative_to(ROOT)} (label {label!r})")
        return

    import anthropic  # optional dependency: pip install -e '.[draft]'

    client = anthropic.Anthropic()
    examples = style_examples()
    for asset in todo:
        label = base_variant_label(asset.get("label") or asset["id"])
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{
                "role": "user",
                "content": [
                    image_block(asset_path(asset)),
                    {"type": "text", "text": INSTRUCTIONS.format(label=label, examples=examples)},
                ],
            }],
        )
        if response.stop_reason == "refusal":
            print(f"{asset['id']}: declined, skipped")
            continue
        text = next((block.text for block in response.content if block.type == "text"), "")
        drafted = json.loads(text)
        candidates[asset["id"]] = {"voice": drafted["voice"], "image": drafted["image"], "model": response.model}
        # Save after each image so an interruption keeps what was already drafted.
        args.output.write_text(json.dumps(candidates, indent=2, ensure_ascii=False) + "\n")
        print(f"{asset['id']}: {drafted['voice']}")

    print(f"\nWrote {len(candidates)} candidates to {args.output}")
    print("Review them, copy the ones you want into reaction_descriptions.json, then run scripts/place_reactions.py.")


if __name__ == "__main__":
    main()
