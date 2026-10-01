from __future__ import annotations

import json
import math
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from PIL import Image, ImageSequence

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
SKIPPED_EXTENSIONS = {".heic", ".heics"}
VARIANT_SUFFIX = re.compile(r"(?:[ _-])0*([0-9]+)$")


def normalize_label(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip().casefold()
    value = re.sub(r"[_-]+", " ", value)
    return re.sub(r"\s+", " ", value)


def base_variant_label(value: str) -> str:
    return re.sub(r"\s+0*[0-9]+$", "", normalize_label(value))


def asset_id(stem: str) -> str:
    normalized = normalize_label(stem)
    return re.sub(r"\s+", "_", normalized)


def is_supported(path: Path) -> bool:
    return path.suffix.casefold() in SUPPORTED_EXTENSIONS


def is_animated(path: Path) -> bool:
    with Image.open(path) as image:
        return bool(getattr(image, "is_animated", False) and getattr(image, "n_frames", 1) > 1)


def transparency_metadata(path: Path) -> tuple[bool, bool]:
    """Report encoded alpha support and whether any frame uses transparency."""
    with Image.open(path) as image:
        has_alpha = "A" in image.getbands() or "transparency" in image.info
        if not has_alpha:
            return False, False
        for frame in ImageSequence.Iterator(image):
            alpha_minimum, _ = frame.convert("RGBA").getchannel("A").getextrema()
            if alpha_minimum < 255:
                return True, True
        return True, False


def is_reaction_layer(kind: str, name: str, original_labels: set[str]) -> bool:
    if name.startswith("_") or kind not in {"pixel", "smartobject"}:
        return False
    normalized = normalize_label(name)
    return normalized in original_labels or any(
        base_variant_label(label) == normalized for label in original_labels
    )


def _coordinates(path: Path) -> dict[str, tuple[float, float]]:
    data = json.loads(path.read_text())
    entries = [*data.get("concepts", []), *data.get("diagram_reference_points", [])]
    return {
        normalize_label(entry["name"]): (float(entry["valence"]), float(entry["arousal"]))
        for entry in entries
    }


def _chart_bounds(psd: Any) -> tuple[int, int, int, int]:
    document_area = psd.width * psd.height
    candidates = []
    for layer in psd.descendants():
        if layer.kind != "pixel" or layer.is_group():
            continue
        left, top, right, bottom = tuple(layer.bbox)
        width, height = right - left, bottom - top
        if width <= 0 or height <= 0:
            continue
        area = width * height
        square_error = abs(width - height) / max(width, height)
        if area < document_area * 0.95 and square_error < 0.02:
            candidates.append((area, (left, top, right, bottom), layer.name))
    if not candidates:
        raise ValueError("Could not identify a square circumplex chart layer confidently.")
    return max(candidates, key=lambda item: item[0])[1]


def _match_layer(stem: str, layers: Iterable[Any], used: set[int]) -> tuple[Any | None, str | None]:
    available = [layer for layer in layers if id(layer) not in used]
    stripped = stem.strip()
    exact_case = [layer for layer in available if layer.name.strip() == stripped]
    if len(exact_case) == 1:
        return exact_case[0], None

    normalized = normalize_label(stem)
    exact = [layer for layer in available if normalize_label(layer.name) == normalized]
    if len(exact) == 1:
        return exact[0], None
    if len(exact) > 1:
        return None, f"multiple PSD layers normalize to {normalized!r}"

    base = base_variant_label(stem)
    if base != normalized:
        raw_base = VARIANT_SUFFIX.sub("", stripped)
        base_case = [layer for layer in available if layer.name.strip() == raw_base]
        if len(base_case) == 1:
            return base_case[0], None
        variants = [layer for layer in available if normalize_label(layer.name) == base]
        if len(variants) == 1:
            return variants[0], None
        if len(variants) > 1:
            return None, f"multiple PSD layers match variant base {base!r}"
    return None, None


def _coordinate_from_bounds(
    bounds: tuple[int, int, int, int], chart: tuple[int, int, int, int]
) -> tuple[float, float]:
    left, top, right, bottom = bounds
    chart_left, chart_top, chart_right, chart_bottom = chart
    center_x = (chart_left + chart_right) / 2
    center_y = (chart_top + chart_bottom) / 2
    radius_x = (chart_right - chart_left) / 2
    radius_y = (chart_bottom - chart_top) / 2
    layer_x = (left + right) / 2
    layer_y = (top + bottom) / 2
    valence = max(-1.0, min(1.0, (layer_x - center_x) / radius_x))
    arousal = max(-1.0, min(1.0, (center_y - layer_y) / radius_y))
    return valence, arousal


def build_manifest(reaction_root: Path, coordinates_path: Path) -> dict[str, Any]:
    from psd_tools import PSDImage

    originals = reaction_root / "originals"
    maps = reaction_root / "maps"
    psd_paths = sorted(maps.glob("*.psd"))
    if len(psd_paths) != 1:
        raise ValueError(f"Expected one PSD master in {maps}, found {len(psd_paths)}.")

    files = sorted(
        (path for path in originals.iterdir() if path.is_file() and path.name != ".DS_Store"),
        key=lambda path: (path.name.casefold(), path.name),
    )
    psd = PSDImage.open(psd_paths[0])
    chart = _chart_bounds(psd)
    original_labels = {normalize_label(path.stem) for path in files}
    image_layers = [
        layer
        for layer in psd.descendants()
        if is_reaction_layer(layer.kind, layer.name, original_labels)
    ]
    coordinates = _coordinates(coordinates_path)
    used_layers: set[int] = set()
    assets: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    unmatched: list[dict[str, str]] = []
    ambiguities: list[dict[str, str]] = []
    preview: list[dict[str, Any]] = []

    for path in files:
        extension = path.suffix.casefold()
        layer, ambiguity = _match_layer(path.stem, image_layers, used_layers)
        if ambiguity:
            ambiguities.append({"filename": path.name, "reason": ambiguity})
        if layer is not None:
            used_layers.add(id(layer))

        if extension in SKIPPED_EXTENSIONS:
            skipped.append({
                "filename": path.name,
                "reason": f"unsupported {extension[1:].upper()} format",
                "psdLayerName": layer.name if layer else None,
            })
            continue
        if extension not in SUPPORTED_EXTENSIONS:
            skipped.append({"filename": path.name, "reason": f"unsupported {extension or 'unknown'} format"})
            continue

        normalized = normalize_label(path.stem)
        base = base_variant_label(path.stem)
        coordinate = coordinates.get(normalized)
        coordinate_source = "json" if coordinate is not None else None
        if coordinate is None and base != normalized:
            coordinate = coordinates.get(base)
            coordinate_source = "json" if coordinate is not None else None

        if coordinate is None and layer is not None:
            coordinate = _coordinate_from_bounds(tuple(layer.bbox), chart)
            coordinate_source = "psd"

        animated = is_animated(path)
        has_alpha, has_transparent_pixels = transparency_metadata(path)
        row = {
            "id": asset_id(path.stem),
            "filename": path.name,
            "format": extension[1:],
            "animated": animated,
            "hasAlphaChannel": has_alpha,
            "hasTransparentPixels": has_transparent_pixels,
            "coordinateSource": coordinate_source,
            "valence": round(coordinate[0], 4) if coordinate else None,
            "arousal": round(coordinate[1], 4) if coordinate else None,
            "psdLayerName": layer.name if layer else None,
        }
        preview.append(row)
        if coordinate is None:
            unmatched.append({"filename": path.name, "reason": "no existing JSON coordinate or matching PSD image layer"})
            continue

        assets.append({
            **row,
            "sourcePath": f"originals/{path.name}",
            "publicPath": f"/reaction-assets/{quote(path.name)}",
            "label": path.stem.strip(),
        })

    assets.sort(key=lambda item: item["id"])
    coordinate_groups: dict[tuple[float, float], list[str]] = defaultdict(list)
    for asset in assets:
        coordinate_groups[(asset["valence"], asset["arousal"])].append(asset["id"])
    exact_coordinate_duplicates = [
        {"valence": coordinate[0], "arousal": coordinate[1], "assetIds": sorted(ids)}
        for coordinate, ids in sorted(coordinate_groups.items())
        if len(ids) > 1
    ]
    unused_layers = [
        {"name": layer.name, "kind": layer.kind, "bounds": list(tuple(layer.bbox))}
        for layer in image_layers
        if id(layer) not in used_layers
    ]
    format_counts = Counter(path.suffix.casefold() for path in files)
    return {
        "version": 1,
        "selection": "deterministic Euclidean nearest neighbor; asset ID alphabetical tie-break",
        "assets": assets,
        "skippedAssets": skipped,
        "unmatchedAssets": unmatched,
        "preflight": {
            "sourceRoot": str(reaction_root),
            "psdFilename": psd_paths[0].name,
            "documentSize": [psd.width, psd.height],
            "chartBounds": list(chart),
            "chartCenter": [(chart[0] + chart[2]) / 2, (chart[1] + chart[3]) / 2],
            "originalCount": len(files),
            "formatCounts": dict(sorted(format_counts.items())),
            "activeCount": len(assets),
            "exactCoordinateDuplicateGroups": exact_coordinate_duplicates,
            "transparentActiveAssets": [asset["id"] for asset in assets if asset["hasTransparentPixels"]],
            "opaquePngWebpActiveAssets": [
                asset["id"]
                for asset in assets
                if asset["format"] in {"png", "webp"} and not asset["hasTransparentPixels"]
            ],
            "jsonCoordinateCount": sum(asset["coordinateSource"] == "json" for asset in assets),
            "psdCoordinateCount": sum(asset["coordinateSource"] == "psd" for asset in assets),
            "unmatchedPsdLayers": unused_layers,
            "ambiguities": ambiguities,
            "supportedPreview": preview,
        },
    }


def write_manifest(reaction_root: Path, manifest: dict[str, Any]) -> Path:
    app_assets = reaction_root / "app-assets"
    metadata = reaction_root / "metadata"
    app_assets.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)
    originals = reaction_root / "originals"
    for asset in manifest["assets"]:
        shutil.copy2(originals / asset["filename"], app_assets / asset["filename"])
    output = metadata / "reaction_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return output


def resolve_descriptions(
    asset: dict[str, Any], own: dict[str, list[str]], starters: dict[str, list[str]]
) -> tuple[list[str], str | None]:
    """Descriptions for reaction matching, and where they came from.

    An asset's own descriptions win, looked up by asset ID and then by its label without a
    variant number, so `neutral_02` can share `neutral`'s. Otherwise it borrows the starter for
    the feeling its label names: exactly (`worried`), or as a word within it (`worried kermit`),
    preferring the longest match so `light-hearted` beats `hearted`.
    """
    label = normalize_label(asset.get("label") or asset["id"])
    keys = [asset["id"], label, base_variant_label(label)]
    for key in keys:
        texts = [text for text in own.get(key, []) if text.strip()]
        if texts:
            return texts, "own"
    normalized = {normalize_label(key): key for key in starters}
    for key in (label, base_variant_label(label)):
        if key in normalized:
            return list(starters[normalized[key]]), f"starter:{normalized[key]}"
    base = base_variant_label(label)
    for name in sorted(normalized, key=len, reverse=True):
        if re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", base):
            return list(starters[normalized[name]]), f"starter:{normalized[name]}"
    return [], None
