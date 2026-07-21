from pathlib import Path

from luminote.reactions import (
    asset_id,
    base_variant_label,
    is_reaction_layer,
    is_supported,
    normalize_label,
    transparency_metadata,
)

from PIL import Image


def test_label_normalization_is_safe_and_case_insensitive():
    assert normalize_label("  Mildly_Skeptical  ") == "mildly skeptical"
    assert normalize_label("worried-kermit") == "worried kermit"


def test_duplicate_variants_remain_distinct_but_share_a_base_label():
    ids = [asset_id(name) for name in ("neutral", "neutral_02", "neutral_03", "neutral_04")]
    assert len(set(ids)) == 4
    assert base_variant_label("neutral_04") == "neutral"


def test_heic_and_heics_are_not_browser_supported():
    assert not is_supported(Path("reaction.HEIC"))
    assert not is_supported(Path("reaction.HEICS"))
    assert is_supported(Path("reaction.webp"))


def test_text_and_unmatched_psd_layers_are_ignored():
    originals = {"neutral", "happy"}
    assert not is_reaction_layer("type", "neutral", originals)
    assert not is_reaction_layer("smartobject", "chart background", originals)
    assert not is_reaction_layer("smartobject", "_annotation", originals)
    assert is_reaction_layer("smartobject", "Happy", originals)


def test_transparency_metadata_distinguishes_real_alpha_from_opaque_png(tmp_path):
    transparent = tmp_path / "transparent.png"
    opaque = tmp_path / "opaque.png"
    Image.new("RGBA", (2, 2), (255, 0, 0, 0)).save(transparent)
    Image.new("RGB", (2, 2), (255, 255, 255)).save(opaque)

    assert transparency_metadata(transparent) == (True, True)
    assert transparency_metadata(opaque) == (False, False)
