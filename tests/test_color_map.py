from PIL import Image

from luminote.color_map import ColorMap


def test_confidence_mutes_color(tmp_path):
    path = tmp_path / "wheel.jpg"
    Image.new("RGB", (919, 849), (255, 0, 0)).save(path)
    mapper = ColorMap(path)
    low = mapper.display_color(0, 0, 0)["hex"]
    high = mapper.display_color(0, 0, 1)["hex"]
    assert low != high
    assert mapper.raw_rgb(0, 0)[0] > 200

