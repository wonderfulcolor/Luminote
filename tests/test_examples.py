import json

import pytest

from luminote.examples import Example, load_examples, names_feeling, validate


def test_naming_the_feeling_is_detected_across_inflections():
    assert names_feeling("angry", "I was so angry I left.") == "angr"
    assert names_feeling("angry", "The madness of it all.") == "mad"
    assert names_feeling("afraid", "She was frightened of the dark.") == "frighten"


def test_displays_that_do_not_name_the_feeling_pass():
    assert names_feeling("angry", "You made a promise and you broke it. Again.") is None
    assert names_feeling("angry", "The nomad set down his pack.") is None
    assert names_feeling("sad", "She kept his mug on the shelf, unwashed.") is None


def test_validation_reports_unknown_fields():
    concepts = {"calm"}
    assert validate(Example("calm", "limerick", "self", "Breathe."), concepts) == "unknown format 'limerick'"
    assert validate(Example("calm", "dialogue", "nobody", "Breathe."), concepts) == "unknown target 'nobody'"
    assert validate(Example("calm", "dialogue", "self", "Stay calm."), concepts) == "names the feeling ('calm')"
    assert validate(Example("calm", "dialogue", "self", "Breathe. We have time."), concepts) is None


def test_loading_rejects_examples_that_name_the_feeling(tmp_path):
    path = tmp_path / "examples.jsonl"
    rows = [
        {"concept": "sad", "format": "first_person", "target": "self", "text": "I left the lights off."},
        {"concept": "sad", "format": "first_person", "target": "self", "text": "I felt so sad."},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="names the feeling"):
        load_examples(path, {"sad"})
    assert len(load_examples(path, {"sad"}, strict=False)) == 1
