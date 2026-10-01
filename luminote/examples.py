"""Calibration examples: displays of an emotion that never name it.

Prototypes built from prompts such as "respond as someone feeling {concept}" partly encode the
presence of the concept word itself. Building them from examples that show the emotion without
naming it, rotated across several voices, keeps the probe on the display rather than the label.

Adapted from the prompt-format rotation and lexical exclusion used when training HatCat lenses:
https://github.com/p0ss/HatCat
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

FORMATS = ("assistant_reply", "user_message", "first_person", "third_person", "dialogue", "situation")
TARGETS = ("self", "other", "situation")

# Stems that name each feeling. An example containing one is labelling rather than displaying.
NAMED_STEMS: dict[str, tuple[str, ...]] = {
    "joyful": ("joy", "happ", "glad", "cheerful", "elat"),
    "excited": ("excit", "thrill", "pumped", "hyped"),
    "enthusiastic": ("enthus", "eager", "keen", "passionate"),
    "delighted": ("delight", "pleased", "overjoy", "tickled"),
    "angry": ("angr", "anger", "furious", "fury", "mad", "rage", "enrag", "irritat", "annoy", "livid"),
    "tense": ("tense", "tension", "stress", "on edge", "uptight"),
    "afraid": ("afraid", "fear", "scare", "frighten", "terrif", "petrif"),
    "distressed": ("distress", "anguish", "upset", "overwhelm", "panick"),
    "sad": ("sad", "unhapp", "sorrow", "melanchol", "heartbroken"),
    "miserable": ("miser", "wretched", "awful feeling"),
    "depressed": ("depress", "hopeless", "despair", "empty inside"),
    "tired": ("tired", "exhaust", "fatigue", "weary", "sleepy", "drained"),
    "calm": ("calm", "composed", "unruffled"),
    "content": ("content", "satisf", "fulfil"),
    "relaxed": ("relax", "at ease", "laid-back", "laid back"),
    "serene": ("seren", "tranquil", "peaceful", "at peace"),
    "neutral": ("neutral",),
    "attentive": ("attentiv", "focus", "alert", "concentrat"),
    "uncertain": ("uncertain", "unsure", "doubt", "confus", "not sure"),
    "reflective": ("reflect", "contemplat", "pensive", "introspect", "ponder"),
}


@dataclass(frozen=True)
class Example:
    concept: str
    format: str
    target: str
    text: str


def names_feeling(concept: str, text: str) -> str | None:
    """Return the offending stem if the text names the concept's feeling."""
    lowered = text.casefold()
    for stem in NAMED_STEMS.get(concept, (concept,)):
        # Stems match at word starts so "mad" catches "madness" but not "made" or "nomad".
        if stem == "mad":
            pattern = r"\bmad(?:der|dest|ly|ness)?\b"
        else:
            pattern = r"\b" + re.escape(stem)
        if re.search(pattern, lowered):
            return stem
    return None


def validate(example: Example, concepts: set[str]) -> str | None:
    if example.concept not in concepts:
        return f"unknown concept {example.concept!r}"
    if example.format not in FORMATS:
        return f"unknown format {example.format!r}"
    if example.target not in TARGETS:
        return f"unknown target {example.target!r}"
    if not example.text.strip():
        return "empty text"
    stem = names_feeling(example.concept, example.text)
    if stem is not None:
        return f"names the feeling ({stem!r})"
    return None


def load_examples(path: Path, concepts: set[str] | None = None, strict: bool = True) -> list[Example]:
    examples = []
    problems = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        example = Example(record["concept"], record["format"], record["target"], record["text"].strip())
        problem = validate(example, concepts or set(NAMED_STEMS))
        if problem is None:
            examples.append(example)
        else:
            problems.append(f"{path.name}:{number}: {problem}")
    if strict and problems:
        raise ValueError("Invalid calibration examples:\n" + "\n".join(problems))
    return examples


def coverage(examples: list[Example]) -> Counter[tuple[str, str]]:
    return Counter((example.concept, example.format) for example in examples)
