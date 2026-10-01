from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from .examples import Example
from .model import LocalModel

SYSTEM_PROMPT = "You are a concise, natural conversational partner."
# Examples are read as the assistant's own turn, which is where runtime replies are read.
# The preceding user turn rotates so no single prompt shapes every prototype.
USER_PROMPTS = (
    "Go on.",
    "What's on your mind?",
    "Say whatever comes next.",
)


def example_messages(text: str, index: int = 0) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPTS[index % len(USER_PROMPTS)]},
        {"role": "assistant", "content": text},
    ]


def pool_texts(model: LocalModel, texts: Iterable[str], progress: bool = False) -> np.ndarray:
    """Mean activation over each text's tokens, read as an assistant turn: [text, layer, hidden]."""
    pooled = []
    texts = list(texts)
    for index, text in enumerate(texts):
        spans = model.inspect_spans(example_messages(text, index))
        start, end = spans.reply_span
        pooled.append(spans.hidden[:, start:end].mean(dim=1).numpy())
        if progress and (index + 1) % 100 == 0:
            print(f"  pooled {index + 1}/{len(texts)}", flush=True)
    return np.stack(pooled).astype(np.float32)


def pool_examples(model: LocalModel, examples: Sequence[Example], progress: bool = False) -> np.ndarray:
    return pool_texts(model, (example.text for example in examples), progress)


def class_prototypes(
    pooled: np.ndarray,
    labels: Sequence[str],
    classes: Sequence[str],
    baseline: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One-vs-rest prototypes: each class mean minus the mean of all classes.

    Centering on the pooled mean of every example, rather than on a handful of neutral prompts,
    removes what all emotional displays share so each prototype keeps what sets its class apart.
    Pass `baseline` to centre on another set's origin instead.
    Returns ([layer, class, hidden], [layer, hidden]).
    """
    labels = np.asarray(labels)
    missing = [name for name in classes if not np.any(labels == name)]
    if missing:
        raise ValueError(f"No examples for: {', '.join(missing)}")
    means = np.stack([pooled[labels == name].mean(axis=0) for name in classes])
    if baseline is None:
        # Balance classes so a well-covered concept does not dominate the shared baseline.
        baseline = means.mean(axis=0)
    prototypes = means - baseline
    return np.transpose(prototypes, (1, 0, 2)).astype(np.float32), baseline.astype(np.float32)


MATCH_SHRINK = 0.6


def within_class_whitener(features: np.ndarray, labels: Sequence[str], shrink: float = MATCH_SHRINK) -> np.ndarray:
    """[hidden, hidden] map that whitens by variation *within* each feeling.

    Examples of one feeling differ mostly in topic and wording. Down-weighting those directions
    lets reaction matching compare stance rather than subject, while keeping every dimension,
    unlike a projection onto the concept prototypes, which discards most of what separates memes.
    `shrink` blends the covariance toward isotropic so rarely seen directions are not blown up.
    """
    labels = np.asarray(labels)
    features = features.astype(np.float64)
    centered = np.concatenate([features[labels == name] - features[labels == name].mean(axis=0) for name in np.unique(labels)])
    covariance = centered.T @ centered / len(centered)
    covariance = (1 - shrink) * covariance + shrink * np.trace(covariance) / len(covariance) * np.eye(len(covariance))
    values, vectors = np.linalg.eigh(covariance)
    return (vectors / np.sqrt(values)).astype(np.float32)


AXIS_RIDGE = 1000.0


def fit_axes(
    pooled: np.ndarray, targets: np.ndarray, ridge: float = AXIS_RIDGE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ridge-regress [example, layer, hidden] activations onto [example, axis] coordinates, per layer.

    Prototype matching keeps only the few nearest concepts; a linear read of each axis uses the
    whole activation and transfers noticeably better to human-written text (see benchmark_layers).
    Returns (weights [layer, hidden, axis], means [layer, hidden], intercepts [layer, axis]).
    """
    targets = np.asarray(targets, dtype=np.float64)
    intercept = targets.mean(axis=0)
    centered_targets = targets - intercept
    weights, means = [], []
    for layer in range(pooled.shape[1]):
        features = pooled[:, layer].astype(np.float64)
        mean = features.mean(axis=0)
        centered = features - mean
        # Dual form: there are far fewer examples than hidden dimensions.
        gram = centered @ centered.T + ridge * np.eye(len(centered))
        weights.append(centered.T @ np.linalg.solve(gram, centered_targets))
        means.append(mean)
    return (
        np.stack(weights).astype(np.float32),
        np.stack(means).astype(np.float32),
        np.tile(intercept, (pooled.shape[1], 1)).astype(np.float32),
    )
