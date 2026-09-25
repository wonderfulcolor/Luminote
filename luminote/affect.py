from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

# Top matches closer than this on the valence/arousal plane are treated as one feeling;
# farther apart they are reported as separate modes instead of being averaged together.
MODE_RADIUS = 0.6
MIN_MODE_WEIGHT = 0.15


@dataclass(frozen=True)
class Match:
    concept: str
    similarity: float


@dataclass(frozen=True)
class Mode:
    concepts: list[str]
    valence: float
    arousal: float
    dominance: float | None
    weight: float


@dataclass(frozen=True)
class AffectEstimate:
    valence: float
    arousal: float
    confidence: float
    matches: list[Match]
    dominance: float | None = None
    modes: list[Mode] = field(default_factory=list)
    target: dict[str, float] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SpanReading:
    """Summary of per-token readings across one span of the conversation."""

    mean: AffectEstimate
    peaks: list[dict]
    trace: list[list[float]]
    token_count: int

    def to_dict(self) -> dict:
        return asdict(self)


class PrototypeSet:
    def __init__(self, prototypes: torch.Tensor, baselines: torch.Tensor) -> None:
        # prototypes: [layer, concept, hidden]; baselines: [layer, hidden]
        self.baselines = baselines
        self.normalized = prototypes / prototypes.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def similarities(self, activations: torch.Tensor, layer: int) -> torch.Tensor:
        """Cosine similarity of [..., hidden] activations against every prototype."""
        centered = activations.float().cpu() - self.baselines[layer]
        centered = centered / centered.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return centered @ self.normalized[layer].T


class AxisProbe:
    """Linear readout of valence, arousal (and dominance) fitted on example activations."""

    def __init__(self, weights: torch.Tensor, means: torch.Tensor, intercepts: torch.Tensor) -> None:
        # weights: [layer, hidden, axis]; means: [layer, hidden]; intercepts: [layer, axis]
        self.weights = weights
        self.means = means
        self.intercepts = intercepts

    def predict(self, activations: torch.Tensor, layer: int) -> torch.Tensor:
        """[..., hidden] activations to [..., axis] coordinates, clipped to the circumplex range."""
        centered = activations.float().cpu() - self.means[layer]
        return torch.clamp(centered @ self.weights[layer] + self.intercepts[layer], -1.0, 1.0)


def summarize(
    similarities: torch.Tensor,
    concepts: list[str],
    coordinates: torch.Tensor,
    target: dict[str, float] | None = None,
) -> AffectEstimate:
    count = min(5, len(concepts))
    values, indices = torch.topk(similarities, count)
    # Only evidence above zero influences the coordinate; softmax keeps it smooth.
    evidence = torch.relu(values)
    weights = torch.softmax(evidence * 6.0, dim=0)
    points = coordinates[indices]

    clusters: list[list[int]] = []
    for rank in range(count):
        for cluster in clusters:
            if torch.dist(points[cluster[0], :2], points[rank, :2]) <= MODE_RADIUS:
                cluster.append(rank)
                break
        else:
            clusters.append([rank])
    modes = []
    for cluster in clusters:
        member_weights = weights[cluster]
        total = member_weights.sum()
        point = (points[cluster] * member_weights[:, None]).sum(dim=0) / total
        modes.append((float(total), point, cluster))
    modes.sort(key=lambda mode: -mode[0])
    coordinate = modes[0][1]

    strength = torch.sigmoid((values[0] - 0.12) * 8.0)
    margin = torch.sigmoid((values[0] - values[min(2, count - 1)]) * 10.0)
    agreement = 1.0 - torch.clamp(
        ((points[:, :2] - coordinate[:2]) ** 2).sum(dim=1).sqrt().mean() / 1.5,
        0.0,
        1.0,
    )
    confidence = float(torch.clamp(strength * (0.55 + 0.45 * margin) * agreement, 0, 1))
    has_dominance = coordinates.shape[1] > 2
    return AffectEstimate(
        valence=round(float(coordinate[0]), 4),
        arousal=round(float(coordinate[1]), 4),
        confidence=round(confidence, 4),
        matches=[
            Match(concepts[int(index)], round(float(value), 4))
            for value, index in zip(values, indices, strict=True)
        ],
        dominance=round(float(coordinate[2]), 4) if has_dominance else None,
        modes=[
            Mode(
                concepts=[concepts[int(indices[rank])] for rank in cluster],
                valence=round(float(point[0]), 4),
                arousal=round(float(point[1]), 4),
                dominance=round(float(point[2]), 4) if has_dominance else None,
                weight=round(weight, 4),
            )
            for weight, point, cluster in modes
            if weight >= MIN_MODE_WEIGHT
        ],
        target=target,
    )


class PrototypeEstimator:
    def __init__(self, vectors_path: Path, metadata_path: Path) -> None:
        archive = np.load(vectors_path)
        metadata = json.loads(metadata_path.read_text())
        self.boundary = PrototypeSet(
            torch.from_numpy(archive["prototypes"]).float(),
            torch.from_numpy(archive["baselines"]).float(),
        )
        # Prototypes built from example displays, read over token spans. Older calibrations lack them.
        self.span = (
            PrototypeSet(
                torch.from_numpy(archive["span_prototypes"]).float(),
                torch.from_numpy(archive["span_baselines"]).float(),
            )
            if "span_prototypes" in archive
            else None
        )
        self.targets: list[str] = metadata.get("targets", [])
        self.target_set = (
            PrototypeSet(
                torch.from_numpy(archive["target_prototypes"]).float(),
                torch.from_numpy(archive["span_baselines"]).float(),
            )
            if "target_prototypes" in archive
            else None
        )
        self.axes = (
            AxisProbe(*(torch.from_numpy(archive[name]).float() for name in ("axis_weights", "axis_means", "axis_intercepts")))
            if "axis_weights" in archive
            else None
        )
        self.match_layer = int(archive["match_layer"]) if "match_whitener" in archive else None
        self.match_whitener = torch.from_numpy(archive["match_whitener"]).float() if self.match_layer is not None else None
        self.match_origin = torch.from_numpy(archive["match_origin"]).float() if self.match_layer is not None else None
        self.concepts: list[str] = metadata["concepts"]
        self.coordinates = torch.tensor(metadata["coordinates"], dtype=torch.float32)
        self.model_name: str = metadata["model_name"]
        self.layer_count = int(archive["prototypes"].shape[0])

    def _clamp(self, layer: int) -> int:
        return max(0, min(layer, self.layer_count - 1))

    def estimate(self, activation: torch.Tensor, layer: int) -> AffectEstimate:
        """Read a single response-boundary activation against the boundary prototypes."""
        layer = self._clamp(layer)
        return summarize(self.boundary.similarities(activation, layer), self.concepts, self.coordinates)

    def _target(self, pooled: torch.Tensor, layer: int) -> dict[str, float] | None:
        if self.target_set is None:
            return None
        scores = torch.softmax(self.target_set.similarities(pooled, layer) * 10.0, dim=0)
        return {name: round(float(score), 4) for name, score in zip(self.targets, scores, strict=True)}

    def estimate_pooled(self, activations: torch.Tensor, layer: int) -> AffectEstimate:
        """Read the mean of a [token, hidden] span against the example-built prototypes."""
        layer = self._clamp(layer)
        prototypes = self.span or self.boundary
        pooled = activations.float().mean(dim=0)
        estimate = summarize(
            prototypes.similarities(pooled, layer),
            self.concepts,
            self.coordinates,
            self._target(pooled, layer),
        )
        if self.axes is None:
            return estimate
        # Prototypes name the feelings; the axis probe places them, which transfers better.
        point = [round(float(value), 4) for value in self.axes.predict(pooled, layer)]
        return replace(
            estimate,
            valence=point[0],
            arousal=point[1],
            dominance=point[2] if len(point) > 2 else estimate.dominance,
        )

    def match_embedding(self, activations: torch.Tensor, layer: int) -> torch.Tensor | None:
        """Unit vector for reaction matching from a [token, hidden] span, or None if unavailable.

        Only the calibrated match layer has a whitener, so other layers return None.
        """
        if self.match_whitener is None or layer != self.match_layer:
            return None
        whitened = (activations.float().cpu().mean(dim=0) - self.match_origin) @ self.match_whitener
        return whitened / whitened.norm().clamp_min(1e-8)

    def estimate_span(self, activations: torch.Tensor, layer: int, peak_count: int = 3) -> SpanReading:
        """Per-token readings over a span: the pooled estimate, strongest peaks, and a trace."""
        layer = self._clamp(layer)
        prototypes = self.span or self.boundary
        similarities = prototypes.similarities(activations, layer)  # [token, concept]
        best, positions = similarities.max(dim=0)
        order = torch.argsort(best, descending=True)[:peak_count]
        peaks = [
            {
                "concept": self.concepts[int(index)],
                "similarity": round(float(best[index]), 4),
                "token_index": int(positions[index]),
            }
            for index in order
        ]
        if self.axes is not None:
            trace = self.axes.predict(activations, layer)[:, :2].tolist()
        else:
            weights = torch.softmax(torch.relu(similarities) * 6.0, dim=1)
            trace = (weights @ self.coordinates[:, :2]).tolist()
        return SpanReading(
            mean=self.estimate_pooled(activations, layer),
            peaks=peaks,
            trace=[[round(value, 3) for value in point] for point in trace],
            token_count=int(activations.shape[0]),
        )
