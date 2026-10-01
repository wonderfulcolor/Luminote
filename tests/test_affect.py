import json

import numpy as np
import torch

from luminote.affect import PrototypeEstimator, summarize


def test_prototype_estimator_prefers_matching_vector(tmp_path):
    vectors = np.array([[[1, 0], [0, 1], [-1, 0]]], dtype=np.float32)
    baseline = np.zeros((1, 2), dtype=np.float32)
    np.savez(tmp_path / "p.npz", prototypes=vectors, baselines=baseline)
    (tmp_path / "p.json").write_text(json.dumps({
        "model_name": "test", "concepts": ["happy", "awake", "sad"],
        "coordinates": [[1, 0], [0, 1], [-1, 0]]
    }))
    estimate = PrototypeEstimator(tmp_path / "p.npz", tmp_path / "p.json").estimate(torch.tensor([1.0, 0.0]), 0)
    assert estimate.matches[0].concept == "happy"
    assert estimate.valence > 0
    assert 0 <= estimate.confidence <= 1
    assert estimate.dominance is None
    assert estimate.target is None


def test_distant_strong_matches_become_separate_modes_instead_of_an_average():
    concepts = ["bitter", "amused", "other"]
    coordinates = torch.tensor([[-0.8, 0.2], [0.5, 0.2], [0.0, -0.9]])
    estimate = summarize(torch.tensor([0.50, 0.48, -0.2]), concepts, coordinates)
    assert [mode.concepts for mode in estimate.modes] == [["bitter"], ["amused"]]
    # The headline follows the strongest mode rather than landing between the two.
    assert estimate.valence < -0.5


def test_nearby_matches_merge_into_one_mode():
    concepts = ["angry", "tense", "calm"]
    coordinates = torch.tensor([[-0.78, 0.82, 0.35], [-0.56, 0.78, -0.30], [0.48, -0.72, 0.30]])
    estimate = summarize(torch.tensor([0.5, 0.45, -0.3]), concepts, coordinates)
    assert estimate.modes[0].concepts == ["angry", "tense"]
    assert estimate.dominance is not None
    assert -0.30 < estimate.dominance < 0.35


def test_span_readings_use_example_prototypes_and_report_peaks(tmp_path):
    layers, hidden = 1, 3
    np.savez(
        tmp_path / "p.npz",
        prototypes=np.zeros((layers, 2, hidden), dtype=np.float32) + 0.1,
        baselines=np.zeros((layers, hidden), dtype=np.float32),
        span_prototypes=np.array([[[1, 0, 0], [0, 1, 0]]], dtype=np.float32),
        span_baselines=np.zeros((layers, hidden), dtype=np.float32),
        target_prototypes=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=np.float32),
    )
    (tmp_path / "p.json").write_text(json.dumps({
        "model_name": "test", "concepts": ["glad", "gloomy"],
        "coordinates": [[0.8, 0.2, 0.3], [-0.8, -0.4, -0.5]], "targets": ["self", "other", "situation"],
    }))
    estimator = PrototypeEstimator(tmp_path / "p.npz", tmp_path / "p.json")
    tokens = torch.tensor([[0.2, 0.1, 0.0], [0.1, 0.9, 0.0], [0.3, 0.2, 0.0]])
    reading = estimator.estimate_span(tokens, 0)
    assert reading.token_count == 3
    assert reading.peaks[0] == {"concept": "gloomy", "similarity": reading.peaks[0]["similarity"], "token_index": 1}
    assert len(reading.trace) == 3
    assert reading.mean.matches[0].concept == "gloomy"
    assert set(reading.mean.target) == {"self", "other", "situation"}
    assert abs(sum(reading.mean.target.values()) - 1) < 1e-3


def test_axis_probe_places_span_readings_when_calibrated(tmp_path):
    layers, hidden = 1, 2
    np.savez(
        tmp_path / "p.npz",
        prototypes=np.zeros((layers, 2, hidden), dtype=np.float32) + 0.1,
        baselines=np.zeros((layers, hidden), dtype=np.float32),
        span_prototypes=np.array([[[1, 0], [0, 1]]], dtype=np.float32),
        span_baselines=np.zeros((layers, hidden), dtype=np.float32),
        # valence reads the first dimension, arousal the second, dominance neither.
        axis_weights=np.array([[[0.5, 0, 0], [0, 0.5, 0]]], dtype=np.float32),
        axis_means=np.zeros((layers, hidden), dtype=np.float32),
        axis_intercepts=np.array([[0, 0, 0.1]], dtype=np.float32),
    )
    (tmp_path / "p.json").write_text(json.dumps({
        "model_name": "test", "concepts": ["glad", "wired"], "coordinates": [[0.8, 0, 0], [0, 0.8, 0]],
    }))
    estimator = PrototypeEstimator(tmp_path / "p.npz", tmp_path / "p.json")
    reading = estimator.estimate_span(torch.tensor([[1.0, 0.2], [0.6, 0.2]]), 0)
    assert reading.mean.matches[0].concept == "glad"
    assert (reading.mean.valence, reading.mean.arousal, reading.mean.dominance) == (0.4, 0.1, 0.1)
    assert reading.trace == [[0.5, 0.1], [0.3, 0.1]]
    # Readings far outside the calibrated range are clipped to the circumplex.
    assert estimator.estimate_pooled(torch.tensor([[9.0, -9.0]]), 0).valence == 1.0
