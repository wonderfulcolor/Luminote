import json

import numpy as np
import torch

from luminote.affect import PrototypeEstimator


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

