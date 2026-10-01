import numpy as np
import pytest

from luminote.calibration import class_prototypes, fit_axes
from luminote.model import char_span_to_tokens


def test_class_prototypes_centre_on_balanced_class_means():
    # [example, layer, hidden]; class "a" has three examples, "b" one, so balancing matters.
    pooled = np.array([[[2.0, 0.0]], [[2.0, 0.0]], [[2.0, 0.0]], [[0.0, 2.0]]], dtype=np.float32)
    prototypes, baseline = class_prototypes(pooled, ["a", "a", "a", "b"], ["a", "b"])
    assert prototypes.shape == (1, 2, 2)
    np.testing.assert_allclose(baseline[0], [1.0, 1.0])
    np.testing.assert_allclose(prototypes[0], [[1.0, -1.0], [-1.0, 1.0]])


def test_class_prototypes_can_share_another_baseline():
    pooled = np.array([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32)
    prototypes, baseline = class_prototypes(pooled, ["x", "y"], ["x", "y"], baseline=np.zeros((1, 2)))
    np.testing.assert_allclose(prototypes[0], [[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(baseline, 0.0)


def test_class_prototypes_require_every_class():
    with pytest.raises(ValueError, match="No examples for: b"):
        class_prototypes(np.zeros((1, 1, 2), dtype=np.float32), ["a"], ["a", "b"])


def test_char_spans_map_to_overlapping_tokens():
    offsets = [(0, 3), (3, 4), (4, 9), (9, 10), (10, 16)]
    assert char_span_to_tokens(offsets, 4, 16) == (2, 5)
    assert char_span_to_tokens(offsets, 5, 6) == (2, 3)
    with pytest.raises(ValueError):
        char_span_to_tokens(offsets, 20, 22)


def test_axis_fit_recovers_a_linear_readout():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 1, 6)).astype(np.float32)
    targets = np.stack([features[:, 0, 0], -features[:, 0, 1]], axis=1) * 0.3
    weights, means, intercepts = fit_axes(features, targets, ridge=1e-3)
    predicted = (features[:, 0] - means[0]) @ weights[0] + intercepts[0]
    np.testing.assert_allclose(predicted, targets, atol=1e-3)
