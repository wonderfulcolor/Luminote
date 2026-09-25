import numpy as np
import torch

from luminote.calibration import within_class_whitener
from luminote.reaction_matching import ReactionIndex


def test_whitening_groups_by_feeling_rather_than_topic():
    # Dimension 0 is the feeling (calm -1, tense +1); dimension 1 is a larger topic swing (A +2, B -2)
    # that varies within each feeling.
    labels = ["calm", "calm", "tense", "tense"] * 25
    features = np.array([[-1.0, 2.0], [-1.0, -2.0], [1.0, 2.0], [1.0, -2.0]] * 25)
    whitener = within_class_whitener(features, labels, shrink=0.3)

    def cosine(left, right):
        return left @ right / np.linalg.norm(left) / np.linalg.norm(right)

    calm_a, calm_b, tense_a = features[0], features[1], features[2]
    # Raw activations match the same topic across feelings...
    assert cosine(calm_a, tense_a) > cosine(calm_a, calm_b)
    # ...whitened ones match the same feeling across topics.
    assert cosine(calm_a @ whitener, calm_b @ whitener) > cosine(calm_a @ whitener, tense_a @ whitener)


def test_reaction_index_ranks_by_similarity_and_round_trips(tmp_path):
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=np.float32)
    index = ReactionIndex(["fine", "panic", "both"], embeddings, layer=16, model_name="test")
    path = tmp_path / "reactions.npz"
    index.save(path)
    loaded = ReactionIndex.load(path)
    assert (loaded.ids, loaded.layer, loaded.model_name) == (["fine", "panic", "both"], 16, "test")
    ranked = loaded.rank(torch.tensor([0.0, 1.0]), limit=2)
    assert [row["id"] for row in ranked] == ["panic", "both"]
    assert ranked[0]["similarity"] == 1.0
    assert ReactionIndex.load(tmp_path / "missing.npz") is None
