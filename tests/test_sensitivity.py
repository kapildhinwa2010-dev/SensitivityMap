import numpy as np
import pytest

from sensitivitymap import row_sensitivity, sensitivity_map


def test_zero_row_has_zero_sensitivity():
    W = np.eye(4)
    W[3] = 0.0
    scores = row_sensitivity(W, samples=32)
    assert scores[3] == pytest.approx(0.0, abs=1e-12)


def test_all_rows_positive_for_generic_matrix():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((8, 6)) + 0.1
    scores = row_sensitivity(W, samples=32)
    assert scores.shape == (8,)
    assert (scores > 0).all()


def test_scores_small_for_small_eps():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((8, 8))
    scores = row_sensitivity(W, samples=32, eps=1e-6)
    assert scores.max() < 1e-4


def test_rejects_bad_args():
    with pytest.raises(ValueError):
        row_sensitivity(np.ones(4))
    with pytest.raises(ValueError):
        row_sensitivity(np.eye(4), samples=0)
    with pytest.raises(ValueError):
        row_sensitivity(np.eye(4), eps=0.0)


def test_sensitivity_map_shape_and_rejects_empty():
    layers = {"a": np.eye(4), "b": np.ones((3, 5))}
    out = sensitivity_map(layers, samples=16)
    assert set(out) == {"a", "b"}
    assert out["a"].shape == (4,)
    with pytest.raises(ValueError):
        sensitivity_map({})
