import numpy as np
import pytest

from sensitivitymap import (
    analyze_model,
    effective_rank,
    find_bottlenecks,
    layer_report,
    layer_spectrum,
)


def _low_rank(rows=32, cols=32, rank=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, rank)) @ rng.standard_normal((rank, cols))


def test_spectrum_descending():
    W = np.random.default_rng(0).standard_normal((16, 12))
    s = layer_spectrum(W)
    assert s.shape == (12,)
    assert np.all(np.diff(s) <= 1e-12)


def test_effective_rank_full_rank():
    W = np.eye(8)
    assert effective_rank(W) == 8


def test_effective_rank_low_rank():
    W = _low_rank(rank=3)
    assert effective_rank(W) == 3


def test_effective_rank_rejects_bad_energy():
    with pytest.raises(ValueError):
        effective_rank(np.eye(4), energy=0.0)


def test_report_keys():
    report = layer_report(_low_rank(rank=4))
    for key in ("shape", "frobenius_norm", "spectral_norm", "condition_number", "effective_rank"):
        assert key in report
    assert report["effective_rank"] == 4


def test_condition_number_finite_for_full_rank():
    rng = np.random.default_rng(1)
    report = layer_report(rng.standard_normal((10, 10)) + np.eye(10))
    assert np.isfinite(report["condition_number"])


def test_condition_number_inf_for_singular():
    report = layer_report(np.ones((4, 4)))
    assert report["condition_number"] == float("inf")


def test_analyze_model():
    layers = {"a": _low_rank(rank=2), "b": np.eye(16)}
    reports = analyze_model(layers)
    assert set(reports) == {"a", "b"}


def test_analyze_model_rejects_empty():
    with pytest.raises(ValueError):
        analyze_model({})


def test_find_bottlenecks_orders_by_rank():
    layers = {"wide": np.eye(16), "narrow": _low_rank(rank=2)}
    reports = analyze_model(layers)
    assert find_bottlenecks(reports, top_k=1) == ["narrow"]
    assert find_bottlenecks(reports, top_k=2) == ["narrow", "wide"]


def test_layer_report_rejects_1d():
    with pytest.raises(ValueError):
        layer_report(np.zeros(5))
