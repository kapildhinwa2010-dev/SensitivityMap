# SensitivityMap - Map output sensitivity of weight rows via perturbation analysis
# Part of the Kapildhinwa2010-dev Toolkit.

"""DeepAnalyze - layer-wise spectral analysis for deep networks.

Lineage: Reverse.Dean (b2/b4). Given a dict of weight matrices ("layers"),
this module computes singular-value spectra, effective ranks and condition
numbers, and ranks layers by bottleneck severity.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "layer_spectrum",
    "effective_rank",
    "layer_report",
    "analyze_model",
    "find_bottlenecks",
    "row_sensitivity",
    "sensitivity_map",
]


def layer_spectrum(weights: np.ndarray) -> np.ndarray:
    """Singular values of ``weights`` in descending order."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError("weights must be a 2-D matrix")
    return np.linalg.svd(weights, compute_uv=False)


def effective_rank(weights: np.ndarray, energy: float = 0.99) -> int:
    """Smallest k whose singular values cover ``energy`` of total energy."""
    if not 0.0 < energy <= 1.0:
        raise ValueError("energy must be in (0, 1]")
    s = layer_spectrum(weights)
    total = float(s.sum())
    if total <= 0.0:
        return 1
    cumulative = np.cumsum(s) / total
    k = int(np.searchsorted(cumulative, energy, side="left") + 1)
    return int(min(k, s.size))


def layer_report(weights: np.ndarray) -> dict:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or weights.size == 0:
        raise ValueError("weights must be a non-empty 2-D matrix")
    s = layer_spectrum(weights)
    s_max = float(s[0])
    s_min = float(s[-1])
    return {
        "shape": (int(weights.shape[0]), int(weights.shape[1])),
        "frobenius_norm": float(np.linalg.norm(weights)),
        "spectral_norm": s_max,
        # Numerically singular: s_min below machine-eps scale => inf
        "condition_number": (
            s_max / s_min
            if s_min > s_max * np.finfo(np.float64).eps * max(weights.shape)
            else float("inf")
        ),
        "effective_rank": effective_rank(weights),
    }


def analyze_model(layers: dict) -> dict:
    """Report every layer of a model dict {name: 2-D matrix}."""
    if not layers:
        raise ValueError("layers must be non-empty")
    return {name: layer_report(w) for name, w in layers.items()}


def find_bottlenecks(reports: dict, top_k: int = 1) -> list:
    """Layer names with the smallest effective rank, worst first."""
    if not reports:
        return []
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    ranked = sorted(reports.items(), key=lambda kv: (kv[1]["effective_rank"], kv[0]))
    return [name for name, _ in ranked[:top_k]]


def row_sensitivity(weights: np.ndarray, samples: int = 64, eps: float = 1e-2, seed: int = 0) -> np.ndarray:
    """Relative output change when each row of ``weights`` is perturbed.

    For each row i, the input X is fixed and the row is scaled by
    (1 + eps); the score is mean |(WX_perturbed - WX)| / max|WX| over
    the sample batch. Rows with near-zero output contribution score ~0.

    Args:
        weights: 2-D weight matrix (out_features, in_features).
        samples: number of random input vectors.
        eps: relative row perturbation size.

    Returns: array of shape (out_features,) with per-row scores.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or weights.size == 0:
        raise ValueError("weights must be a non-empty 2-D matrix")
    if samples < 1:
        raise ValueError("samples must be >= 1")
    if eps <= 0:
        raise ValueError("eps must be > 0")
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((samples, weights.shape[1]))
    base = X @ weights.T
    denom = np.abs(base).max()
    if denom == 0.0:
        return np.zeros(weights.shape[0])
    scores = np.zeros(weights.shape[0])
    for i in range(weights.shape[0]):
        perturbed = weights.copy()
        perturbed[i] *= (1.0 + eps)
        delta = np.abs((X @ perturbed.T) - base).mean() / denom
        scores[i] = float(delta)
    return scores


def sensitivity_map(layers: dict, samples: int = 64, eps: float = 1e-2, seed: int = 0) -> dict:
    """Row sensitivity scores for every layer in a {name: matrix} dict.

    Returns dict {layer_name: (out_features,) score array}.
    """
    if not layers:
        raise ValueError("layers must be non-empty")
    return {
        name: row_sensitivity(w, samples=samples, eps=eps, seed=seed)
        for name, w in layers.items()
    }