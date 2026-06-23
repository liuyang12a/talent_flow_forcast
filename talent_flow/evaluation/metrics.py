#!/usr/bin/env python3
"""Basic regression metric library.

Convention: **all functions take ``(target, prediction)`` in that order**.
The legacy ``metrics.py`` mixed orders (``mae(pred, target)`` vs
``calculate_metrics(target, pred)``); this module unifies them to avoid
foot-guns. Targets always come first.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np


def mae(target: np.ndarray, prediction: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(target) - np.asarray(prediction))))


def mse(target: np.ndarray, prediction: np.ndarray) -> float:
    """Mean Squared Error."""
    diff = np.asarray(target) - np.asarray(prediction)
    return float(np.mean(diff**2))


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(target, prediction)))


def mape(target: np.ndarray, prediction: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (percent). Skips near-zero targets."""
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    mask = np.abs(target) > eps
    if not np.any(mask):
        return 0.0
    return float(
        np.mean(np.abs((prediction[mask] - target[mask]) / (target[mask] + eps))) * 100
    )


def wape(target: np.ndarray, prediction: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (percent)."""
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    return float(np.sum(np.abs(prediction - target)) / (np.sum(np.abs(target)) + 1e-8) * 100)


def smape(target: np.ndarray, prediction: np.ndarray, eps: float = 1e-8) -> float:
    """Symmetric MAPE (percent)."""
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    denom = (np.abs(target) + np.abs(prediction)) / 2 + eps
    return float(np.mean(np.abs(prediction - target) / denom) * 100)


def r2_score(target: np.ndarray, prediction: np.ndarray) -> float:
    """R-squared coefficient of determination."""
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    ss_res = np.sum((target - prediction) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1 - (ss_res / ss_tot))


def correlation(target: np.ndarray, prediction: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    t = np.asarray(target, dtype=float).flatten()
    p = np.asarray(prediction, dtype=float).flatten()
    if len(t) < 2:
        return 0.0
    t_std, p_std = np.std(t), np.std(p)
    if t_std == 0 or p_std == 0:
        return 0.0
    return float(np.mean((t - np.mean(t)) * (p - np.mean(p))) / (t_std * p_std))


def directional_accuracy(
    target: np.ndarray, prediction: np.ndarray, prev: np.ndarray
) -> float:
    """Directional accuracy: fraction of correctly predicted up/down moves.

    Args:
        target: ground truth ``[...]``
        prediction: forecast ``[...]``
        prev: the value immediately before the forecast window (same shape as
            target, or broadcastable) used to define the true/pred direction.
    """
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    prev = np.asarray(prev, dtype=float)
    true_dir = np.sign(target - prev)
    pred_dir = np.sign(prediction - prev)
    return float(np.mean(true_dir == pred_dir))


METRIC_REGISTRY: Dict[str, Callable[..., float]] = {
    "mae": mae,
    "mse": mse,
    "rmse": rmse,
    "mape": mape,
    "wape": wape,
    "smape": smape,
    "r2": r2_score,
    "correlation": correlation,
    "directional_accuracy": directional_accuracy,
}

DEFAULT_METRICS: list[str] = ["mae", "rmse", "mape"]


def calculate_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    metrics: Optional[list[str]] = None,
    **kwargs,
) -> Dict[str, float]:
    """Compute multiple metrics. **Argument order: ``(target, prediction)``**.

    Args:
        target: ground truth array.
        prediction: forecast array.
        metrics: metric names; defaults to :data:`DEFAULT_METRICS`.
        **kwargs: extra keyword args forwarded to each metric (e.g. ``prev``
            for ``directional_accuracy``).
    """
    metrics = metrics or DEFAULT_METRICS
    results: Dict[str, float] = {}
    for name in metrics:
        fn = METRIC_REGISTRY.get(name)
        if fn is None:
            raise ValueError(f"Unknown metric: {name}")
        try:
            # Only forward kwargs the function accepts (directional_accuracy
            # needs `prev`); others ignore them.
            import inspect

            sig = inspect.signature(fn)
            if "prev" in kwargs and "prev" in sig.parameters:
                results[name] = float(fn(target, prediction, **kwargs))
            else:
                results[name] = float(fn(target, prediction))
        except Exception:
            results[name] = float("nan")
    return results
