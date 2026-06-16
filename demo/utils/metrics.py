"""
Evaluation metrics for time series forecasting.
"""

from typing import Dict, List, Optional
import numpy as np


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(prediction - target)))


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((prediction - target) ** 2))


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(prediction, target)))


def mape(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error.

    Handles zero values by adding eps to denominator.
    """
    mask = np.abs(target) > eps
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((prediction[mask] - target[mask]) / (target[mask] + eps))) * 100)


def wape(prediction: np.ndarray, target: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error.

    Also known as MAPE with sum normalization.
    """
    return float(np.sum(np.abs(prediction - target)) / (np.sum(np.abs(target)) + 1e-8) * 100)


def smape(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """
    Symmetric Mean Absolute Percentage Error.

    More robust than MAPE for values near zero.
    """
    denom = (np.abs(target) + np.abs(prediction)) / 2 + eps
    return float(np.mean(np.abs(prediction - target) / denom) * 100)


def r2_score(prediction: np.ndarray, target: np.ndarray) -> float:
    """R-squared coefficient of determination."""
    ss_res = np.sum((target - prediction) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1 - (ss_res / ss_tot))


def correlation(prediction: np.ndarray, target: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    pred_flat = prediction.flatten()
    target_flat = target.flatten()

    if len(pred_flat) < 2:
        return 0.0

    pred_mean = np.mean(pred_flat)
    target_mean = np.mean(target_flat)

    pred_std = np.std(pred_flat)
    target_std = np.std(target_flat)

    if pred_std == 0 or target_std == 0:
        return 0.0

    covariance = np.mean((pred_flat - pred_mean) * (target_flat - target_mean))
    return float(covariance / (pred_std * target_std))


METRIC_FUNCTIONS = {
    'mae': mae,
    'mse': mse,
    'rmse': rmse,
    'mape': mape,
    'wape': wape,
    'smape': smape,
    'r2': r2_score,
    'correlation': correlation,
}


def calculate_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate multiple metrics at once.

    Args:
        target: Ground truth values
        prediction: Predicted values
        metrics: List of metric names to compute. If None, computes all metrics.

    Returns:
        Dictionary mapping metric names to values
    """
    if metrics is None:
        metrics = list(METRIC_FUNCTIONS.keys())

    results = {}
    for metric_name in metrics:
        if metric_name in METRIC_FUNCTIONS:
            try:
                results[metric_name] = METRIC_FUNCTIONS[metric_name](prediction, target)
            except Exception as e:
                results[metric_name] = float('nan')
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

    return results
