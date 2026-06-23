#!/usr/bin/env python3
"""Probabilistic forecast calibration metrics."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def picp(target: np.ndarray, intervals: Dict[str, Any]) -> float:
    """Prediction Interval Coverage Probability.

    Fraction of ground-truth values falling within ``[lower, upper]``.
    """
    lower = np.asarray(intervals["lower"], dtype=float)
    upper = np.asarray(intervals["upper"], dtype=float)
    target = np.asarray(target, dtype=float)
    inside = (target >= lower) & (target <= upper)
    return float(np.mean(inside))


def pinaw(target: np.ndarray, intervals: Dict[str, Any]) -> float:
    """Prediction Interval Normalized Average Width.

    Interval width normalized by the range of the target.
    """
    lower = np.asarray(intervals["lower"], dtype=float)
    upper = np.asarray(intervals["upper"], dtype=float)
    target = np.asarray(target, dtype=float)
    width = np.mean(upper - lower)
    span = np.ptp(target)
    if span == 0:
        return 0.0
    return float(width / span)


def crps_gaussian(
    target: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> float:
    """CRPS for Gaussian predictive distributions."""
    from scipy.stats import norm  # local import: scipy optional for some metrics

    std = np.asarray(std, dtype=float) + 1e-8
    z = (np.asarray(target, dtype=float) - np.asarray(mean, dtype=float)) / std
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    return float(np.mean(std * (z * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))))


class ProbabilisticEvaluator:
    """Evaluate probabilistic forecasts (intervals)."""

    def evaluate(
        self, target: np.ndarray, intervals: Dict[str, Any]
    ) -> Dict[str, float]:
        return {
            "picp": picp(target, intervals),
            "pinaw": pinaw(target, intervals),
        }
