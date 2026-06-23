#!/usr/bin/env python3
"""Forecast-stage accuracy evaluation, with optional core/periphery split."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from talent_flow.core import ForecastResult
from .metrics import calculate_metrics, DEFAULT_METRICS


class ForecastEvaluator:
    """Evaluate a :class:`ForecastResult` with unified metrics.

    Supports layered evaluation (overall / core / periphery) when a
    ``core_mask`` is provided, and probabilistic evaluation when the result
    carries prediction intervals.
    """

    def __init__(self, metrics: Optional[list[str]] = None):
        self.metrics = metrics or DEFAULT_METRICS

    def evaluate(
        self,
        result: ForecastResult,
        core_mask: Optional[np.ndarray] = None,
        prev_values: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate ``result``.

        Args:
            result: the forecast to evaluate.
            core_mask: optional ``[K]`` boolean array marking core super-nodes.
                If given, metrics are also reported per layer (core/periphery)
                on the in/out flows of those nodes.
            prev_values: optional ``[K, K]`` array (the last observed OD
                matrix) needed for ``directional_accuracy``.
        """
        target = result.ground_truth
        pred = result.predictions
        kwargs = {}
        if prev_values is not None and "directional_accuracy" in self.metrics:
            kwargs["prev"] = prev_values

        report: Dict[str, Dict[str, float]] = {}
        report["overall"] = calculate_metrics(target, pred, self.metrics, **kwargs)

        if core_mask is not None:
            core_mask = np.asarray(core_mask, dtype=bool)
            # core layer: rows/columns indexed by core nodes
            report["core"] = calculate_metrics(
                target[:, core_mask][:, :, core_mask],
                pred[:, core_mask][:, :, core_mask],
                self.metrics,
                **(
                    {"prev": prev_values[core_mask][:, core_mask]}
                    if prev_values is not None and "directional_accuracy" in self.metrics
                    else {}
                ),
            )
            periphery = ~core_mask
            report["periphery"] = calculate_metrics(
                target[:, periphery][:, :, periphery],
                pred[:, periphery][:, :, periphery],
                self.metrics,
                **(
                    {"prev": prev_values[periphery][:, periphery]}
                    if prev_values is not None and "directional_accuracy" in self.metrics
                    else {}
                ),
            )

        if result.prediction_intervals is not None:
            from .probabilistic import ProbabilisticEvaluator

            report["probabilistic"] = ProbabilisticEvaluator().evaluate(
                target, result.prediction_intervals
            )
        return report
