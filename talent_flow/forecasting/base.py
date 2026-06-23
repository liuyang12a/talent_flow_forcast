#!/usr/bin/env python3
"""Base class for all OD-matrix forecasters.

A forecaster consumes an :class:`ODMatrixSeries` (the pooler output) and
emits a :class:`ForecastResult`. Subclasses implement :meth:`fit` and
:meth:`predict`. The generic :meth:`evaluate` delegates to the unified
evaluation framework so every method shares the same metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from talent_flow.core import ForecastResult, ODMatrixSeries


class BaseForecaster(ABC):
    """Abstract base for OD-matrix forecasters."""

    name: str = "base"

    def __init__(self, input_len: int, output_len: int, **config: Any):
        self.input_len = input_len
        self.output_len = output_len
        self.config: Dict[str, Any] = config
        self.is_fitted: bool = False

    @abstractmethod
    def fit(
        self,
        od_series: ODMatrixSeries,
        val_series: Optional[ODMatrixSeries] = None,
    ) -> "BaseForecaster":
        """Learn from an OD-matrix series."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, od_series: ODMatrixSeries) -> ForecastResult:
        """Forecast the next ``output_len`` steps after the end of ``od_series``."""
        raise NotImplementedError

    def evaluate(
        self,
        test_series: ODMatrixSeries,
        metrics: Optional[list[str]] = None,
        core_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Default evaluation: predict on ``test_series`` then score.

        The default implementation assumes the forecaster produces a
        ``ForecastResult`` whose ``ground_truth`` aligns with the last
        ``output_len`` steps of ``test_series``. Subclasses with custom
        windowing may override.
        """
        from talent_flow.evaluation import ForecastEvaluator

        result = self.predict(test_series)
        # if ground_truth not populated, fill from the tail of test_series
        if result.ground_truth.shape[0] != self.output_len:
            gt = test_series.matrix[-self.output_len :]
            result.ground_truth = gt
        prev = test_series.matrix[-self.output_len - 1]
        evaluator = ForecastEvaluator(metrics=metrics)
        return evaluator.evaluate(result, core_mask=core_mask, prev_values=prev)

    # ---- persistence (default: pickle; DL models may override) ----
    def save(self, path: str) -> None:
        import pickle
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "BaseForecaster":
        import pickle
        from pathlib import Path

        with Path(path).open("rb") as f:
            return pickle.load(f)
