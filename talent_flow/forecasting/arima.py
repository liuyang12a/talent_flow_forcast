#!/usr/bin/env python3
"""ARIMA forecaster (per OD-pair), migrated to the new contract.

Fits one statsmodels ARIMA per ``(i, j)`` OD pair on the training series and
forecasts each pair independently. This is the legacy statistical baseline;
it ignores cross-pair correlation but is a strong per-flow reference.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from talent_flow.core import ForecastResult, ODMatrixSeries
from talent_flow.core.registry import FORECASTER_REGISTRY
from .base import BaseForecaster


@FORECASTER_REGISTRY.register("arima")
class ARIMAForecaster(BaseForecaster):
    """Per-OD-pair ARIMA.

    Config:
        order: ``(p, d, q)`` ARIMA order.
        fallback_last_value: if a pair's fit fails, forecast its last value.
    """

    name = "arima"

    def __init__(
        self,
        input_len: int = 12,
        output_len: int = 1,
        order: tuple = (1, 1, 1),
        fallback_last_value: bool = True,
        **kwargs,
    ):
        super().__init__(
            input_len, output_len, order=list(order), **kwargs
        )
        self.order = tuple(order)
        self.fallback_last_value = fallback_last_value
        self._models = None
        self._shape = None

    def fit(self, od_series: ODMatrixSeries, val_series=None):
        from statsmodels.tsa.arima.model import ARIMA

        M = od_series.matrix  # [T, K, K]
        T, K, _ = M.shape
        self._shape = (K, K)
        self._models = {}
        for i in range(K):
            for j in range(K):
                series = M[:, i, j]
                # ARIMA needs enough non-constant data
                if np.all(series == series[0]):
                    self._models[(i, j)] = None
                    continue
                try:
                    model = ARIMA(series, order=self.order).fit()
                    self._models[(i, j)] = model
                except Exception:
                    self._models[(i, j)] = None
        self.is_fitted = True
        return self

    def predict(self, od_series: ODMatrixSeries) -> ForecastResult:
        if not self.is_fitted:
            raise RuntimeError("ARIMAForecaster not fitted")
        K, _ = self._shape
        M = od_series.matrix
        preds = np.zeros((self.output_len, K, K), dtype=float)
        for i in range(K):
            for j in range(K):
                model = self._models.get((i, j))
                if model is not None:
                    try:
                        f = model.forecast(steps=self.output_len)
                        preds[:, i, j] = np.asarray(f, dtype=float)
                        continue
                    except Exception:
                        pass
                # fallback
                if self.fallback_last_value:
                    preds[:, i, j] = M[-1, i, j]
        gt = M[-self.output_len :] if M.shape[0] >= self.output_len else M
        return ForecastResult(
            predictions=preds,
            ground_truth=gt,
            forecaster_name=self.name,
            metadata={"order": self.order},
        )
