#!/usr/bin/env python3
"""Naive baselines: persistence and historical-mean forecasters."""

from __future__ import annotations

from typing import Optional

import numpy as np

from talent_flow.core import ForecastResult, ODMatrixSeries
from talent_flow.core.registry import FORECASTER_REGISTRY
from .base import BaseForecaster


@FORECASTER_REGISTRY.register("naive")
class NaiveForecaster(BaseForecaster):
    """Persistence baseline: predict the last observed OD matrix repeats.

    Config:
        strategy: ``"persistence"`` (repeat last) or ``"historical_mean"``
            (mean over the input window).
    """

    name = "naive"

    def __init__(
        self,
        input_len: int = 12,
        output_len: int = 1,
        strategy: str = "persistence",
        **kwargs,
    ):
        super().__init__(input_len, output_len, strategy=strategy, **kwargs)
        self.strategy = strategy

    def fit(self, od_series, val_series=None):
        self.is_fitted = True
        return self

    def predict(self, od_series: ODMatrixSeries) -> ForecastResult:
        M = od_series.matrix
        if self.strategy == "persistence":
            base = M[-1]
            preds = np.repeat(base[None], self.output_len, axis=0)
        elif self.strategy == "historical_mean":
            window = M[-self.input_len :]
            base = window.mean(axis=0)
            preds = np.repeat(base[None], self.output_len, axis=0)
        else:
            raise ValueError(f"unknown strategy: {self.strategy}")
        gt = M[-self.output_len :] if M.shape[0] >= self.output_len else M
        return ForecastResult(
            predictions=preds,
            ground_truth=gt,
            forecaster_name=self.name,
            metadata={"strategy": self.strategy},
        )
