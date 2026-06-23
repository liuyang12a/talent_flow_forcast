#!/usr/bin/env python3
"""Two-stage pooling+forecasting pipeline orchestration.

Loosely coupled: the pipeline only glues a :class:`BasePooler` and a
:class:`BaseForecaster` together via the :class:`ODMatrixSeries` contract.
Either stage can also be run standalone (see ``scripts/run_pooling.py`` and
``scripts/run_forecast.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from talent_flow.core import (
    AssignmentMatrix,
    FlowNetwork,
    ForecastResult,
    ODMatrixSeries,
    PoolingResult,
    POOLER_REGISTRY,
    FORECASTER_REGISTRY,
)
from talent_flow.forecasting import SplitRatios, split_od_series
from talent_flow.pooling import BasePooler
from talent_flow.pooling.core_periphery import CorePeripheryPooler
from talent_flow.forecasting.base import BaseForecaster


@dataclass
class PipelineResult:
    """Outcome of a full pipeline run."""

    pooling: PoolingResult
    forecast: ForecastResult
    metrics: Dict[str, Any] = field(default_factory=dict)
    core_mask: Optional[np.ndarray] = None


class PoolingForecastPipeline:
    """Orchestrate pooling -> forecasting -> evaluation."""

    def __init__(
        self,
        pooler: BasePooler,
        forecaster: BaseForecaster,
        split: Optional[SplitRatios] = None,
    ):
        self.pooler = pooler
        self.forecaster = forecaster
        self.split = split or SplitRatios()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PoolingForecastPipeline":
        """Build from a config dict (e.g. parsed from YAML).

        Expected keys: ``pooling.name``, ``pooling.params``,
        ``forecasting.name``, ``forecasting.params``.
        """
        pooler = POOLER_REGISTRY.build(
            config["pooling"]["name"], **config["pooling"].get("params", {})
        )
        fc = FORECASTER_REGISTRY.build(
            config["forecasting"]["name"],
            **config["forecasting"].get("params", {}),
        )
        split_cfg = config.get("split", {})
        split = SplitRatios(
            train=split_cfg.get("train_ratio", 0.7),
            val=split_cfg.get("val_ratio", 0.15),
            test=split_cfg.get("test_ratio", 0.15),
        )
        return cls(pooler, fc, split)

    def _core_mask(self, pooling_result: PoolingResult) -> Optional[np.ndarray]:
        if isinstance(self.pooler, CorePeripheryPooler):
            return self.pooler.get_core_mask(pooling_result.assignment)
        return None

    def run(
        self,
        networks: Dict[str, FlowNetwork],
        node_attributes: Optional[Dict[Any, Any]] = None,
        metrics: Optional[list[str]] = None,
    ) -> PipelineResult:
        # Stage 1: pooling (uses the full time range)
        pooling_result = self.pooler.pool(networks, node_attributes=node_attributes)
        od_series = pooling_result.od_series

        # Stage 2: split + fit + predict
        train, val, _test = split_od_series(od_series, self.split)
        self.forecaster.fit(train, val_series=val)
        forecast = self.forecaster.predict(od_series)

        # align ground truth to the forecast horizon tail
        h = self.forecaster.output_len
        if forecast.ground_truth.shape[0] != h:
            forecast.ground_truth = od_series.matrix[-h:]

        # Stage 3: evaluate
        from talent_flow.evaluation import ForecastEvaluator

        core_mask = self._core_mask(pooling_result)
        prev = od_series.matrix[-h - 1]
        eval_metrics = ForecastEvaluator(metrics=metrics).evaluate(
            forecast, core_mask=core_mask, prev_values=prev
        )
        return PipelineResult(
            pooling=pooling_result,
            forecast=forecast,
            metrics=eval_metrics,
            core_mask=core_mask,
        )
