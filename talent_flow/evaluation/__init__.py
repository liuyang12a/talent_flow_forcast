"""Unified evaluation framework.

Two layers:
- :mod:`pooling_eval` : intrinsic pooling quality (densification, structure,
  clustering, scale).
- :mod:`forecast_eval` : forecast accuracy with optional core/periphery split
  and probabilistic calibration.

All metrics use the ``(target, prediction)`` argument order.
"""

from .metrics import (
    METRIC_REGISTRY,
    DEFAULT_METRICS,
    calculate_metrics,
    mae,
    mse,
    rmse,
    mape,
    wape,
    smape,
    r2_score,
    correlation,
    directional_accuracy,
)
from .pooling_eval import PoolingQualityEvaluator
from .forecast_eval import ForecastEvaluator
from .probabilistic import ProbabilisticEvaluator, picp, pinaw, crps_gaussian
from .significance import SignificanceTester, paired_t_test, wilcoxon
from .report import EvaluationReport, ReportGenerator, pooling_quality_to_dict

__all__ = [
    "METRIC_REGISTRY",
    "DEFAULT_METRICS",
    "calculate_metrics",
    "mae",
    "mse",
    "rmse",
    "mape",
    "wape",
    "smape",
    "r2_score",
    "correlation",
    "directional_accuracy",
    "PoolingQualityEvaluator",
    "ForecastEvaluator",
    "ProbabilisticEvaluator",
    "picp",
    "pinaw",
    "crps_gaussian",
    "SignificanceTester",
    "paired_t_test",
    "wilcoxon",
    "EvaluationReport",
    "ReportGenerator",
    "pooling_quality_to_dict",
]
