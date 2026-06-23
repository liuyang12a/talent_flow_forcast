#!/usr/bin/env python3
"""Evaluation report aggregation and comparison-table generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from talent_flow.core import PoolingQualityMetrics


@dataclass
class EvaluationReport:
    """A single method's evaluation outcome (pooling and/or forecast)."""

    method_name: str
    method_type: str  # "pooling" | "forecasting" | "pipeline"
    pooling_quality: Optional[Dict[str, Any]] = None
    forecast_metrics: Optional[Dict[str, Dict[str, float]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReportGenerator:
    """Build comparison tables across multiple :class:`EvaluationReport`s."""

    def generate_pooling_table(
        self, reports: List[EvaluationReport]
    ) -> "list[dict]":
        rows = []
        for r in reports:
            q = r.pooling_quality or {}
            rows.append(
                {
                    "method": r.method_name,
                    "K": q.get("pooled_K"),
                    "compression": q.get("compression_ratio"),
                    "density_improvement": q.get("density_improvement_ratio"),
                    "zero_reduction": q.get("zero_reduction"),
                    "reconstruction_error": q.get("reconstruction_error"),
                    "spectral_error": q.get("spectral_error"),
                    "modularity": q.get("modularity"),
                }
            )
        return rows

    def generate_forecast_table(
        self, reports: List[EvaluationReport], layer: str = "overall"
    ) -> "list[dict]":
        rows = []
        for r in reports:
            fm = r.forecast_metrics or {}
            layer_metrics = fm.get(layer, {})
            row = {"method": r.method_name}
            row.update(layer_metrics)
            rows.append(row)
        return rows

    def to_markdown(self, rows: List[Dict[str, Any]]) -> str:
        if not rows:
            return ""
        cols = list(rows[0].keys())
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        lines = [header, sep]
        for r in rows:
            lines.append(
                "| "
                + " | ".join(_fmt(r.get(c)) for c in cols)
                + " |"
            )
        return "\n".join(lines)


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if np.isnan(v):
            return ""
        return f"{v:.4f}"
    return str(v)


def pooling_quality_to_dict(q: PoolingQualityMetrics) -> Dict[str, Any]:
    return q.to_dict()
