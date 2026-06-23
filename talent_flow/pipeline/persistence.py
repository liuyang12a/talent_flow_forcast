#!/usr/bin/env python3
"""Persistence for PoolingResult and ForecastResult.

Lets the two stages run (and be debugged) independently: a pooler writes its
result to disk; a forecaster reads it back later, with no in-memory coupling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from talent_flow.core import (
    AssignmentMatrix,
    ForecastResult,
    ODMatrixSeries,
    PoolingQualityMetrics,
    PoolingResult,
)
from talent_flow.utils.io import save_json, load_json, ensure_dir


class PoolingResultStore:
    """Save/load a :class:`PoolingResult` to a directory."""

    def save(self, result: PoolingResult, dir_path: str | Path) -> Path:
        d = ensure_dir(dir_path)
        np.save(str(d / "od_matrix.npy"), result.od_series.matrix)
        # timestamps / supernode ids / metadata as JSON
        meta = {
            "timestamps": list(result.od_series.timestamps),
            "supernode_ids": [str(s) for s in result.od_series.supernode_ids],
            "od_metadata": result.od_series.metadata,
            "pooler_name": result.pooler_name,
            "config": result.config,
            "quality": result.quality.to_dict(),
            "assignment": {
                "original_node_ids": [str(n) for n in result.assignment.original_node_ids],
                "supernode_ids": [str(s) for s in result.assignment.supernode_ids],
                "is_soft": result.assignment.is_soft,
            },
        }
        save_json(meta, d / "metadata.json")
        np.savez_compressed(
            str(d / "assignment.npz"), S=result.assignment.S
        )
        return d

    def load(self, dir_path: str | Path) -> PoolingResult:
        d = Path(dir_path)
        meta = load_json(d / "metadata.json")
        matrix = np.load(str(d / "od_matrix.npy"))
        S = np.load(str(d / "assignment.npz"))["S"]
        assignment = AssignmentMatrix(
            S=S,
            original_node_ids=meta["assignment"]["original_node_ids"],
            supernode_ids=meta["assignment"]["supernode_ids"],
            is_soft=meta["assignment"]["is_soft"],
        )
        od_series = ODMatrixSeries(
            matrix=matrix,
            timestamps=meta["timestamps"],
            supernode_ids=meta["supernode_ids"],
            metadata=meta.get("od_metadata", {}),
        )
        quality = PoolingQualityMetrics(**meta.get("quality", {}))
        return PoolingResult(
            od_series=od_series,
            assignment=assignment,
            quality=quality,
            config=meta.get("config", {}),
            pooler_name=meta.get("pooler_name", ""),
        )


class ForecastResultStore:
    """Save/load a :class:`ForecastResult` to a directory."""

    def save(self, result: ForecastResult, dir_path: str | Path) -> Path:
        d = ensure_dir(dir_path)
        np.save(str(d / "predictions.npy"), result.predictions)
        np.save(str(d / "ground_truth.npy"), result.ground_truth)
        meta: Dict[str, Any] = {
            "forecaster_name": result.forecaster_name,
            "metadata": result.metadata,
            "timestamps": result.timestamps,
        }
        if result.prediction_intervals is not None:
            intervals = {}
            for k, v in result.prediction_intervals.items():
                intervals[k] = v.tolist() if isinstance(v, np.ndarray) else v
            meta["prediction_intervals"] = intervals
        save_json(meta, d / "metadata.json")
        return d

    def load(self, dir_path: str | Path) -> ForecastResult:
        d = Path(dir_path)
        meta = load_json(d / "metadata.json")
        preds = np.load(str(d / "predictions.npy"))
        gt = np.load(str(d / "ground_truth.npy"))
        intervals = meta.get("prediction_intervals")
        if intervals:
            for k in ("lower", "upper"):
                if k in intervals:
                    intervals[k] = np.asarray(intervals[k])
        return ForecastResult(
            predictions=preds,
            ground_truth=gt,
            prediction_intervals=intervals,
            timestamps=meta.get("timestamps"),
            forecaster_name=meta.get("forecaster_name", ""),
            metadata=meta.get("metadata", {}),
        )
