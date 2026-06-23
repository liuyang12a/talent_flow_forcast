#!/usr/bin/env python3
"""
Core data contracts connecting the Pooling and Forecasting stages.

These dataclasses are the *only* coupling between the two stages. Both stages
agree on these types so that any pooler can feed any forecaster.

Design rationale
----------------
- The pooler emits an :class:`ODMatrixSeries` ``[T, K, K]`` (node-centric OD
  matrix sequence over K super-nodes) plus an :class:`AssignmentMatrix`
  describing how original nodes map to super-nodes.
- The forecaster consumes :class:`ODMatrixSeries` and emits a
  :class:`ForecastResult`.
- The evaluation framework consumes either stage's output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np

NodeId = Union[int, str]


@dataclass
class ODMatrixSeries:
    """A sequence of K x K origin-destination (OD) matrices over T time steps.

    This is the unified output of the pooling stage and the unified input of
    the forecasting stage.

    Attributes:
        matrix: array of shape ``[T, K, K]``; ``matrix[t, i, j]`` is the flow
            from super-node *i* to super-node *j* at time step *t*.
        timestamps: length-T list of timestamp strings (e.g. ``"YYYY-MM"``).
        supernode_ids: length-K list of super-node identifiers.
        metadata: optional bag of provenance info (pooler name, labels, ...).
    """

    matrix: np.ndarray
    timestamps: List[str]
    supernode_ids: List[Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=np.float64)
        if self.matrix.ndim != 3:
            raise ValueError(
                f"matrix must be 3D [T, K, K], got shape {self.matrix.shape}"
            )
        t, k1, k2 = self.matrix.shape
        if k1 != k2:
            raise ValueError(
                f"OD matrix must be square, got shape {self.matrix.shape}"
            )
        if len(self.timestamps) != t:
            raise ValueError(
                f"timestamps length {len(self.timestamps)} != T={t}"
            )
        if len(self.supernode_ids) != k1:
            raise ValueError(
                f"supernode_ids length {len(self.supernode_ids)} != K={k1}"
            )

    @property
    def T(self) -> int:
        """Number of time steps."""
        return self.matrix.shape[0]

    @property
    def K(self) -> int:
        """Number of super-nodes."""
        return self.matrix.shape[1]

    def slice(self, start: int, end: int) -> "ODMatrixSeries":
        """Return a temporal sub-slice ``[start:end]`` as a new series."""
        return ODMatrixSeries(
            matrix=self.matrix[start:end].copy(),
            timestamps=list(self.timestamps[start:end]),
            supernode_ids=list(self.supernode_ids),
            metadata=dict(self.metadata),
        )


@dataclass
class AssignmentMatrix:
    """Mapping from original nodes to super-nodes, ``S in R^{N x K}``.

    For hard assignment each row has a single 1; for soft assignment each row
    is a probability distribution. Used for de-pooling and quality evaluation.

    Attributes:
        S: array of shape ``[N, K]``.
        original_node_ids: length-N list of original node identifiers.
        supernode_ids: length-K list of super-node identifiers.
        is_soft: whether S is a soft (probabilistic) assignment.
    """

    S: np.ndarray
    original_node_ids: List[Any]
    supernode_ids: List[Any]
    is_soft: bool = False

    def __post_init__(self) -> None:
        self.S = np.asarray(self.S, dtype=np.float64)
        if self.S.ndim != 2:
            raise ValueError(f"S must be 2D [N, K], got shape {self.S.shape}")
        n, k = self.S.shape
        if len(self.original_node_ids) != n:
            raise ValueError(
                f"original_node_ids length {len(self.original_node_ids)} != N={n}"
            )
        if len(self.supernode_ids) != k:
            raise ValueError(
                f"supernode_ids length {len(self.supernode_ids)} != K={k}"
            )


@dataclass
class PoolingQualityMetrics:
    """Intrinsic quality metrics of a pooling result.

    Fields may be ``None`` when not computed. See ``evaluation/pooling_eval.py``
    for computation. Mirrors the framework in ``pooling_evaluation_framework.md``.
    """

    # Densification
    original_density: Optional[float] = None
    pooled_density: Optional[float] = None
    density_improvement_ratio: Optional[float] = None
    zero_reduction: Optional[float] = None
    # Information retention
    reconstruction_error: Optional[float] = None
    # Structure preservation
    spectral_error: Optional[float] = None
    modularity: Optional[float] = None
    # Clustering quality
    cluster_homogeneity: Optional[float] = None
    # Scale
    original_N: Optional[int] = None
    pooled_K: Optional[int] = None
    compression_ratio: Optional[float] = None
    # Free-form extras
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class PoolingResult:
    """Complete output of the pooling stage."""

    od_series: ODMatrixSeries
    assignment: AssignmentMatrix
    quality: PoolingQualityMetrics
    config: Dict[str, Any] = field(default_factory=dict)
    pooler_name: str = ""


@dataclass
class ForecastResult:
    """Unified output of the forecasting stage.

    Attributes:
        predictions: array of shape ``[h, K, K]`` (point forecasts).
        ground_truth: array of shape ``[h, K, K]`` (actuals, for evaluation).
        prediction_intervals: optional dict with ``"lower"``, ``"upper"`` arrays
            of shape ``[h, K, K]`` and a ``"level"`` float.
        timestamps: optional length-h list of forecast timestamps.
        forecaster_name: name of the forecaster that produced this result.
        metadata: free-form provenance.
    """

    predictions: np.ndarray
    ground_truth: np.ndarray
    prediction_intervals: Optional[Dict[str, Any]] = None
    timestamps: Optional[List[str]] = None
    forecaster_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.predictions = np.asarray(self.predictions, dtype=np.float64)
        self.ground_truth = np.asarray(self.ground_truth, dtype=np.float64)
        if self.predictions.shape != self.ground_truth.shape:
            raise ValueError(
                f"predictions shape {self.predictions.shape} != "
                f"ground_truth shape {self.ground_truth.shape}"
            )
        if self.prediction_intervals is not None:
            for key in ("lower", "upper"):
                if key in self.prediction_intervals:
                    self.prediction_intervals[key] = np.asarray(
                        self.prediction_intervals[key], dtype=np.float64
                    )
