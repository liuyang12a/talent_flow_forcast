"""Core data structures and contracts for the talent flow pipeline.

This package is the lowest layer: it defines :class:`FlowNetwork` (the raw
sparse graph) and the dataclasses (:mod:`contracts`) that couple the pooling
and forecasting stages, plus the plugin :mod:`registry`.
"""

from .flow_network import FlowNetwork, merge_networks
from .contracts import (
    ODMatrixSeries,
    AssignmentMatrix,
    PoolingQualityMetrics,
    PoolingResult,
    ForecastResult,
    NodeId,
)
from .registry import Registry, POOLER_REGISTRY, FORECASTER_REGISTRY

__all__ = [
    "FlowNetwork",
    "merge_networks",
    "ODMatrixSeries",
    "AssignmentMatrix",
    "PoolingQualityMetrics",
    "PoolingResult",
    "ForecastResult",
    "NodeId",
    "Registry",
    "POOLER_REGISTRY",
    "FORECASTER_REGISTRY",
]
