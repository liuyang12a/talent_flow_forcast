#!/usr/bin/env python3
"""Adapter wrapping the existing DenseSubgraphExtractor as a BasePooler.

The legacy :class:`DenseSubgraphExtractor` (in ``src/data/dense_subgraph.py``)
selects a spatio-temporally dense core node set and produces an edge-centric
tensor. This adapter reuses its core-node selection logic but emits the
node-centric :class:`ODMatrixSeries` contract: each retained core node becomes
its own atomic super-node (identity assignment), and the OD matrix is built
by restricting each monthly adjacency to the core node set.

This preserves the proven three-stage densification algorithm while fitting
the new two-stage pipeline contract.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from talent_flow.core import AssignmentMatrix, FlowNetwork, ODMatrixSeries
from talent_flow.core.registry import POOLER_REGISTRY
from .base import BasePooler
from .assignment import build_hard_assignment


@POOLER_REGISTRY.register("dense_subgraph")
class DenseSubgraphPooler(BasePooler):
    """Pool by the legacy dense-subgraph core-node selection.

    Config mirrors :class:`DenseSubgraphConfig` from the legacy module. The
    adapter only uses the *core node selection* (Stages A-B) and a light
    temporal filter; it then aggregates monthly adjacencies over the core
    nodes into a ``[T, K, K]`` OD matrix.
    """

    name = "dense_subgraph"

    def __init__(
        self,
        spatial_strategy: str = "flow_core",
        max_nodes: int = 200,
        min_nodes: int = 20,
        target_coverage: float = 0.80,
        min_activity_ratio: float = 0.30,
        max_allowed_gap: int = 12,
        min_temporal_score: float = 0.25,
        **kwargs,
    ):
        super().__init__(
            spatial_strategy=spatial_strategy,
            max_nodes=max_nodes,
            min_nodes=min_nodes,
            target_coverage=target_coverage,
            min_activity_ratio=min_activity_ratio,
            max_allowed_gap=max_allowed_gap,
            min_temporal_score=min_temporal_score,
            **kwargs,
        )
        self.spatial_strategy = spatial_strategy
        self.max_nodes = max_nodes
        self.min_nodes = min_nodes
        self.target_coverage = target_coverage
        self.min_activity_ratio = min_activity_ratio
        self.max_allowed_gap = max_allowed_gap
        self.min_temporal_score = min_temporal_score

    def _build_extractor(self):
        # import the legacy module directly from its file path (the legacy
        # ``src/`` tree is not a regular package).
        import importlib.util
        from pathlib import Path

        legacy_file = (
            Path(__file__).resolve().parents[2] / "src" / "data" / "dense_subgraph.py"
        )
        if not legacy_file.exists():
            raise FileNotFoundError(f"legacy dense_subgraph.py not found: {legacy_file}")
        spec = importlib.util.spec_from_file_location(
            "legacy_dense_subgraph", legacy_file
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg = mod.DenseSubgraphConfig(
            spatial_strategy=self.spatial_strategy,
            max_nodes=self.max_nodes,
            min_nodes=self.min_nodes,
            target_coverage=self.target_coverage,
            min_activity_ratio=self.min_activity_ratio,
            max_allowed_gap=self.max_allowed_gap,
            min_temporal_score=self.min_temporal_score,
            tensor_type="edge_centric",
        )
        return mod.DenseSubgraphExtractor(cfg, mod.EdgeCentricTensorBuilder())

    def build_assignment(self, networks) -> AssignmentMatrix:
        extractor = self._build_extractor()
        # Stage A: statistics
        extractor.compute_statistics(networks)
        # Stage B: spatial core selection
        core_nodes = extractor.extract_spatial_core(strategy=self.spatial_strategy)
        if not core_nodes:
            raise RuntimeError("DenseSubgraphPooler: empty core node set")
        core_list = sorted(core_nodes, key=lambda x: (isinstance(x, str), x))
        node_to_cluster = {n: n for n in core_list}
        return build_hard_assignment(node_to_cluster, core_list, list(core_list))
