#!/usr/bin/env python3
"""Base class for all pooling methods.

A pooler only needs to implement :meth:`build_assignment`, which produces the
time-invariant node->super-node assignment matrix ``S``. The generic
:meth:`pool` flow then:
  1. aggregates every monthly adjacency via ``S^T A_t S`` -> ODMatrixSeries
  2. evaluates intrinsic quality via the evaluation framework

All poolers share this flow, guaranteeing a consistent downstream interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from talent_flow.core import (
    AssignmentMatrix,
    FlowNetwork,
    ODMatrixSeries,
    PoolingQualityMetrics,
    PoolingResult,
)
from talent_flow.core.flow_network import merge_networks


class BasePooler(ABC):
    """Abstract base for network pooling methods."""

    name: str = "base"

    def __init__(self, **config: Any):
        self.config: Dict[str, Any] = config

    @abstractmethod
    def build_assignment(
        self, networks: Dict[str, FlowNetwork]
    ) -> AssignmentMatrix:
        """Build the time-invariant assignment matrix ``S`` (N x K).

        This is the *only* method subclasses must implement. ``S`` must be
        computed from a time-aggregated graph or static attributes so that it
        does not vary across time steps.
        """
        raise NotImplementedError

    def pool(
        self,
        networks: Dict[str, FlowNetwork],
        node_attributes: Optional[Dict[Any, Any]] = None,
    ) -> PoolingResult:
        """Run the full pooling flow: assignment -> aggregation -> quality."""
        assignment = self.build_assignment(networks)
        od_series = self._aggregate(networks, assignment)
        quality = self._evaluate_quality(
            networks, assignment, od_series, node_attributes
        )
        return PoolingResult(
            od_series=od_series,
            assignment=assignment,
            quality=quality,
            config=dict(self.config),
            pooler_name=self.name,
        )

    # ---- generic helpers (subclasses normally do not override) ----

    def _aggregate(
        self, networks: Dict[str, FlowNetwork], assignment: AssignmentMatrix
    ) -> ODMatrixSeries:
        """Aggregate every monthly adjacency via ``S^T A_t S`` -> ODMatrixSeries.

        Implemented sparsely: iterates over each network's edges (rather than
        materializing an N x N dense adjacency), accumulating the
        contribution ``S[i] outer S[j] * weight`` into the K x K OD matrix.
        This keeps the operation tractable even when N (original nodes) is
        very large, since the assignment is sparse (each original node maps
        to exactly one super-node in the hard-assignment case).
        """
        S = assignment.S
        N, K = S.shape
        node_ids = list(assignment.original_node_ids)
        node_to_row = {nid: i for i, nid in enumerate(node_ids)}
        months = sorted(networks.keys())
        T = len(months)
        matrix = np.zeros((T, K, K), dtype=float)

        # Precompute, per original node, its super-node index (hard assignment)
        # or its soft row (soft assignment).
        if assignment.is_soft:
            node_super_rows = {i: S[i] for i in range(N)}
        else:
            node_super = np.argmax(S, axis=1)  # [N] super-node index per node
            node_super_rows = None

        for t, month in enumerate(months):
            net = networks[month]
            for src, tgt, w in net.iter_edges():
                i = node_to_row.get(src)
                j = node_to_row.get(tgt)
                if i is None or j is None:
                    continue  # edge incident to a dropped node
                if node_super_rows is not None:
                    # soft assignment: outer product contribution
                    matrix[t] += float(w) * np.outer(S[i], S[j])
                else:
                    si, sj = int(node_super[i]), int(node_super[j])
                    matrix[t, si, sj] += float(w)
        return ODMatrixSeries(
            matrix=matrix,
            timestamps=months,
            supernode_ids=list(assignment.supernode_ids),
            metadata={"pooler_name": self.name, "N": N, "K": K, "T": T},
        )

    def _evaluate_quality(
        self,
        networks: Dict[str, FlowNetwork],
        assignment: AssignmentMatrix,
        od_series: ODMatrixSeries,
        node_attributes: Optional[Dict[Any, Any]],
    ) -> PoolingQualityMetrics:
        from talent_flow.evaluation import PoolingQualityEvaluator

        return PoolingQualityEvaluator().evaluate(
            networks, assignment, od_series, original_node_attributes=node_attributes
        )

    # ---- utility for subclasses ----

    @staticmethod
    def aggregated_network(
        networks: Dict[str, FlowNetwork]
    ) -> FlowNetwork:
        """Time-aggregated network (sum of all months)."""
        return merge_networks(list(networks.values()))

    @staticmethod
    def node_degree_scores(networks: Dict[str, FlowNetwork]) -> Dict[Any, float]:
        """Per-node total degree (in+out weighted) in the aggregated graph.

        Single pass over all edges of all months — O(total edges) — avoiding
        the O(N) per-node lookups that are prohibitive for large N.
        """
        scores: Dict[Any, float] = {}
        for net in networks.values():
            for src, tgt, w in net.iter_edges():
                scores[src] = scores.get(src, 0.0) + float(w)
                scores[tgt] = scores.get(tgt, 0.0) + float(w)
        return scores
