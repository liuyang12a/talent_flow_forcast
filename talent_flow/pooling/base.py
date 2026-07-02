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
        mode: str = "full",
        assignment: Optional[AssignmentMatrix] = None,
    ) -> PoolingResult:
        """Run the pooling flow in one of three modes.

        Modes:
            ``"full"`` (default): assignment -> aggregate -> evaluate. Returns a
                complete :class:`PoolingResult` with ``od_series`` and ``quality``.
            ``"assignment"``: assignment -> evaluate only. Internally computes a
                time-summed ``K x K`` (``sum_t S^T A_t S``, no ``[T,K,K]``
                materialized) so all quality metrics remain available, but
                ``od_series`` is ``None``. Cheaper in memory (O(K^2) vs
                O(T*K^2)) when the per-month OD series is not needed.
            ``"aggregate"``: aggregate only, given a pre-built ``assignment``.
                Returns ``od_series`` with ``quality=None``. Requires passing
                ``assignment=``.
        """
        if mode == "aggregate":
            if assignment is None:
                raise ValueError("mode='aggregate' requires an assignment=")
            od_series = self._aggregate(networks, assignment)
            return PoolingResult(
                od_series=od_series,
                assignment=assignment,
                quality=None,
                config=dict(self.config),
                pooler_name=self.name,
            )

        assignment = assignment if assignment is not None else self.build_assignment(networks)

        if mode == "assignment":
            # Time-summed K x K only (no [T,K,K]); wrap as a single-step series
            # so the existing evaluator (which does sum(axis=0)) is reusable.
            adj_summed = self._aggregate_summed(networks, assignment)
            od_tmp = ODMatrixSeries(
                matrix=adj_summed[None, :, :],
                timestamps=["__summed__"],
                supernode_ids=list(assignment.supernode_ids),
                metadata={"pooler_name": self.name, "mode": "assignment"},
            )
            quality = self._evaluate_quality(
                networks, assignment, od_tmp, node_attributes
            )
            return PoolingResult(
                od_series=None,
                assignment=assignment,
                quality=quality,
                config=dict(self.config),
                pooler_name=self.name,
            )

        # mode == "full"
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
        node_super = assignment.node_super
        N, K = assignment.N, assignment.K
        node_ids = list(assignment.original_node_ids)
        node_to_row = {nid: i for i, nid in enumerate(node_ids)}
        months = sorted(networks.keys())
        T = len(months)
        matrix = np.zeros((T, K, K), dtype=float)

        for t, month in enumerate(months):
            net = networks[month]
            for src, tgt, w in net.iter_edges():
                i = node_to_row.get(src)
                j = node_to_row.get(tgt)
                if i is None or j is None:
                    continue  # edge incident to a dropped node
                si, sj = int(node_super[i]), int(node_super[j])
                matrix[t, si, sj] += float(w)
        return ODMatrixSeries(
            matrix=matrix,
            timestamps=months,
            supernode_ids=list(assignment.supernode_ids),
            metadata={"pooler_name": self.name, "N": N, "K": K, "T": T},
        )

    def _aggregate_summed(
        self, networks: Dict[str, FlowNetwork], assignment: AssignmentMatrix
    ) -> np.ndarray:
        """Time-summed pooled adjacency ``sum_t (S^T A_t S)`` as a ``K x K``
        array, without materializing the ``[T, K, K]`` OD series.

        Equivalent to ``_aggregate(...).matrix.sum(axis=0)`` (by linearity,
        ``sum_t S^T A_t S == S^T (sum_t A_t) S``) but uses O(K^2) memory
        instead of O(T*K^2). Used by the ``"assignment"`` mode to keep all
        quality metrics available while skipping the per-month OD series.
        """
        node_super = assignment.node_super
        K = assignment.K
        node_ids = list(assignment.original_node_ids)
        node_to_row = {nid: i for i, nid in enumerate(node_ids)}

        adj_summed = np.zeros((K, K), dtype=float)
        for net in networks.values():
            for src, tgt, w in net.iter_edges():
                i = node_to_row.get(src)
                j = node_to_row.get(tgt)
                if i is None or j is None:
                    continue  # edge incident to a dropped node
                si, sj = int(node_super[i]), int(node_super[j])
                adj_summed[si, sj] += float(w)
        return adj_summed

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
