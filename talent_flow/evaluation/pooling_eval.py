#!/usr/bin/env python3
"""Pooling-stage intrinsic quality evaluation.

Mirrors the framework in ``pooling_evaluation_framework.md``:
densification, information retention, structure preservation, clustering
quality, scale.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from talent_flow.core import (
    AssignmentMatrix,
    FlowNetwork,
    ODMatrixSeries,
    PoolingQualityMetrics,
)
from talent_flow.core.flow_network import merge_networks


def _density(n_nodes: int, n_edges: int) -> float:
    if n_nodes <= 1:
        return 0.0
    possible = n_nodes * (n_nodes - 1)
    return n_edges / possible if possible > 0 else 0.0


def _spectral_error(
    adj_original: np.ndarray, adj_pooled: np.ndarray, k: int = 10
) -> float:
    """Relative error of the first ``k`` Laplacian eigenvalues."""
    try:
        from scipy.sparse.csgraph import laplacian
        from scipy.sparse.linalg import eigsh
    except ImportError:
        return float("nan")

    def _top_eigvals(adj: np.ndarray, k: int) -> np.ndarray:
        adj = np.asarray(adj, dtype=float)
        L = laplacian(adj)
        k = min(k, adj.shape[0] - 1)
        if k < 1:
            return np.array([0.0])
        # eigsh on a dense array needs it positive-semidefinite; use shift-invert.
        try:
            vals = eigsh(L, k=k, which="SM", return_eigenvectors=False)
            return np.sort(vals)
        except Exception:
            return np.array([0.0])

    e_o = _top_eigvals(adj_original, k)
    e_p = _top_eigvals(adj_pooled, k)
    n = min(len(e_o), len(e_p))
    if n == 0:
        return float("nan")
    denom = e_o[:n] + 1e-8
    return float(np.mean(np.abs(e_o[:n] - e_p[:n]) / denom))


def _modularity(adj: np.ndarray, assignment: AssignmentMatrix) -> float:
    """Modularity of the hard assignment on the (aggregated) graph."""
    A = np.asarray(adj, dtype=float)
    n = A.shape[0]
    if n == 0:
        return 0.0
    m = A.sum()
    if m == 0:
        return 0.0
    deg = A.sum(axis=1)
    # communities: argmax of S rows (hard assignment)
    S = assignment.S
    comm = np.argmax(S, axis=1)
    K = S.shape[1]
    Q = 0.0
    for c in range(K):
        idx = np.where(comm == c)[0]
        if len(idx) == 0:
            continue
        in_sum = A[np.ix_(idx, idx)].sum()
        deg_sum = deg[idx].sum()
        Q += in_sum / m - (deg_sum / (2 * m)) ** 2
    return float(Q)


class PoolingQualityEvaluator:
    """Compute :class:`PoolingQualityMetrics` for a pooling result."""

    def __init__(self, n_eigenvalues: int = 10):
        self.n_eigenvalues = n_eigenvalues

    def evaluate(
        self,
        networks: Dict[str, FlowNetwork],
        assignment: AssignmentMatrix,
        od_series: ODMatrixSeries,
        original_node_attributes: Optional[Dict] = None,
    ) -> PoolingQualityMetrics:
        # --- aggregated original graph on the original node set ---
        agg = merge_networks(list(networks.values()))
        N = len(assignment.original_node_ids)
        # restrict to nodes present in the assignment
        node_ids = list(assignment.original_node_ids)
        adj_orig_list, _ = agg.to_adjacency_matrix(node_order=node_ids)
        adj_orig = np.array(adj_orig_list, dtype=float)
        n_edges_orig = int((adj_orig > 0).sum())
        density_orig = _density(N, n_edges_orig)

        # --- pooled graph (time-averaged OD) ---
        adj_pooled = od_series.matrix.mean(axis=0)
        K = adj_pooled.shape[0]
        n_edges_pooled = int((adj_pooled > 0).sum())
        density_pooled = _density(K, n_edges_pooled)

        density_ratio = (
            density_pooled / density_orig if density_orig > 0 else float("nan")
        )
        zero_orig = 1.0 - density_orig
        zero_pooled = 1.0 - density_pooled
        zero_red = (zero_orig - zero_pooled) / zero_orig if zero_orig > 0 else 0.0

        # --- reconstruction error: ||A - S A' S^T||_F / ||A||_F ---
        S = assignment.S
        recon = S @ adj_pooled @ S.T
        recon_err = float(
            np.linalg.norm(adj_orig - recon, "fro")
            / (np.linalg.norm(adj_orig, "fro") + 1e-8)
        )

        # --- spectral preservation ---
        spec_err = _spectral_error(adj_orig, adj_pooled, self.n_eigenvalues)

        # --- modularity ---
        mod = _modularity(adj_orig, assignment)

        # --- cluster homogeneity (if attributes provided) ---
        homog = self._cluster_homogeneity(assignment, original_node_attributes)

        return PoolingQualityMetrics(
            original_density=density_orig,
            pooled_density=density_pooled,
            density_improvement_ratio=density_ratio,
            zero_reduction=zero_red,
            reconstruction_error=recon_err,
            spectral_error=spec_err,
            modularity=mod,
            cluster_homogeneity=homog,
            original_N=N,
            pooled_K=K,
            compression_ratio=K / N if N > 0 else float("nan"),
        )

    @staticmethod
    def _cluster_homogeneity(
        assignment: AssignmentMatrix, attributes: Optional[Dict]
    ) -> Optional[float]:
        if attributes is None:
            return None
        comm = np.argmax(assignment.S, axis=1)
        K = assignment.S.shape[1]
        scores = []
        for c in range(K):
            idx = np.where(comm == c)[0]
            if len(idx) == 0:
                continue
            cats = [
                attributes.get(assignment.original_node_ids[i])
                for i in idx
            ]
            cats = [c for c in cats if c is not None]
            if not cats:
                continue
            from collections import Counter

            most_common = Counter(cats).most_common(1)[0][1]
            scores.append(most_common / len(cats))
        return float(np.mean(scores)) if scores else None
