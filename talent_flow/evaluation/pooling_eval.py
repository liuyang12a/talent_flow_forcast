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


def _dense_adjacency(network: FlowNetwork, node_ids, node_to_row) -> np.ndarray:
    """Build a dense ``[len(node_ids), len(node_ids)]`` adjacency by iterating
    edges. Only edges among ``node_ids`` are included."""
    n = len(node_ids)
    A = np.zeros((n, n), dtype=float)
    for src, tgt, w in network.iter_edges():
        i = node_to_row.get(src)
        j = node_to_row.get(tgt)
        if i is not None and j is not None:
            A[i, j] += float(w)
    return A


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


def _sub_assignment(assignment: AssignmentMatrix, keep_idx: np.ndarray) -> AssignmentMatrix:
    """Return a sub-assignment restricted to rows in ``keep_idx``."""
    return AssignmentMatrix(
        S=assignment.S[keep_idx],
        original_node_ids=[assignment.original_node_ids[i] for i in keep_idx],
        supernode_ids=list(assignment.supernode_ids),
        is_soft=assignment.is_soft,
    )


class PoolingQualityEvaluator:
    """Compute :class:`PoolingQualityMetrics` for a pooling result."""

    def __init__(self, n_eigenvalues: int = 10, spectral_max_nodes: int = 500):
        self.n_eigenvalues = n_eigenvalues
        # Spectral (and reconstruction) ops are O(N^2)/O(N^3); cap the
        # original-graph size used for these expensive metrics. When N exceeds
        # the cap, the original adjacency is subsampled to a random node subset
        # of this size (density/modularity become approximate; spectral skipped
        # entirely if still too large).
        self.spectral_max_nodes = spectral_max_nodes

    def evaluate(
        self,
        networks: Dict[str, FlowNetwork],
        assignment: AssignmentMatrix,
        od_series: ODMatrixSeries,
        original_node_attributes: Optional[Dict] = None,
    ) -> PoolingQualityMetrics:
        # --- aggregated original graph (sparse) ---
        agg = merge_networks(list(networks.values()))
        N = len(assignment.original_node_ids)
        node_ids = list(assignment.original_node_ids)
        node_id_set = set(node_ids)
        node_to_row = {nid: i for i, nid in enumerate(node_ids)}

        # Count original edges among assigned nodes (single pass, no N x N).
        n_edges_orig = 0
        for src, tgt, _w in agg.iter_edges():
            if src in node_id_set and tgt in node_id_set:
                n_edges_orig += 1
        density_orig = _density(N, n_edges_orig)

        # --- pooled graph (time-summed OD, same scale as the aggregated
        # original graph so reconstruction is comparable) ---
        adj_pooled = od_series.matrix.sum(axis=0)
        K = adj_pooled.shape[0]
        n_edges_pooled = int((adj_pooled > 0).sum())
        density_pooled = _density(K, n_edges_pooled)

        density_ratio = (
            density_pooled / density_orig if density_orig > 0 else float("nan")
        )
        zero_orig = 1.0 - density_orig
        zero_pooled = 1.0 - density_pooled
        zero_red = (zero_orig - zero_pooled) / zero_orig if zero_orig > 0 else 0.0

        # --- reconstruction error / modularity / spectral ---
        # Build a *capped* dense adjacency only over a sampled subset of the
        # original nodes when N is large; otherwise use the full set.
        S = assignment.S
        if N <= self.spectral_max_nodes:
            adj_orig = _dense_adjacency(agg, node_ids, node_to_row)
            recon = S @ adj_pooled @ S.T
            recon_err = float(
                np.linalg.norm(adj_orig - recon, "fro")
                / (np.linalg.norm(adj_orig, "fro") + 1e-8)
            )
            spec_err = _spectral_error(adj_orig, adj_pooled, self.n_eigenvalues)
            mod = _modularity(adj_orig, assignment)
        else:
            rng = np.random.default_rng(0)
            keep = rng.choice(N, size=self.spectral_max_nodes, replace=False)
            keep.sort()
            keep_ids = [node_ids[i] for i in keep]
            keep_to_row = {nid: i for i, nid in enumerate(keep_ids)}
            adj_sub = _dense_adjacency(agg, keep_ids, keep_to_row)
            S_sub = S[keep]
            recon = S_sub @ adj_pooled @ S_sub.T
            recon_err = float(
                np.linalg.norm(adj_sub - recon, "fro")
                / (np.linalg.norm(adj_sub, "fro") + 1e-8)
            )
            spec_err = float("nan")
            mod = _modularity(adj_sub, _sub_assignment(assignment, keep))

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
