#!/usr/bin/env python3
"""Pooling-stage intrinsic quality evaluation.

Mirrors the framework in ``pooling_evaluation_framework.md``:
densification, information retention, structure preservation, clustering
quality, scale.

Large-N handling: when the original graph has many more nodes than
``spectral_max_nodes``, reconstruction / spectral metrics are computed on a
*stratified* sample that guarantees every super-node (especially small core
super-nodes) is represented, rather than a uniform sample that would miss the
core entirely for core-periphery-style assignments. Modularity is always
computed on the full merged graph via a sparse edge pass (O(edges)), so it is
unaffected by sampling.
"""

from __future__ import annotations

from typing import Dict, List, Optional

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


def _top_laplacian_eigvals(adj: np.ndarray, k: int) -> np.ndarray:
    """Smallest ``k`` Laplacian eigenvalues of ``adj``, normalized by the
    largest eigenvalue (so spectra of different-scale graphs are comparable).
    Returns an empty array on failure."""
    try:
        from scipy.sparse.csgraph import laplacian
        from scipy.sparse.linalg import eigsh
    except ImportError:
        return np.array([])

    adj = np.asarray(adj, dtype=float)
    if adj.shape[0] <= 1:
        return np.array([])
    L = laplacian(adj)
    k = min(k, adj.shape[0] - 1)
    if k < 1:
        return np.array([])
    try:
        vals = eigsh(L, k=k, which="SM", return_eigenvectors=False)
        vals = np.sort(np.abs(vals))
    except Exception:
        return np.array([])
    # normalize by the largest (last) eigenvalue for cross-scale comparison
    if vals[-1] > 1e-12:
        vals = vals / vals[-1]
    return vals


def _spectral_error(adj_original: np.ndarray, adj_pooled: np.ndarray, k: int = 10) -> float:
    """Relative error of the first ``k`` normalized Laplacian eigenvalues.

    Both spectra are normalized by their own largest eigenvalue before
    comparison, so a small pooled ``K x K`` graph can be compared against a
    larger sampled original subgraph.
    """
    e_o = _top_laplacian_eigvals(adj_original, k)
    e_p = _top_laplacian_eigvals(adj_pooled, k)
    n = min(len(e_o), len(e_p))
    if n == 0:
        return float("nan")
    denom = e_o[:n] + 1e-8
    return float(np.mean(np.abs(e_o[:n] - e_p[:n]) / denom))


def _modularity_sparse(
    networks: Dict[str, FlowNetwork], assignment: AssignmentMatrix
) -> float:
    """Newman modularity of the hard assignment on the full merged graph.

    Computed by a single sparse pass over edges (O(edges)), so it is exact on
    the full graph regardless of N — no sampling. This avoids the bias where a
    uniform sample misses small core super-nodes and collapses Q to 0.
    """
    comm = assignment.node_super  # [N] super-node index per node
    K = assignment.K
    node_to_row = {nid: i for i, nid in enumerate(assignment.original_node_ids)}

    in_sum = np.zeros(K)  # intra-community edge weight
    deg = np.zeros(K)  # out-strength per community
    m = 0.0
    agg = merge_networks(list(networks.values()))
    for src, tgt, w in agg.iter_edges():
        i = node_to_row.get(src)
        j = node_to_row.get(tgt)
        if i is None or j is None:
            continue
        w = float(w)
        m += w
        deg[comm[i]] += w
        if comm[i] == comm[j]:
            in_sum[comm[i]] += w
    if m == 0:
        return 0.0
    Q = 0.0
    for c in range(K):
        Q += in_sum[c] / m - (deg[c] / (2 * m)) ** 2
    return float(Q)


def _stratified_sample(
    assignment: AssignmentMatrix, n_sample: int, seed: int = 0
) -> np.ndarray:
    """Row indices of original nodes to keep for sample-based metrics.

    Stratified by super-node so that *every* super-node is represented:
      1. all nodes of any super-node with <= per-supernode quota are kept;
      2. larger super-nodes contribute a proportional share up to the budget.

    This guarantees small core super-nodes (which a uniform sample would miss)
    are always included — critical for core-periphery assignments where the
    core is a tiny fraction of N. Falls back to all rows when N <= n_sample.
    """
    N = assignment.N
    if N <= n_sample:
        return np.arange(N)

    comm = assignment.node_super
    K = assignment.K
    rng = np.random.default_rng(seed)

    # group node rows by community
    groups: List[np.ndarray] = []
    for c in range(K):
        idx = np.where(comm == c)[0]
        if len(idx) > 0:
            groups.append(idx)

    # first pass: keep all small communities (size <= per-supernode fair share)
    fair = max(1, n_sample // max(1, len(groups)))
    keep = []
    remaining_budget = n_sample
    big_groups = []
    for g in groups:
        if len(g) <= fair:
            keep.append(g)
            remaining_budget -= len(g)
        else:
            big_groups.append(g)

    # second pass: distribute remaining budget proportionally among big groups
    if big_groups:
        total_big = sum(len(g) for g in big_groups)
        for g in big_groups:
            q = max(1, int(round(remaining_budget * len(g) / total_big)))
            q = min(q, len(g))
            keep.append(rng.choice(g, size=q, replace=False))

    out = np.sort(np.concatenate(keep))
    # final cap (rounding may slightly overshoot)
    if len(out) > n_sample:
        out = np.sort(rng.choice(out, size=n_sample, replace=False))
    return out


class PoolingQualityEvaluator:
    """Compute :class:`PoolingQualityMetrics` for a pooling result."""

    def __init__(self, n_eigenvalues: int = 10, spectral_max_nodes: int = 500):
        self.n_eigenvalues = n_eigenvalues
        # Budget for the stratified sample used by reconstruction / spectral
        # metrics when N is large. Modularity is computed on the full graph
        # (sparse) and is unaffected by this cap.
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

        # --- reconstruction error / spectral (sample-based when N is large) ---
        # Stratified sample guarantees every super-node is represented, avoiding
        # the uniform-sample bias that misses small core super-nodes.
        if N <= self.spectral_max_nodes:
            keep = np.arange(N)
        else:
            keep = _stratified_sample(assignment, self.spectral_max_nodes)
        keep_ids = [node_ids[i] for i in keep]
        keep_to_row = {nid: i for i, nid in enumerate(keep_ids)}
        adj_sub = _dense_adjacency(agg, keep_ids, keep_to_row)
        # reconstruct the sampled sub-adjacency from the pooled K x K: for a
        # hard assignment, recon[i,j] = adj_pooled[node_super[i], node_super[j]]
        ns_sub = assignment.node_super[keep]
        recon = adj_pooled[np.ix_(ns_sub, ns_sub)]
        recon_err = float(
            np.linalg.norm(adj_sub - recon, "fro")
            / (np.linalg.norm(adj_sub, "fro") + 1e-8)
        )
        spec_err = _spectral_error(adj_sub, adj_pooled, self.n_eigenvalues)

        # --- modularity: full-graph sparse pass (exact, no sampling) ---
        mod = _modularity_sparse(networks, assignment)

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
        comm = assignment.node_super
        K = assignment.K
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
