#!/usr/bin/env python3
"""Helpers for building and validating :class:`AssignmentMatrix` objects."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from talent_flow.core import AssignmentMatrix, FlowNetwork


def build_hard_assignment(
    node_to_cluster: Mapping[Any, Any],
    original_node_ids: Sequence[Any],
    supernode_ids: Sequence[Any],
) -> AssignmentMatrix:
    """Build a hard (one-hot) assignment from a node->cluster mapping.

    Args:
        node_to_cluster: maps each original node id to a super-node id.
        original_node_ids: ordered list of original node ids (rows of S).
        supernode_ids: ordered list of super-node ids (columns of S).

    Returns an :class:`AssignmentMatrix` whose ``node_super`` is a compact
    length-N index array (the super-node index of each original node), rather
    than a dense ``N x K`` one-hot matrix.
    """
    super_idx = {sid: j for j, sid in enumerate(supernode_ids)}
    node_super = np.empty(len(original_node_ids), dtype=np.int64)
    for i, node in enumerate(original_node_ids):
        cluster = node_to_cluster.get(node)
        if cluster is None or cluster not in super_idx:
            raise ValueError(
                f"node {node} maps to unknown super-node {cluster!r}"
            )
        node_super[i] = super_idx[cluster]
    return AssignmentMatrix(
        node_super=node_super,
        original_node_ids=list(original_node_ids),
        supernode_ids=list(supernode_ids),
    )


def aggregated_adjacency(
    network: FlowNetwork, node_order: Sequence[Any]
) -> np.ndarray:
    """Return a dense ``[N, N]`` numpy adjacency matrix for ``network``."""
    mat, _ = network.to_adjacency_matrix(node_order=list(node_order))
    return np.array(mat, dtype=float)
