#!/usr/bin/env python3
"""Community-detection pooling (Louvain / Leiden).

Aggregates nodes into super-nodes by Louvain communities detected on the
time-aggregated graph. Produces a topology-aware, interpretable coarsening
with good modularity.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from talent_flow.core import AssignmentMatrix, FlowNetwork
from talent_flow.core.registry import POOLER_REGISTRY
from .base import BasePooler
from .assignment import build_hard_assignment


@POOLER_REGISTRY.register("louvain")
class LouvainPooler(BasePooler):
    """Pool nodes by Louvain communities on the aggregated graph.

    Config:
        resolution: Louvain resolution parameter (higher -> more, smaller
            communities).
        min_community_size: communities smaller than this are merged into a
            single ``"other"`` super-node.
    """

    name = "louvain"

    def __init__(
        self,
        resolution: float = 1.0,
        min_community_size: int = 5,
        **kwargs,
    ):
        super().__init__(
            resolution=resolution, min_community_size=min_community_size, **kwargs
        )
        self.resolution = resolution
        self.min_community_size = min_community_size

    def build_assignment(self, networks) -> AssignmentMatrix:
        import networkx as nx

        try:
            import community as community_louvain
        except ImportError as e:
            raise ImportError(
                "python-louvain is required for LouvainPooler. "
                "Install via: uv add 'python-louvain>=0.16'"
            ) from e

        agg = self.aggregated_network(networks)
        G = nx.DiGraph()
        for src, tgt, w in agg.iter_edges():
            G.add_edge(src, tgt, weight=float(w))
        # Louvain needs an undirected graph
        G_und = nx.Graph()
        for u, v, data in G.edges(data=True):
            w = data.get("weight", 1.0)
            if G_und.has_edge(u, v):
                G_und[u][v]["weight"] += w
            else:
                G_und.add_edge(u, v, weight=w)

        partition = community_louvain.best_partition(
            G_und, resolution=self.resolution
        )

        # build community lists
        comm_members: Dict[int, list] = {}
        for node, cid in partition.items():
            comm_members.setdefault(cid, []).append(node)

        # relabel communities; merge small ones into "other"
        all_nodes = sorted(partition.keys(), key=lambda x: (isinstance(x, str), x))
        node_to_cluster: Dict[Any, Any] = {}
        cluster_labels: list = []
        for cid, members in comm_members.items():
            if len(members) >= self.min_community_size:
                label = f"comm_{cid}"
                cluster_labels.append(label)
                for n in members:
                    node_to_cluster[n] = label
            else:
                for n in members:
                    node_to_cluster[n] = "other"
        if "other" in node_to_cluster.values() and "other" not in cluster_labels:
            cluster_labels.append("other")

        return build_hard_assignment(node_to_cluster, all_nodes, cluster_labels)
