#!/usr/bin/env python3
"""Core subgraph truncation pooling.

Keeps the top-N most active nodes (by aggregated degree/flow) as individual
super-nodes (one-to-one identity) and drops all other nodes. This is the
legacy "dense subgraph" baseline: very dense output, but loses all long-tail
information.
"""

from __future__ import annotations

from typing import Any, Dict

from talent_flow.core import AssignmentMatrix, FlowNetwork
from talent_flow.core.registry import POOLER_REGISTRY
from .base import BasePooler
from .assignment import build_hard_assignment


@POOLER_REGISTRY.register("truncation")
class TruncationPooler(BasePooler):
    """Keep the top-N nodes by total degree; drop the rest.

    Config:
        n_core: number of top nodes to keep.
        rank_by: ``"degree"`` (in+out degree) or ``"flow"`` (in+out weight).
    """

    name = "truncation"

    def __init__(self, n_core: int = 50, rank_by: str = "flow", **kwargs):
        super().__init__(n_core=n_core, rank_by=rank_by, **kwargs)
        self.n_core = n_core
        self.rank_by = rank_by

    def build_assignment(self, networks) -> AssignmentMatrix:
        scores = self.node_degree_scores(networks)
        all_nodes = sorted(scores.keys(), key=lambda x: (isinstance(x, str), x))
        ranked = sorted(all_nodes, key=lambda n: scores[n], reverse=True)
        core = ranked[: self.n_core]
        node_to_cluster = {n: n for n in core}
        return build_hard_assignment(node_to_cluster, core, list(core))
