#!/usr/bin/env python3
"""Semantic / attribute-driven pooling.

Aggregates original nodes into super-nodes by a categorical attribute
(industry, geography, ...). The simplest, fully interpretable baseline; the
assignment is 100% time-invariant because the attribute is static.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from talent_flow.core import AssignmentMatrix, FlowNetwork
from talent_flow.core.registry import POOLER_REGISTRY
from .base import BasePooler
from .assignment import build_hard_assignment


@POOLER_REGISTRY.register("semantic")
class SemanticPooler(BasePooler):
    """Pool nodes by a categorical attribute (e.g. industry or geography).

    Config:
        attribute_map: ``{node_id: category}``. Required.
        fallback: category label for nodes missing from the map (default
            ``"other"``).
    """

    name = "semantic"

    def __init__(
        self,
        attribute_map: Dict[Any, Any],
        fallback: Any = "other",
        **kwargs,
    ):
        super().__init__(
            attribute_map=None, fallback=fallback, **kwargs
        )  # do not store potentially huge dict in config snapshot
        self.attribute_map = dict(attribute_map)
        self.fallback = fallback

    def build_assignment(self, networks) -> AssignmentMatrix:
        agg = self.aggregated_network(networks)
        node_ids = sorted(agg.get_nodes(), key=lambda x: (isinstance(x, str), x))
        node_to_cluster: Dict[Any, Any] = {}
        clusters: list = []
        for node in node_ids:
            cat = self.attribute_map.get(node, self.fallback)
            if cat not in clusters:
                clusters.append(cat)
            node_to_cluster[node] = cat
        return build_hard_assignment(node_to_cluster, node_ids, clusters)
