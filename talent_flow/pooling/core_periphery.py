#!/usr/bin/env python3
"""Core-Periphery decomposition pooling. (论文核心创新)

Tailored to scale-free talent-flow networks:
  - A few HUB (core) companies account for most of the flow and are highly
    interconnected. They are kept as *individual* atomic super-nodes so their
    high-resolution dynamics are preserved.
  - The long tail (periphery) is aggregated into a small number of
    "edge super-nodes" by a categorical attribute (industry/geography),
    capturing their macroscopic supply/drain role toward the hubs.

This yields a K = n_core + n_edge super-node OD matrix that is both dense and
interpretable, distinguishing "hub vs hub" talent battles from "hub vs
periphery" drain/spillover.

Research gap: no prior work combines core-periphery decomposition with
differentiable pooling. This module provides the static (non-differentiable)
form; a differentiable variant is a future direction.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from talent_flow.core import AssignmentMatrix, FlowNetwork
from talent_flow.core.registry import POOLER_REGISTRY
from .base import BasePooler
from .assignment import build_hard_assignment


@POOLER_REGISTRY.register("core_periphery")
class CorePeripheryPooler(BasePooler):
    """Pool by core-periphery decomposition.

    Config:
        n_core: number of top hub nodes to keep as atomic super-nodes.
        core_method: ``"degree"`` (top-N by total flow) or ``"k_core"``
            (nodes with k-core number >= ``k_core_threshold``).
        k_core_threshold: threshold for the ``"k_core"`` method.
        periphery_attribute_map: ``{node_id: category}`` for aggregating the
            periphery. If ``None``, periphery nodes are all merged into a
            single ``"periphery_other"`` super-node.
        periphery_fallback: category for periphery nodes missing from the map.
    """

    name = "core_periphery"

    def __init__(
        self,
        n_core: int = 50,
        core_method: str = "degree",
        k_core_threshold: int = 5,
        periphery_attribute_map: Optional[Dict[Any, Any]] = None,
        periphery_fallback: Any = "other",
        **kwargs,
    ):
        super().__init__(
            n_core=n_core,
            core_method=core_method,
            k_core_threshold=k_core_threshold,
            periphery_fallback=periphery_fallback,
            **kwargs,
        )
        self.n_core = n_core
        self.core_method = core_method
        self.k_core_threshold = k_core_threshold
        self.periphery_attribute_map = (
            dict(periphery_attribute_map) if periphery_attribute_map else None
        )
        self.periphery_fallback = periphery_fallback

    def _identify_core(self, networks) -> list:
        if self.core_method == "degree":
            scores = self.node_degree_scores(networks)
            ranked = sorted(scores.keys(), key=lambda n: scores[n], reverse=True)
            return ranked[: self.n_core]
        elif self.core_method == "k_core":
            import networkx as nx

            agg = self.aggregated_network(networks)
            G = nx.DiGraph()
            for src, tgt, w in agg.iter_edges():
                G.add_edge(src, tgt, weight=float(w))
            core_number = nx.core_number(G.to_undirected())
            core = [
                n for n, k in core_number.items() if k >= self.k_core_threshold
            ]
            if len(core) > self.n_core:
                scores = self.node_degree_scores(networks)
                core = sorted(core, key=lambda n: scores.get(n, 0.0), reverse=True)[
                    : self.n_core
                ]
            return core
        else:
            raise ValueError(f"unknown core_method: {self.core_method}")

    def build_assignment(self, networks) -> AssignmentMatrix:
        core_nodes = self._identify_core(networks)
        core_set = set(core_nodes)
        # all nodes seen across the time range
        all_nodes_set: set = set()
        for net in networks.values():
            all_nodes_set.update(net.get_nodes())
        all_nodes = sorted(all_nodes_set, key=lambda x: (isinstance(x, str), x))
        periphery = [n for n in all_nodes if n not in core_set]

        # super-node ids: core atoms first, then periphery clusters
        supernode_ids: list = list(core_nodes)

        node_to_cluster: Dict[Any, Any] = {n: n for n in core_nodes}

        if self.periphery_attribute_map is None:
            # single catch-all periphery super-node
            periphery_label = ("periphery", "other")
            supernode_ids.append(periphery_label)
            for n in periphery:
                node_to_cluster[n] = periphery_label
        else:
            # aggregate periphery by attribute
            periphery_cats: list = []
            for n in periphery:
                cat = self.periphery_attribute_map.get(
                    n, self.periphery_fallback
                )
                if cat not in periphery_cats:
                    periphery_cats.append(cat)
                node_to_cluster[n] = ("periphery", cat)
            for cat in periphery_cats:
                supernode_ids.append(("periphery", cat))

        return build_hard_assignment(node_to_cluster, all_nodes, supernode_ids)

    def get_core_mask(self, assignment: AssignmentMatrix) -> np.ndarray:
        """Boolean ``[K]`` mask: True for core (atomic) super-nodes.

        Useful for the forecasting evaluator's core/periphery split.
        """
        K = len(assignment.supernode_ids)
        mask = np.zeros(K, dtype=bool)
        for j, sid in enumerate(assignment.supernode_ids):
            if not (isinstance(sid, tuple) and sid[0] == "periphery"):
                mask[j] = True
        return mask
