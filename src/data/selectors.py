"""
Selectors for choosing representative time series from flow networks.

This module provides various strategies for selecting typical time series:
1. High-weight edges: edges with largest total flow
2. Hub nodes: edges surrounding high-degree nodes
3. Community edges: edges within well-defined communities
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple, Optional, Union
from collections import defaultdict
import numpy as np
import logging

from flow_network import FlowNetwork

logger = logging.getLogger(__name__)


class BaseSelector(ABC):
    """Abstract base class for time series selectors."""

    @abstractmethod
    def select(
        self,
        networks: Dict[str, "FlowNetwork"]
    ) -> List[Tuple[Union[int, str], Union[int, str]]]:
        """
        Select edges from the flow networks.

        Args:
            networks: Dictionary mapping timestamp to FlowNetwork

        Returns:
            List of selected (source, target) edge tuples
        """
        pass


class HighWeightSelector(BaseSelector):
    """
    Select edges with highest total weight across all time periods.

    This selector identifies the most significant flows in terms of
    total number of employee transitions.
    """

    def __init__(
        self,
        top_k: int = 100,
        min_months: int = 6,
        exclude_self_loops: bool = True
    ):
        """
        Initialize high weight selector.

        Args:
            top_k: Number of top edges to select
            min_months: Minimum number of months edge must appear
            exclude_self_loops: Whether to exclude self-loop edges
        """
        self.top_k = top_k
        self.min_months = min_months
        self.exclude_self_loops = exclude_self_loops

    def select(
        self,
        networks: Dict[str, "FlowNetwork"]
    ) -> List[Tuple[Union[int, str], Union[int, str]]]:
        """
        Select top-k edges by total weight.

        Args:
            networks: Dictionary mapping timestamp to FlowNetwork

        Returns:
            List of selected (source, target) tuples
        """
        # Aggregate edge weights across all time periods
        edge_weights = defaultdict(lambda: {'total': 0, 'months': 0})

        for timestamp, network in networks.items():
            for edge, weight in network.get_edges().items():
                source, target = edge

                # Skip self-loops if configured
                if self.exclude_self_loops and source == target:
                    continue

                edge_weights[edge]['total'] += weight
                edge_weights[edge]['months'] += 1

        # Filter by minimum months and sort by total weight
        qualified_edges = [
            (edge, data['total'])
            for edge, data in edge_weights.items()
            if data['months'] >= self.min_months
        ]

        # Sort by weight descending and select top-k
        qualified_edges.sort(key=lambda x: x[1], reverse=True)
        selected = [edge for edge, _ in qualified_edges[:self.top_k]]

        logger.info(
            f"HighWeightSelector: Selected {len(selected)} edges from "
            f"{len(edge_weights)} total edges (min_months={self.min_months})"
        )

        return selected


class HubNodeSelector(BaseSelector):
    """
    Select edges surrounding high-degree (hub) nodes.

    This selector identifies nodes with high connectivity and selects
    edges connected to them, capturing star-like network structures.
    """

    def __init__(
        self,
        hub_threshold: Union[int, float] = 10,
        max_edges_per_hub: int = 20,
        degree_type: str = "total",
        selection_mode: str = "both"
    ):
        """
        Initialize hub node selector.

        Args:
            hub_threshold: If int, minimum degree to be considered a hub.
                          If float in (0, 1), fraction of nodes to select
                          from the top (e.g. 0.2 = top 20%, 1e-3 = top 0.1%).
            max_edges_per_hub: Maximum edges to select per hub node
            degree_type: Type of degree ('in', 'out', or 'total')
            selection_mode: Which edges to select ('in', 'out', or 'both')
        """
        self.hub_threshold = hub_threshold
        self.max_edges_per_hub = max_edges_per_hub
        self.degree_type = degree_type
        self.selection_mode = selection_mode

    def _compute_node_degrees(
        self,
        networks: Dict[str, "FlowNetwork"]
    ) -> Dict[Union[int, str], Dict[str, int]]:
        """
        Compute in, out, and total degrees for all nodes.

        Iterates edges directly (O(T * E)) instead of calling per-node
        get_in_degree/get_out_degree (which would be O(T * N * E)).
        """
        degrees = defaultdict(lambda: {'in': 0, 'out': 0, 'total': 0})

        for network in networks.values():
            for (source, target), weight in network.get_edges().items():
                # Source contributes out-degree
                degrees[source]['out'] += weight
                degrees[source]['total'] += weight
                # Target contributes in-degree
                degrees[target]['in'] += weight
                degrees[target]['total'] += weight

        return degrees

    def select(
        self,
        networks: Dict[str, "FlowNetwork"]
    ) -> List[Tuple[Union[int, str], Union[int, str]]]:
        """
        Select edges connected to hub nodes.

        Args:
            networks: Dictionary mapping timestamp to FlowNetwork

        Returns:
            List of selected (source, target) tuples
        """
        # Compute degrees
        degrees = self._compute_node_degrees(networks)

        # Determine hub threshold
        if isinstance(self.hub_threshold, float) and 0 < self.hub_threshold < 1:
            # hub_threshold=0.2 → top 20% → threshold at 80th percentile
            # hub_threshold=1e-3 → top 0.1% → threshold at 99.9th percentile
            all_degrees = [d[self.degree_type] for d in degrees.values()]
            threshold = np.percentile(all_degrees, (1 - self.hub_threshold) * 100)
        else:
            threshold = self.hub_threshold

        # Identify hub nodes
        hub_nodes = [
            node for node, deg in degrees.items()
            if deg[self.degree_type] >= threshold
        ]

        logger.info(f"HubNodeSelector: Found {len(hub_nodes)} hub nodes (threshold={threshold})")

        # Collect edges connected to hubs
        selected_edges = set()

        for network in networks.values():
            for edge in network.get_edges().keys():
                source, target = edge

                if source in hub_nodes or target in hub_nodes:
                    # Check selection mode
                    if self.selection_mode == 'in' and target not in hub_nodes:
                        continue
                    if self.selection_mode == 'out' and source not in hub_nodes:
                        continue

                    selected_edges.add(edge)

        # Limit edges per hub
        if self.max_edges_per_hub:
            edge_count_per_hub = defaultdict(int)
            filtered_edges = []

            for edge in selected_edges:
                source, target = edge
                hub = source if source in hub_nodes else target

                if edge_count_per_hub[hub] < self.max_edges_per_hub:
                    filtered_edges.append(edge)
                    edge_count_per_hub[hub] += 1

            selected_edges = filtered_edges

        logger.info(f"HubNodeSelector: Selected {len(selected_edges)} edges")

        return list(selected_edges)


class CommunitySelector(BaseSelector):
    """
    Select edges within well-defined communities.

    This selector uses community detection algorithms (Louvain) to
    identify densely connected communities and selects edges within them.
    """

    def __init__(
        self,
        resolution: float = 1.0,
        min_community_size: int = 5,
        max_communities: int = 5,
        edge_selection: str = "internal"
    ):
        """
        Initialize community selector.

        Args:
            resolution: Resolution parameter for Louvain algorithm.
                       Higher values = more communities.
            min_community_size: Minimum number of nodes in a community
            max_communities: Maximum number of communities to consider
            edge_selection: Which edges to select ('internal', 'external', 'both')
        """
        self.resolution = resolution
        self.min_community_size = min_community_size
        self.max_communities = max_communities
        self.edge_selection = edge_selection

    def _build_aggregate_network(
        self,
        networks: Dict[str, "FlowNetwork"]
    ) -> "FlowNetwork":
        """Build aggregate network by summing all monthly networks."""
        from flow_network import FlowNetwork

        aggregate = FlowNetwork()
        for network in networks.values():
            aggregate += network

        return aggregate

    def _detect_communities(
        self,
        network: "FlowNetwork"
    ) -> Dict[Union[int, str], int]:
        """
        Detect communities using Louvain algorithm.

        Returns:
            Dictionary mapping node to community ID
        """
        try:
            import networkx as nx
            import community as community_louvain
        except ImportError:
            logger.error("networkx and python-louvain required for community detection")
            raise

        # Convert to NetworkX graph
        G = nx.DiGraph()

        for node in network.get_nodes():
            G.add_node(node)

        for (source, target), weight in network.get_edges().items():
            G.add_edge(source, target, weight=weight)

        # Convert to undirected for community detection
        G_undirected = G.to_undirected()

        # Run Louvain algorithm
        partition = community_louvain.best_partition(
            G_undirected,
            resolution=self.resolution,
            weight='weight'
        )

        return partition

    def select(
        self,
        networks: Dict[str, "FlowNetwork"]
    ) -> List[Tuple[Union[int, str], Union[int, str]]]:
        """
        Select edges based on community structure.

        Args:
            networks: Dictionary mapping timestamp to FlowNetwork

        Returns:
            List of selected (source, target) tuples
        """
        # Build aggregate network
        aggregate = self._build_aggregate_network(networks)

        # Detect communities
        partition = self._detect_communities(aggregate)

        # Group nodes by community
        communities = defaultdict(list)
        for node, comm_id in partition.items():
            communities[comm_id].append(node)

        # Filter communities by size
        valid_communities = {
            comm_id: nodes
            for comm_id, nodes in communities.items()
            if len(nodes) >= self.min_community_size
        }

        # Sort by size and select top communities
        sorted_comms = sorted(
            valid_communities.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:self.max_communities]

        logger.info(
            f"CommunitySelector: Found {len(communities)} communities, "
            f"selected {len(sorted_comms)} (size >= {self.min_community_size})"
        )

        # Collect edges based on selection mode
        selected_edges = set()
        community_nodes = set()

        for comm_id, nodes in sorted_comms:
            community_nodes.update(nodes)

        for network in networks.values():
            for (source, target), weight in network.get_edges().items():
                source_comm = partition.get(source)
                target_comm = partition.get(target)

                if self.edge_selection == "internal":
                    # Select edges within communities
                    if source_comm == target_comm and source_comm in dict(sorted_comms):
                        selected_edges.add((source, target))

                elif self.edge_selection == "external":
                    # Select edges between different communities
                    if source_comm != target_comm:
                        selected_edges.add((source, target))

                elif self.edge_selection == "both":
                    # Select any edge involving community nodes
                    if source in community_nodes or target in community_nodes:
                        selected_edges.add((source, target))

        logger.info(f"CommunitySelector: Selected {len(selected_edges)} edges")

        return list(selected_edges)


class CompositeSelector(BaseSelector):
    """
    Combine multiple selection strategies.

    Allows using multiple selectors together with different weights.
    """

    def __init__(
        self,
        selectors: List[Tuple[BaseSelector, float]],
        max_total_edges: int = 200
    ):
        """
        Initialize composite selector.

        Args:
            selectors: List of (selector, weight) tuples
            max_total_edges: Maximum total edges to select
        """
        self.selectors = selectors
        self.max_total_edges = max_total_edges

    def select(
        self,
        networks: Dict[str, "FlowNetwork"]
    ) -> List[Tuple[Union[int, str], Union[int, str]]]:
        """
        Combine selections from multiple selectors.

        Uses weighted voting to prioritize edges selected by multiple selectors.
        """
        edge_scores = defaultdict(float)

        for selector, weight in self.selectors:
            selected = selector.select(networks)
            for edge in selected:
                edge_scores[edge] += weight

        # Sort by score and select top edges
        sorted_edges = sorted(
            edge_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        selected = [edge for edge, _ in sorted_edges[:self.max_total_edges]]

        logger.info(f"CompositeSelector: Selected {len(selected)} edges from {len(edge_scores)} candidates")

        return selected
