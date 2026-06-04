#!/usr/bin/env python3
"""
Flow Network Module for Company-to-Company Job Transition Networks.

This module provides a sparse graph representation for storing and manipulating
talent flow networks between companies, where nodes represent companies and
edges represent the intensity of employee transitions.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Union, Iterator
import copy


class FlowNetwork:
    """
    Sparse representation of a company-to-company talent flow network.

    The network is stored as a dictionary mapping (source, target) company pairs
to integer weights representing the number of employee transitions.

    Attributes:
        _edges: Dictionary mapping (from_company, to_company) tuples to integer weights
        _nodes: Set of all company IDs in the network
    """

    def __init__(self):
        """Initialize an empty flow network."""
        # Sparse storage: {(from_company, to_company): weight}
        self._edges: Dict[Tuple[Union[int, str], Union[int, str]], int] = {}
        self._nodes: Set[Union[int, str]] = set()

    @classmethod
    def empty(cls) -> "FlowNetwork":
        """
        Create and return an empty network.

        Returns:
            A new empty FlowNetwork instance.
        """
        return cls()

    def add_node(self, company_id: Union[int, str]) -> None:
        """
        Add a node (company) to the network.

        Args:
            company_id: Unique identifier for the company
        """
        self._nodes.add(company_id)

    def add_edge(
        self,
        from_company: Union[int, str],
        to_company: Union[int, str],
        weight: int = 1
    ) -> None:
        """
        Add or update an edge between two companies.

        Args:
            from_company: Source company ID
            to_company: Target company ID
            weight: Weight to add to the edge (default: 1)
        """
        if weight <= 0:
            raise ValueError("Weight must be a positive integer")

        edge_key = (from_company, to_company)
        self._edges[edge_key] = self._edges.get(edge_key, 0) + weight
        self._nodes.add(from_company)
        self._nodes.add(to_company)

    def increment_edge(
        self,
        from_company: Union[int, str],
        to_company: Union[int, str],
        delta: int = 1
    ) -> None:
        """
        Increment the weight of an existing edge.

        Args:
            from_company: Source company ID
            to_company: Target company ID
            delta: Amount to add to the edge weight (default: 1)
        """
        edge_key = (from_company, to_company)
        self._edges[edge_key] = self._edges.get(edge_key, 0) + delta
        if self._edges[edge_key] <= 0:
            del self._edges[edge_key]
        else:
            self._nodes.add(from_company)
            self._nodes.add(to_company)

    def set_edge_weight(
        self,
        from_company: Union[int, str],
        to_company: Union[int, str],
        weight: int
    ) -> None:
        """
        Set the weight of an edge directly.

        Args:
            from_company: Source company ID
            to_company: Target company ID
            weight: New weight value (must be positive)
        """
        if weight <= 0:
            raise ValueError("Weight must be a positive integer")

        edge_key = (from_company, to_company)
        self._edges[edge_key] = weight
        self._nodes.add(from_company)
        self._nodes.add(to_company)

    def remove_edge(
        self,
        from_company: Union[int, str],
        to_company: Union[int, str]
    ) -> bool:
        """
        Remove an edge from the network.

        Args:
            from_company: Source company ID
            to_company: Target company ID

        Returns:
            True if the edge was removed, False if it didn't exist
        """
        edge_key = (from_company, to_company)
        if edge_key in self._edges:
            del self._edges[edge_key]
            return True
        return False

    def remove_node(self, company_id: Union[int, str]) -> bool:
        """
        Remove a node and all its connected edges from the network.

        Args:
            company_id: Company ID to remove

        Returns:
            True if the node was removed, False if it didn't exist
        """
        if company_id not in self._nodes:
            return False

        # Remove all edges connected to this node
        edges_to_remove = [
            key for key in self._edges.keys()
            if key[0] == company_id or key[1] == company_id
        ]
        for key in edges_to_remove:
            del self._edges[key]

        self._nodes.remove(company_id)
        return True

    def get_edge_weight(
        self,
        from_company: Union[int, str],
        to_company: Union[int, str]
    ) -> int:
        """
        Get the weight of a specific edge.

        Args:
            from_company: Source company ID
            to_company: Target company ID

        Returns:
            Edge weight (0 if edge doesn't exist)
        """
        return self._edges.get((from_company, to_company), 0)

    def has_edge(
        self,
        from_company: Union[int, str],
        to_company: Union[int, str]
    ) -> bool:
        """
        Check if an edge exists.

        Args:
            from_company: Source company ID
            to_company: Target company ID

        Returns:
            True if the edge exists, False otherwise
        """
        return (from_company, to_company) in self._edges

    def has_node(self, company_id: Union[int, str]) -> bool:
        """
        Check if a node exists.

        Args:
            company_id: Company ID to check

        Returns:
            True if the node exists, False otherwise
        """
        return company_id in self._nodes

    def get_nodes(self) -> Set[Union[int, str]]:
        """
        Get all nodes in the network.

        Returns:
            Set of company IDs
        """
        return self._nodes.copy()

    def get_edges(
        self
    ) -> Dict[Tuple[Union[int, str], Union[int, str]], int]:
        """
        Get all edges in the network.

        Returns:
            Dictionary mapping (from, to) tuples to weights
        """
        return self._edges.copy()

    def get_outgoing_edges(
        self,
        company_id: Union[int, str]
    ) -> Dict[Union[int, str], int]:
        """
        Get all outgoing edges from a company.

        Args:
            company_id: Source company ID

        Returns:
            Dictionary mapping target companies to weights
        """
        return {
            target: weight
            for (source, target), weight in self._edges.items()
            if source == company_id
        }

    def get_incoming_edges(
        self,
        company_id: Union[int, str]
    ) -> Dict[Union[int, str], int]:
        """
        Get all incoming edges to a company.

        Args:
            company_id: Target company ID

        Returns:
            Dictionary mapping source companies to weights
        """
        return {
            source: weight
            for (source, target), weight in self._edges.items()
            if target == company_id
        }

    def __add__(self, other: "FlowNetwork") -> "FlowNetwork":
        """
        Add two networks together (union with weight addition).

        For overlapping edges, weights are summed. Nodes from both networks
        are included in the result.

        Args:
            other: Another FlowNetwork to add to this one

        Returns:
            New FlowNetwork containing the sum of both networks
        """
        result = FlowNetwork()

        # Copy all edges from self
        for (source, target), weight in self._edges.items():
            result._edges[(source, target)] = weight

        # Add edges from other (summing weights for overlapping edges)
        for (source, target), weight in other._edges.items():
            result._edges[(source, target)] = (
                result._edges.get((source, target), 0) + weight
            )

        # Union of nodes
        result._nodes = self._nodes.union(other._nodes)

        return result

    def __iadd__(self, other: "FlowNetwork") -> "FlowNetwork":
        """
        In-place addition (merge another network into this one).

        Args:
            other: Another FlowNetwork to merge into this one

        Returns:
            self
        """
        for (source, target), weight in other._edges.items():
            self._edges[(source, target)] = (
                self._edges.get((source, target), 0) + weight
            )

        self._nodes.update(other._nodes)
        return self

    def intersection(self, other: "FlowNetwork") -> "FlowNetwork":
        """
        Compute the intersection of two networks.

        Returns a new network containing only edges that exist in both
        input networks, with the minimum of the two weights.

        Args:
            other: Another FlowNetwork to intersect with

        Returns:
            New FlowNetwork containing the intersection
        """
        result = FlowNetwork()

        # Find common edges and take minimum weight
        common_edges = set(self._edges.keys()) & set(other._edges.keys())

        for edge_key in common_edges:
            weight = min(self._edges[edge_key], other._edges[edge_key])
            result._edges[edge_key] = weight
            result._nodes.add(edge_key[0])
            result._nodes.add(edge_key[1])

        return result

    def __and__(self, other: "FlowNetwork") -> "FlowNetwork":
        """
        Operator overload for intersection (&).

        Args:
            other: Another FlowNetwork to intersect with

        Returns:
            New FlowNetwork containing the intersection
        """
        return self.intersection(other)

    def to_adjacency_matrix(
        self,
        node_order: Optional[List[Union[int, str]]] = None
    ) -> Tuple[List[List[int]], List[Union[int, str]]]:
        """
        Convert the network to an adjacency matrix representation.

        Args:
            node_order: Optional list specifying the order of nodes in the matrix.
                       If not provided, nodes are sorted alphabetically/numerically.

        Returns:
            Tuple of (adjacency_matrix, node_list) where:
                - adjacency_matrix is a 2D list of integers
                - node_list is the ordered list of company IDs corresponding to matrix indices
        """
        if node_order is None:
            node_order = sorted(self._nodes, key=lambda x: (isinstance(x, str), x))

        # Validate that all nodes in node_order exist in the network
        for node in node_order:
            if node not in self._nodes:
                raise ValueError(f"Node {node} in node_order does not exist in the network")

        # Create node to index mapping
        node_to_idx = {node: idx for idx, node in enumerate(node_order)}
        n = len(node_order)

        # Initialize matrix with zeros
        matrix = [[0 for _ in range(n)] for _ in range(n)]

        # Fill in edge weights
        for (source, target), weight in self._edges.items():
            if source in node_to_idx and target in node_to_idx:
                i, j = node_to_idx[source], node_to_idx[target]
                matrix[i][j] = weight

        return matrix, node_order

    def get_out_degree(self, company_id: Union[int, str]) -> int:
        """
        Get the total outgoing flow from a company.

        Args:
            company_id: Company ID

        Returns:
            Sum of all outgoing edge weights
        """
        return sum(
            weight
            for (source, _), weight in self._edges.items()
            if source == company_id
        )

    def get_in_degree(self, company_id: Union[int, str]) -> int:
        """
        Get the total incoming flow to a company.

        Args:
            company_id: Company ID

        Returns:
            Sum of all incoming edge weights
        """
        return sum(
            weight
            for (_, target), weight in self._edges.items()
            if target == company_id
        )

    def get_total_flow(self) -> int:
        """
        Get the total flow (sum of all edge weights) in the network.

        Returns:
            Total sum of all edge weights
        """
        return sum(self._edges.values())

    def get_edge_count(self) -> int:
        """
        Get the number of edges in the network.

        Returns:
            Number of edges
        """
        return len(self._edges)

    def get_node_count(self) -> int:
        """
        Get the number of nodes in the network.

        Returns:
            Number of nodes
        """
        return len(self._nodes)

    def copy(self) -> "FlowNetwork":
        """
        Create a deep copy of the network.

        Returns:
            New FlowNetwork that is a copy of this one
        """
        result = FlowNetwork()
        result._edges = copy.deepcopy(self._edges)
        result._nodes = self._nodes.copy()
        return result

    def clear(self) -> None:
        """Clear all nodes and edges from the network."""
        self._edges.clear()
        self._nodes.clear()

    def is_empty(self) -> bool:
        """
        Check if the network is empty.

        Returns:
            True if the network has no nodes, False otherwise
        """
        return len(self._nodes) == 0

    def __len__(self) -> int:
        """Return the number of nodes in the network."""
        return len(self._nodes)

    def __contains__(self, item: Union[int, str, Tuple]) -> bool:
        """
        Check if a node or edge exists in the network.

        Args:
            item: Company ID (node) or (source, target) tuple (edge)

        Returns:
            True if the item exists, False otherwise
        """
        if isinstance(item, tuple) and len(item) == 2:
            return item in self._edges
        return item in self._nodes

    def __repr__(self) -> str:
        """Return a string representation of the network."""
        return (
            f"FlowNetwork(nodes={len(self._nodes)}, "
            f"edges={len(self._edges)}, "
            f"total_flow={self.get_total_flow()})"
        )

    def iter_edges(
        self
    ) -> Iterator[Tuple[Union[int, str], Union[int, str], int]]:
        """
        Iterate over all edges in the network.

        Yields:
            Tuples of (source, target, weight)
        """
        for (source, target), weight in self._edges.items():
            yield (source, target, weight)

    def get_density(self) -> float:
        """
        Calculate the network density.

        Density is the ratio of actual edges to possible edges.

        Returns:
            Network density (0.0 to 1.0)
        """
        n = len(self._nodes)
        if n <= 1:
            return 0.0
        possible_edges = n * (n - 1)  # Directed graph
        return len(self._edges) / possible_edges if possible_edges > 0 else 0.0


def merge_networks(networks: List[FlowNetwork]) -> FlowNetwork:
    """
    Merge multiple networks into one.

    Args:
        networks: List of FlowNetwork instances to merge

    Returns:
        New FlowNetwork containing the sum of all input networks
    """
    if not networks:
        return FlowNetwork.empty()

    result = networks[0].copy()
    for network in networks[1:]:
        result += network

    return result
