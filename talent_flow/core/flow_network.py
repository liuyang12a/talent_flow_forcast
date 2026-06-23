#!/usr/bin/env python3
"""
Flow Network Module for Company-to-Company Job Transition Networks.

Sparse graph representation for talent flow networks between companies,
where nodes represent companies and edges represent employee transitions.

Migrated from the top-level ``flow_network.py`` module unchanged in behavior.
"""

from typing import Dict, List, Set, Tuple, Optional, Union, Iterator
import copy
import sys


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
        # Sparse storage: {(from_company, to_company): weight}
        self._edges: Dict[Tuple[Union[int, str], Union[int, str]], int] = {}
        self._nodes: Set[Union[int, str]] = set()

    @classmethod
    def empty(cls) -> "FlowNetwork":
        """Create and return an empty network."""
        return cls()

    def add_node(self, company_id: Union[int, str]) -> None:
        self._nodes.add(company_id)

    def add_edge(
        self,
        from_company: Union[int, str],
        to_company: Union[int, str],
        weight: int = 1,
    ) -> None:
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
        delta: int = 1,
    ) -> None:
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
        weight: int,
    ) -> None:
        if weight <= 0:
            raise ValueError("Weight must be a positive integer")
        edge_key = (from_company, to_company)
        self._edges[edge_key] = weight
        self._nodes.add(from_company)
        self._nodes.add(to_company)

    def remove_edge(
        self, from_company: Union[int, str], to_company: Union[int, str]
    ) -> bool:
        edge_key = (from_company, to_company)
        if edge_key in self._edges:
            del self._edges[edge_key]
            return True
        return False

    def remove_node(self, company_id: Union[int, str]) -> bool:
        if company_id not in self._nodes:
            return False
        edges_to_remove = [
            key
            for key in self._edges.keys()
            if key[0] == company_id or key[1] == company_id
        ]
        for key in edges_to_remove:
            del self._edges[key]
        self._nodes.remove(company_id)
        return True

    def get_edge_weight(
        self, from_company: Union[int, str], to_company: Union[int, str]
    ) -> int:
        return self._edges.get((from_company, to_company), 0)

    def has_edge(
        self, from_company: Union[int, str], to_company: Union[int, str]
    ) -> bool:
        return (from_company, to_company) in self._edges

    def has_node(self, company_id: Union[int, str]) -> bool:
        return company_id in self._nodes

    def get_nodes(self) -> Set[Union[int, str]]:
        return self._nodes.copy()

    def get_edges(
        self,
    ) -> Dict[Tuple[Union[int, str], Union[int, str]], int]:
        return self._edges.copy()

    def get_outgoing_edges(
        self, company_id: Union[int, str]
    ) -> Dict[Union[int, str], int]:
        return {
            target: weight
            for (source, target), weight in self._edges.items()
            if source == company_id
        }

    def get_incoming_edges(
        self, company_id: Union[int, str]
    ) -> Dict[Union[int, str], int]:
        return {
            source: weight
            for (source, target), weight in self._edges.items()
            if target == company_id
        }

    def __add__(self, other: "FlowNetwork") -> "FlowNetwork":
        result = FlowNetwork()
        for (source, target), weight in self._edges.items():
            result._edges[(source, target)] = weight
        for (source, target), weight in other._edges.items():
            result._edges[(source, target)] = (
                result._edges.get((source, target), 0) + weight
            )
        result._nodes = self._nodes.union(other._nodes)
        return result

    def __iadd__(self, other: "FlowNetwork") -> "FlowNetwork":
        for (source, target), weight in other._edges.items():
            self._edges[(source, target)] = (
                self._edges.get((source, target), 0) + weight
            )
        self._nodes.update(other._nodes)
        return self

    def intersection(self, other: "FlowNetwork") -> "FlowNetwork":
        result = FlowNetwork()
        common_edges = set(self._edges.keys()) & set(other._edges.keys())
        for edge_key in common_edges:
            weight = min(self._edges[edge_key], other._edges[edge_key])
            result._edges[edge_key] = weight
            result._nodes.add(edge_key[0])
            result._nodes.add(edge_key[1])
        return result

    def __and__(self, other: "FlowNetwork") -> "FlowNetwork":
        return self.intersection(other)

    def to_adjacency_matrix(
        self,
        node_order: Optional[List[Union[int, str]]] = None,
    ) -> Tuple[List[List[int]], List[Union[int, str]]]:
        """Convert the network to an adjacency matrix.

        Args:
            node_order: Optional list specifying the order of nodes in the matrix.
                If not provided, nodes are sorted alphabetically/numerically.

        Returns:
            Tuple of (adjacency_matrix, node_list) where adjacency_matrix is a 2D
            list of integers and node_list is the ordered company IDs.
        """
        if node_order is None:
            node_order = sorted(self._nodes, key=lambda x: (isinstance(x, str), x))

        for node in node_order:
            if node not in self._nodes:
                raise ValueError(
                    f"Node {node} in node_order does not exist in the network"
                )

        node_to_idx = {node: idx for idx, node in enumerate(node_order)}
        n = len(node_order)
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        for (source, target), weight in self._edges.items():
            if source in node_to_idx and target in node_to_idx:
                i, j = node_to_idx[source], node_to_idx[target]
                matrix[i][j] = weight
        return matrix, node_order

    def get_out_degree(self, company_id: Union[int, str]) -> int:
        return sum(
            weight
            for (source, _), weight in self._edges.items()
            if source == company_id
        )

    def get_in_degree(self, company_id: Union[int, str]) -> int:
        return sum(
            weight
            for (_, target), weight in self._edges.items()
            if target == company_id
        )

    def get_total_flow(self) -> int:
        return sum(self._edges.values())

    def get_edge_count(self) -> int:
        return len(self._edges)

    def get_node_count(self) -> int:
        return len(self._nodes)

    def copy(self) -> "FlowNetwork":
        result = FlowNetwork()
        result._edges = copy.deepcopy(self._edges)
        result._nodes = self._nodes.copy()
        return result

    def clear(self) -> None:
        self._edges.clear()
        self._nodes.clear()

    def is_empty(self) -> bool:
        return len(self._nodes) == 0

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, item: Union[int, str, Tuple]) -> bool:
        if isinstance(item, tuple) and len(item) == 2:
            return item in self._edges
        return item in self._nodes

    def __repr__(self) -> str:
        return (
            f"FlowNetwork(nodes={len(self._nodes)}, "
            f"edges={len(self._edges)}, "
            f"total_flow={self.get_total_flow()})"
        )

    def iter_edges(
        self,
    ) -> Iterator[Tuple[Union[int, str], Union[int, str], int]]:
        for (source, target), weight in self._edges.items():
            yield (source, target, weight)

    def get_density(self) -> float:
        n = len(self._nodes)
        if n <= 1:
            return 0.0
        possible_edges = n * (n - 1)  # Directed graph
        return len(self._edges) / possible_edges if possible_edges > 0 else 0.0


def merge_networks(networks: List[FlowNetwork]) -> FlowNetwork:
    """Merge multiple networks into one (weights summed)."""
    if not networks:
        return FlowNetwork.empty()
    result = networks[0].copy()
    for network in networks[1:]:
        result += network
    return result


# Backwards-compatibility: legacy pickled ``FlowNetwork`` objects were stored
# under the top-level module path ``flow_network.FlowNetwork``. Register that
# module path as an alias of this module so that ``pickle.load`` resolves the
# legacy class to the migrated class (single definition).
sys.modules.setdefault("flow_network", sys.modules[__name__])
