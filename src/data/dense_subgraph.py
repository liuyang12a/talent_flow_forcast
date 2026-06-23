"""
Dense Spatio-Temporal Subgraph Extraction.

This module provides algorithms for extracting temporally and spatially dense
subgraphs from a sequence of monthly FlowNetwork instances. The goal is to
discard sparse regions (low-degree nodes, intermittently-active edges) while
minimising structural information loss.

Architecture:
    DenseSubgraphExtractor  ── pipeline A→B→C (extraction)
            │
            │  delegates to
            ▼
    BaseTensorBuilder  ◄── EdgeCentricTensorBuilder   (default, [T,E,1])
                       ◄── NodeCentricTensorBuilder   ([T,N,C])

The extraction algorithm is independent of the output tensor format.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Union

import numpy as np

from flow_network import FlowNetwork

logger = logging.getLogger(__name__)

# ── type aliases ────────────────────────────────────────────────────────────

NodeId = Union[int, str]
Edge = Tuple[NodeId, NodeId]


# ── configuration ───────────────────────────────────────────────────────────

@dataclass
class DenseSubgraphConfig:
    """Configuration for the dense-subgraph extraction pipeline.

    Attributes
    ----------
    spatial_strategy : str
        ``"flow_core"`` (default), ``"k_core"``, or ``"greedy_density"``.
    max_nodes : int
        Upper bound on the number of core company nodes.
    min_nodes : int
        Lower bound — the extractor will warn if the result is smaller.
    target_coverage : float
        Minimum fraction of total flow that the core must cover
        (only used by ``"flow_core"`` strategy).
    min_activity_ratio : float
        Minimum fraction of months an edge must have non-zero weight.
    max_allowed_gap : int
        Maximum number of consecutive zero-weight months permitted.
    min_temporal_score : float
        Composite temporal-score threshold (overridden when
        ``min_activity_ratio`` or ``max_allowed_gap`` fire first).
    tensor_type : str
        ``"edge_centric"`` (default) or ``"node_centric"``.
    adj_type : str
        For edge-centric mode: ``"shared_node"`` (default) or
        ``"temporal_correlation"``.
    node_features : List[str] or None
        For node-centric mode: which features to compute
        (e.g. ``["net_flow", "in_flow", "out_flow"]``).
    exclude_self_loops : bool
        Whether to exclude edges where source == target.
    """

    # spatial core
    spatial_strategy: str = "flow_core"
    max_nodes: int = 200
    min_nodes: int = 20
    target_coverage: float = 0.80

    # temporal density
    min_activity_ratio: float = 0.30
    max_allowed_gap: int = 12
    min_temporal_score: float = 0.25

    # tensor builder
    tensor_type: str = "edge_centric"
    adj_type: str = "shared_node"
    node_features: Optional[List[str]] = None

    # general
    exclude_self_loops: bool = True


# ── result containers ───────────────────────────────────────────────────────

@dataclass
class NodeStats:
    """Per-node statistics accumulated across all time steps."""

    flow_volume: float = 0.0          # sum of weighted in+out degree
    degree: int = 0                   # unique partners
    active_months: int = 0            # months with ≥1 incident edge
    temporal_activity: float = 0.0    # active_months / T
    node_score: float = 0.0           # flow_volume × temporal_activity


@dataclass
class EdgeStats:
    """Per-edge statistics accumulated across all time steps."""

    total_weight: float = 0.0
    active_months: int = 0
    activity_ratio: float = 0.0       # active_months / T
    max_gap: int = 0                  # longest run of zero-weight months
    temporal_score: float = 0.0       # activity_ratio × (1 - max_gap/T)


@dataclass
class QualityMetrics:
    """Post-extraction quality indicators."""

    node_count: int = 0
    edge_count: int = 0
    spatial_density: float = 0.0       # |E| / (|V|*(|V|-1))
    flow_coverage: float = 0.0         # fraction of total flow retained
    avg_degree: float = 0.0
    mean_activity: float = 0.0         # mean activity_ratio across edges
    median_activity: float = 0.0
    low_activity_pct: float = 0.0      # fraction with activity_ratio < 0.3
    mean_max_gap: float = 0.0
    total_flow_original: float = 0.0
    total_flow_retained: float = 0.0


@dataclass
class DenseSubgraphResult:
    """Complete output of the extraction pipeline."""

    nodes: Set[NodeId] = field(default_factory=set)
    edges: List[Edge] = field(default_factory=list)
    tensor: Optional[np.ndarray] = None
    adjacency: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)
    quality: Optional[QualityMetrics] = None


# ── helper: gap analysis ────────────────────────────────────────────────────

def _longest_zero_run(series: np.ndarray) -> int:
    """Return the length of the longest consecutive run of zeros."""
    if series.size == 0:
        return 0
    # boolean mask: True where value is 0
    is_zero = (series == 0).astype(int)
    # find runs … pad with 0 at both ends so diff catches boundaries
    padded = np.concatenate(([0], is_zero, [0]))
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    if len(starts) == 0:
        return 0
    return int((ends - starts).max())


# ── tensor builders ─────────────────────────────────────────────────────────

class BaseTensorBuilder(ABC):
    """Abstract interface for building (tensor, adjacency) from a dense
    subgraph.

    Subclasses implement ``build()``.  Use the factory
    ``BaseTensorBuilder.from_config()`` to instantiate the correct variant.
    """

    @abstractmethod
    def build(
        self,
        networks: Dict[str, FlowNetwork],
        nodes: Set[NodeId],
        edges: List[Edge],
        timestamps: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Build tensor and adjacency matrix.

        Parameters
        ----------
        networks : dict
            Mapping ``{timestamp_str: FlowNetwork}`` for all months.
        nodes : set
            Core company nodes selected by the extractor.
        edges : list of (src, tgt)
            Dense edges (after temporal filtering).
        timestamps : list of str
            Sorted timestamps (length *T*).

        Returns
        -------
        tensor : np.ndarray
        adjacency : np.ndarray
        metadata : dict
            Must include ``"tensor_type"``.
        """
        ...

    @staticmethod
    def from_config(config: dict) -> "BaseTensorBuilder":
        """Factory: instantiate the builder prescribed by the config dict."""
        tensor_type = config.get("tensor_type", "edge_centric")
        if tensor_type == "node_centric":
            return NodeCentricTensorBuilder(
                node_features=config.get("node_features"),
            )
        return EdgeCentricTensorBuilder(
            adj_type=config.get("adj_type", "shared_node"),
        )


class EdgeCentricTensorBuilder(BaseTensorBuilder):
    """Build ``[T, |E|, 1]`` tensor with line-graph adjacency.

    This is the **default** builder — it preserves the full "which company
    flows to which company" pairing information.
    """

    def __init__(self, adj_type: str = "shared_node"):
        self.adj_type = adj_type

    # ------------------------------------------------------------------
    def build(
        self,
        networks: Dict[str, FlowNetwork],
        nodes: Set[NodeId],
        edges: List[Edge],
        timestamps: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        T = len(timestamps)
        E = len(edges)
        tensor = np.zeros((T, E, 1), dtype=np.float32)

        for t, ts in enumerate(timestamps):
            net = networks[ts]
            for e_idx, (src, tgt) in enumerate(edges):
                tensor[t, e_idx, 0] = float(net.get_edge_weight(src, tgt))

        adjacency = self._build_adjacency(edges, timestamps, networks)

        metadata = {
            "tensor_type": "edge_centric",
            "shape": list(tensor.shape),
            "num_edges": E,
            "adj_type": self.adj_type,
        }
        return tensor, adjacency, metadata

    # ------------------------------------------------------------------
    def _build_adjacency(
        self,
        edges: List[Edge],
        timestamps: List[str],
        networks: Dict[str, FlowNetwork],
    ) -> np.ndarray:
        n = len(edges)
        if self.adj_type == "shared_node":
            return _build_line_graph_adjacency(edges)
        elif self.adj_type == "temporal_correlation":
            return _build_correlation_adjacency(edges, timestamps, networks)
        else:
            logger.warning(
                "Unknown adj_type '%s', falling back to shared_node.",
                self.adj_type,
            )
            return _build_line_graph_adjacency(edges)

    # ------------------------------------------------------------------
    @staticmethod
    def to_node_centric(
        tensor_e: np.ndarray,
        edges: List[Edge],
        nodes: Set[NodeId],
        node_features: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Derive a node-centric view from an edge-centric tensor.

        This is a **pure function** — the original edge tensor is not
        modified.  Information is aggregated, not discarded.

        Parameters
        ----------
        tensor_e : np.ndarray  shape ``[T, E, 1]``
        edges : list of (src, tgt)
        nodes : set of node ids
        node_features : list of str, optional
            Which features to compute.  Default: ``["net_flow"]``.

        Returns
        -------
        tensor_n : np.ndarray  shape ``[T, N, C]``
        adjacency_n : np.ndarray  shape ``[N, N]``
        metadata : dict
        """
        if node_features is None:
            node_features = ["net_flow"]

        node_list = sorted(nodes, key=lambda x: (isinstance(x, str), x))
        node_to_idx = {n: i for i, n in enumerate(node_list)}

        T, E, _ = tensor_e.shape
        N = len(node_list)
        C = len(node_features)

        tensor_n = np.zeros((T, N, C), dtype=np.float32)

        for t in range(T):
            for e_idx, (src, tgt) in enumerate(edges):
                w = tensor_e[t, e_idx, 0]
                if w == 0:
                    continue
                si = node_to_idx.get(src)
                ti = node_to_idx.get(tgt)
                if si is None or ti is None:
                    continue

                for c, feat in enumerate(node_features):
                    if feat == "in_flow":
                        tensor_n[t, ti, c] += w
                    elif feat == "out_flow":
                        tensor_n[t, si, c] += w
                    elif feat == "net_flow":
                        tensor_n[t, si, c] += w   # out-flow positive
                        tensor_n[t, ti, c] -= w   # in-flow negative
                    elif feat == "total_flow":
                        tensor_n[t, si, c] += w
                        tensor_n[t, ti, c] += w

        # build node-level adjacency from edge list
        adj_n = _build_node_adjacency_from_edges(edges, node_list, node_to_idx)

        metadata = {
            "tensor_type": "node_centric",
            "shape": list(tensor_n.shape),
            "num_nodes": N,
            "features": node_features,
            "derived_from": "edge_centric",
        }
        return tensor_n, adj_n, metadata


class NodeCentricTensorBuilder(BaseTensorBuilder):
    """Build ``[T, |V|, C]`` tensor with company-level adjacency.

    This is an aggregated view — each company is a node and features are
    derived from the edge weights incident to it.
    """

    def __init__(self, node_features: Optional[List[str]] = None):
        self.node_features = node_features or ["net_flow"]

    # ------------------------------------------------------------------
    def build(
        self,
        networks: Dict[str, FlowNetwork],
        nodes: Set[NodeId],
        edges: List[Edge],
        timestamps: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        node_list = sorted(nodes, key=lambda x: (isinstance(x, str), x))
        node_to_idx = {n: i for i, n in enumerate(node_list)}

        T = len(timestamps)
        N = len(node_list)
        C = len(self.node_features)

        tensor = np.zeros((T, N, C), dtype=np.float32)

        for t, ts in enumerate(timestamps):
            net = networks[ts]
            for src, tgt in edges:
                w = float(net.get_edge_weight(src, tgt))
                if w == 0:
                    continue
                si = node_to_idx.get(src)
                ti = node_to_idx.get(tgt)
                if si is None or ti is None:
                    continue

                for c, feat in enumerate(self.node_features):
                    if feat == "in_flow":
                        tensor[t, ti, c] += w
                    elif feat == "out_flow":
                        tensor[t, si, c] += w
                    elif feat == "net_flow":
                        tensor[t, si, c] += w
                        tensor[t, ti, c] -= w
                    elif feat == "total_flow":
                        tensor[t, si, c] += w
                        tensor[t, ti, c] += w

        adjacency = _build_node_adjacency_from_edges(edges, node_list, node_to_idx)

        metadata = {
            "tensor_type": "node_centric",
            "shape": list(tensor.shape),
            "num_nodes": N,
            "features": self.node_features,
        }
        return tensor, adjacency, metadata


# ── adjacency helpers ───────────────────────────────────────────────────────

def _build_line_graph_adjacency(edges: List[Edge]) -> np.ndarray:
    """Build |E|×|E| adjacency where two edge-nodes connect if they share
    at least one company endpoint."""
    n = len(edges)
    adj = np.eye(n, dtype=np.float32)
    for i in range(n):
        si, ti = edges[i]
        for j in range(i + 1, n):
            sj, tj = edges[j]
            if si == sj or ti == tj or si == tj or ti == sj:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
    return adj


def _build_node_adjacency_from_edges(
    edges: List[Edge],
    node_list: List[NodeId],
    node_to_idx: Dict[NodeId, int],
) -> np.ndarray:
    """Build |V|×|V| weighted adjacency from the retained edge set."""
    n = len(node_list)
    adj = np.zeros((n, n), dtype=np.float32)
    for src, tgt in edges:
        si = node_to_idx.get(src)
        ti = node_to_idx.get(tgt)
        if si is not None and ti is not None:
            adj[si, ti] += 1.0  # count occurrences (or could use weight)
    return adj


def _build_correlation_adjacency(
    edges: List[Edge],
    timestamps: List[str],
    networks: Dict[str, FlowNetwork],
    threshold: float = 0.5,
) -> np.ndarray:
    """Build |E|×|E| adjacency based on Pearson correlation of edge time
    series.  Two edges are connected if |r| ≥ *threshold*."""
    n = len(edges)
    T = len(timestamps)
    if T < 3 or n < 2:
        return np.eye(n, dtype=np.float32)

    # extract [T, E] weight matrix
    weights = np.zeros((T, n), dtype=np.float32)
    for t, ts in enumerate(timestamps):
        net = networks[ts]
        for e_idx, (src, tgt) in enumerate(edges):
            weights[t, e_idx] = float(net.get_edge_weight(src, tgt))

    adj = np.eye(n, dtype=np.float32)
    for i in range(n):
        si = weights[:, i]
        si_std = si.std()
        if si_std < 1e-8:
            continue
        for j in range(i + 1, n):
            sj = weights[:, j]
            sj_std = sj.std()
            if sj_std < 1e-8:
                continue
            corr = np.corrcoef(si, sj)[0, 1]
            if not np.isfinite(corr):
                continue
            if abs(corr) >= threshold:
                adj[i, j] = abs(corr)
                adj[j, i] = abs(corr)

    return adj


# ── main extractor ──────────────────────────────────────────────────────────

class DenseSubgraphExtractor:
    """Extract a spatio-temporally dense subgraph from a monthly network
    sequence.

    The pipeline runs three stages internally:

    A. **compute_statistics** — per-node & per-edge aggregates.
    B. **extract_spatial_core** — select dense core company nodes.
    C. **filter_temporal_density** — keep only temporally-dense edges.
    E. **tensor_builder.build** — output tensor + adjacency.
    D. **evaluate_quality** — compute quality report.

    Parameters
    ----------
    config : DenseSubgraphConfig
    tensor_builder : BaseTensorBuilder
    """

    def __init__(
        self,
        config: DenseSubgraphConfig,
        tensor_builder: BaseTensorBuilder,
    ):
        self.config = config
        self.tensor_builder = tensor_builder

        # intermediate caches
        self._node_stats: Dict[NodeId, NodeStats] = {}
        self._edge_stats: Dict[Edge, EdgeStats] = {}
        self._total_flow: float = 0.0

    # ── stage A ────────────────────────────────────────────────────────

    def compute_statistics(
        self,
        networks: Dict[str, FlowNetwork],
    ) -> Tuple[Dict[NodeId, NodeStats], Dict[Edge, EdgeStats]]:
        """Aggregate per-node and per-edge statistics across all months.

        Parameters
        ----------
        networks : dict  ``{timestamp: FlowNetwork}``

        Returns
        -------
        node_stats : dict  ``{node_id: NodeStats}``
        edge_stats : dict  ``{(src, tgt): EdgeStats}``
        """
        T = len(networks)
        if T == 0:
            logger.warning("Empty network sequence.")
            return {}, {}

        # accumulators
        node_flow: Dict[NodeId, float] = defaultdict(float)
        node_partners: Dict[NodeId, set] = defaultdict(set)
        node_active_months: Dict[NodeId, set] = defaultdict(set)

        edge_weight: Dict[Edge, float] = defaultdict(float)
        edge_active_months: Dict[Edge, set] = defaultdict(set)

        # per-edge monthly snapshots for gap analysis
        timestamps = sorted(networks.keys())
        edge_series: Dict[Edge, List[float]] = defaultdict(
            lambda: [0.0] * T,
        )

        self._total_flow = 0.0

        for t, ts in enumerate(timestamps):
            net = networks[ts]
            for (src, tgt), w in net.get_edges().items():
                if self.config.exclude_self_loops and src == tgt:
                    continue
                wf = float(w)

                # nodes
                node_flow[src] += wf
                node_flow[tgt] += wf
                node_partners[src].add(tgt)
                node_partners[tgt].add(src)
                node_active_months[src].add(t)
                node_active_months[tgt].add(t)

                # edges
                e = (src, tgt)
                edge_weight[e] += wf
                edge_active_months[e].add(t)
                edge_series[e][t] = wf

                self._total_flow += wf

        # build NodeStats
        node_stats: Dict[NodeId, NodeStats] = {}
        for n in set(list(node_flow.keys())):
            ns = NodeStats()
            ns.flow_volume = node_flow[n]
            ns.degree = len(node_partners.get(n, set()))
            ns.active_months = len(node_active_months.get(n, set()))
            ns.temporal_activity = ns.active_months / T if T > 0 else 0.0
            ns.node_score = ns.flow_volume * ns.temporal_activity
            node_stats[n] = ns

        # build EdgeStats
        edge_stats: Dict[Edge, EdgeStats] = {}
        for e, total_w in edge_weight.items():
            es = EdgeStats()
            es.total_weight = total_w
            es.active_months = len(edge_active_months.get(e, set()))
            es.activity_ratio = es.active_months / T if T > 0 else 0.0
            series_arr = np.array(edge_series[e], dtype=np.float32)
            es.max_gap = _longest_zero_run(series_arr)
            continuity_penalty = 1.0 - (es.max_gap / T) if T > 0 else 0.0
            es.temporal_score = es.activity_ratio * continuity_penalty
            edge_stats[e] = es

        self._node_stats = node_stats
        self._edge_stats = edge_stats

        logger.info(
            "compute_statistics: %d nodes, %d edges, total flow=%.1f (T=%d)",
            len(node_stats), len(edge_stats), self._total_flow, T,
        )
        return node_stats, edge_stats

    # ── stage B ────────────────────────────────────────────────────────

    def extract_spatial_core(
        self,
        strategy: Optional[str] = None,
    ) -> Set[NodeId]:
        """Select a core set of company nodes using the configured strategy.

        Parameters
        ----------
        strategy : str or None
            Override the configured strategy.  One of ``"flow_core"``,
            ``"k_core"``, ``"greedy_density"``.

        Returns
        -------
        core_nodes : set of node ids
        """
        strategy = strategy or self.config.spatial_strategy

        if strategy == "flow_core":
            return self._extract_flow_core()
        elif strategy == "k_core":
            return self._extract_k_core()
        elif strategy == "greedy_density":
            return self._extract_greedy_density()
        else:
            logger.warning(
                "Unknown spatial_strategy '%s', falling back to flow_core.",
                strategy,
            )
            return self._extract_flow_core()

    def _extract_flow_core(self) -> Set[NodeId]:
        """Greedy flow-preserving core: add nodes in descending node_score
        order until coverage or size bound is met."""
        if not self._node_stats:
            raise RuntimeError("Call compute_statistics() first.")

        sorted_nodes = sorted(
            self._node_stats.items(),
            key=lambda kv: kv[1].node_score,
            reverse=True,
        )

        total = sum(
            ns.flow_volume for ns in self._node_stats.values()
        )
        if total == 0:
            logger.warning("Total flow is zero — returning empty core.")
            return set()

        cumulative = 0.0
        core: Set[NodeId] = set()
        max_n = self.config.max_nodes
        target = self.config.target_coverage

        for node_id, ns in sorted_nodes:
            if len(core) >= max_n:
                break
            core.add(node_id)
            cumulative += ns.flow_volume
            if cumulative / total >= target:
                break

        if len(core) < self.config.min_nodes:
            logger.warning(
                "Core has only %d nodes (min=%d). "
                "Consider lowering target_coverage or raising max_nodes.",
                len(core), self.config.min_nodes,
            )

        logger.info(
            "flow_core: %d nodes, coverage=%.2f%% (target=%.0f%%)",
            len(core), 100 * cumulative / total, 100 * target,
        )
        return core

    def _extract_k_core(self) -> Set[NodeId]:
        """k-core decomposition on the undirected aggregate graph."""
        if not self._edge_stats:
            raise RuntimeError("Call compute_statistics() first.")

        # build an undirected adjacency dict from edge_stats
        adj: Dict[NodeId, Set[NodeId]] = defaultdict(set)
        for (src, tgt) in self._edge_stats:
            if (src, tgt) in self._edge_stats:
                adj[src].add(tgt)
                adj[tgt].add(src)

        # k-core peeling
        degrees = {n: len(neighbors) for n, neighbors in adj.items()}
        max_k = max(degrees.values()) if degrees else 0
        # map from node to its coreness
        coreness: Dict[NodeId, int] = {}
        remaining = set(degrees.keys())
        current_degree = dict(degrees)

        # bin nodes by degree
        bins: Dict[int, Set[NodeId]] = defaultdict(set)
        for n, d in current_degree.items():
            bins[d].add(n)

        for k in range(0, max_k + 1):
            while k in bins and bins[k]:
                n = bins[k].pop()
                if n not in remaining:
                    continue
                coreness[n] = k
                remaining.discard(n)
                for neighbor in list(adj.get(n, set())):
                    if neighbor not in remaining:
                        continue
                    old_d = current_degree[neighbor]
                    if old_d > k:
                        bins[old_d].discard(neighbor)
                        new_d = old_d - 1
                        current_degree[neighbor] = new_d
                        if new_d <= k:
                            bins.setdefault(k, set()).add(neighbor)
                        else:
                            bins.setdefault(new_d, set()).add(neighbor)

        # pick the highest k such that |k-core| is in [min, max]
        k_core_nodes: Dict[int, Set[NodeId]] = defaultdict(set)
        for n, k_val in coreness.items():
            k_core_nodes[k_val].add(n)

        min_n = self.config.min_nodes
        max_n = self.config.max_nodes

        for k in range(max_k, -1, -1):
            size = len(k_core_nodes.get(k, set()))
            if min_n <= size <= max_n:
                logger.info(
                    "k_core: k=%d → %d nodes", k, size,
                )
                return k_core_nodes[k]
            elif size > max_n and k < max_k:
                # k+1 was > max, k is < min but might still be usable
                # fall back to k+1 if it exists
                for k2 in range(k + 1, max_k + 1):
                    s2 = len(k_core_nodes.get(k2, set()))
                    if s2 <= max_n:
                        logger.info(
                            "k_core: k=%d → %d nodes (adjusted)", k2, s2,
                        )
                        return k_core_nodes[k2]

        # fallback: take the largest k-core ≤ max_n
        for k in range(max_k, -1, -1):
            s = len(k_core_nodes.get(k, set()))
            if s <= max_n:
                logger.info(
                    "k_core (fallback): k=%d → %d nodes", k, s,
                )
                return k_core_nodes.get(k, set())

        return set()

    def _extract_greedy_density(self) -> Set[NodeId]:
        """Greedy max-density subgraph starting from top temporal-score
        seed edges."""
        if not self._edge_stats or not self._node_stats:
            raise RuntimeError("Call compute_statistics() first.")

        # sort edges by temporal_score descending
        sorted_edges = sorted(
            self._edge_stats.items(),
            key=lambda kv: kv[1].temporal_score,
            reverse=True,
        )

        core_nodes: Set[NodeId] = set()
        core_edges: Set[Edge] = set()

        max_n = self.config.max_nodes

        for (src, tgt), _es in sorted_edges:
            if len(core_nodes) >= max_n:
                break
            # candidate addition: would adding both endpoints improve density?
            new_nodes = core_nodes | {src, tgt}
            new_edges = core_edges | {(src, tgt)}
            # add edges between existing core nodes and the new ones
            for e_key, _ in self._edge_stats.items():
                s, t = e_key
                if s in new_nodes and t in new_nodes:
                    new_edges.add(e_key)

            n_n = len(new_nodes)
            e_n = len(new_edges)
            new_density = e_n / (n_n * (n_n - 1)) if n_n > 1 else 0.0

            n_o = len(core_nodes)
            e_o = len(core_edges)
            old_density = e_o / (n_o * (n_o - 1)) if n_o > 1 else 0.0

            if new_density >= old_density or len(core_nodes) < self.config.min_nodes:
                core_nodes = new_nodes
                core_edges = new_edges

        logger.info(
            "greedy_density: %d nodes, %d edges, density=%.4f",
            len(core_nodes), len(core_edges),
            len(core_edges) / (len(core_nodes) * (len(core_nodes) - 1))
            if len(core_nodes) > 1 else 0.0,
        )
        return core_nodes

    # ── stage C ────────────────────────────────────────────────────────

    def filter_temporal_density(
        self,
        core_nodes: Set[NodeId],
    ) -> List[Edge]:
        """Filter edges to keep only those whose **both endpoints** are in
        *core_nodes* **and** whose temporal quality passes the configured
        thresholds.

        Parameters
        ----------
        core_nodes : set
            Core company node ids from stage B.

        Returns
        -------
        dense_edges : list of (src, tgt)
        """
        if not self._edge_stats:
            raise RuntimeError("Call compute_statistics() first.")

        min_rho = self.config.min_activity_ratio
        max_gap = self.config.max_allowed_gap
        min_score = self.config.min_temporal_score

        kept: List[Edge] = []
        for e, es in self._edge_stats.items():
            src, tgt = e
            if src not in core_nodes or tgt not in core_nodes:
                continue
            if es.activity_ratio < min_rho:
                continue
            if es.max_gap > max_gap:
                continue
            if es.temporal_score < min_score:
                continue
            kept.append(e)

        logger.info(
            "filter_temporal: %d edges kept (min_activity=%.2f, max_gap=%d, min_score=%.2f)",
            len(kept), min_rho, max_gap, min_score,
        )
        return kept

    # ── quality ────────────────────────────────────────────────────────

    def evaluate_quality(
        self,
        core_nodes: Set[NodeId],
        dense_edges: List[Edge],
    ) -> QualityMetrics:
        """Compute quality indicators for the extracted subgraph.

        Parameters
        ----------
        core_nodes : set
        dense_edges : list

        Returns
        -------
        QualityMetrics
        """
        q = QualityMetrics()
        q.node_count = len(core_nodes)
        q.edge_count = len(dense_edges)

        n = q.node_count
        q.spatial_density = (
            q.edge_count / (n * (n - 1)) if n > 1 else 0.0
        )

        q.total_flow_original = self._total_flow
        q.total_flow_retained = sum(
            self._edge_stats[e].total_weight
            for e in dense_edges
            if e in self._edge_stats
        )
        q.flow_coverage = (
            q.total_flow_retained / self._total_flow
            if self._total_flow > 0 else 0.0
        )

        if n > 0:
            degree_sum = 0
            for node in core_nodes:
                out_deg = len({tgt for (s, tgt) in dense_edges if s == node})
                in_deg = len({src for (src, t) in dense_edges if t == node})
                degree_sum += out_deg + in_deg
            q.avg_degree = degree_sum / n

        if dense_edges:
            activity_ratios = [
                self._edge_stats[e].activity_ratio
                for e in dense_edges if e in self._edge_stats
            ]
            if activity_ratios:
                q.mean_activity = float(np.mean(activity_ratios))
                q.median_activity = float(np.median(activity_ratios))
                q.low_activity_pct = sum(
                    1 for a in activity_ratios if a < 0.3
                ) / len(activity_ratios)

            gaps = [
                self._edge_stats[e].max_gap
                for e in dense_edges if e in self._edge_stats
            ]
            if gaps:
                q.mean_max_gap = float(np.mean(gaps))

        return q

    # ── main entry point ───────────────────────────────────────────────

    def extract(
        self,
        networks: Dict[str, FlowNetwork],
    ) -> DenseSubgraphResult:
        """Run the full extraction pipeline and return a result.

        Parameters
        ----------
        networks : dict  ``{timestamp: FlowNetwork}``

        Returns
        -------
        DenseSubgraphResult
        """
        timestamps = sorted(networks.keys())

        # A
        logger.info("Stage A: computing statistics …")
        self.compute_statistics(networks)

        # B
        logger.info("Stage B: extracting spatial core …")
        core_nodes = self.extract_spatial_core()

        if len(core_nodes) == 0:
            logger.warning("Empty core — returning empty result.")
            return DenseSubgraphResult()

        # C
        logger.info("Stage C: filtering temporal density …")
        dense_edges = self.filter_temporal_density(core_nodes)

        if len(dense_edges) == 0:
            logger.warning(
                "No edges passed temporal filter — "
                "try lowering min_activity_ratio or max_allowed_gap."
            )

        # E
        logger.info("Stage E: building tensor with %s …",
                     type(self.tensor_builder).__name__)
        tensor, adjacency, metadata = self.tensor_builder.build(
            networks, core_nodes, dense_edges, timestamps,
        )

        # D
        logger.info("Quality evaluation …")
        quality = self.evaluate_quality(core_nodes, dense_edges)

        result = DenseSubgraphResult(
            nodes=core_nodes,
            edges=dense_edges,
            tensor=tensor,
            adjacency=adjacency,
            metadata=metadata,
            quality=quality,
        )

        # log summary
        if quality is not None:
            logger.info(
                "Result: nodes=%d edges=%d density=%.4f "
                "coverage=%.2f%% mean_activity=%.3f",
                quality.node_count,
                quality.edge_count,
                quality.spatial_density,
                100 * quality.flow_coverage,
                quality.mean_activity,
            )

        return result
