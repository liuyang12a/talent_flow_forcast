"""Tests for the pooling stage."""

import numpy as np

from talent_flow.core import FlowNetwork
from talent_flow.pooling import POOLER_REGISTRY, BasePooler
from talent_flow.pooling.core_periphery import CorePeripheryPooler
from talent_flow.pooling.assignment import build_hard_assignment


def _toy_networks():
    """Build a small synthetic dynamic network over 3 months."""
    nets = {}
    n0 = FlowNetwork.empty()
    n0.add_edge("A", "B", 10)
    n0.add_edge("B", "C", 5)
    n0.add_edge("A", "C", 2)
    nets["2010-01"] = n0
    n1 = FlowNetwork.empty()
    n1.add_edge("A", "B", 8)
    n1.add_edge("B", "C", 6)
    nets["2010-02"] = n1
    n2 = FlowNetwork.empty()
    n2.add_edge("A", "B", 12)
    n2.add_edge("B", "C", 4)
    n2.add_edge("C", "A", 1)
    nets["2010-03"] = n2
    return nets


def test_all_poolers_registered():
    expected = {"core_periphery", "dense_subgraph", "louvain", "semantic", "truncation"}
    assert expected.issubset(set(POOLER_REGISTRY.available()))


def test_truncation_pool_flow():
    nets = _toy_networks()
    res = POOLER_REGISTRY.build("truncation", n_core=2).pool(nets)
    assert res.od_series.K == 2
    assert res.od_series.T == 3
    # core nodes are A and B (highest degree)
    assert set(res.od_series.supernode_ids) == {"A", "B"}
    assert res.quality.original_N == 2  # assignment lists only core nodes
    # A->B flow preserved across months
    a_idx = res.od_series.supernode_ids.index("A")
    b_idx = res.od_series.supernode_ids.index("B")
    assert res.od_series.matrix[0, a_idx, b_idx] == 10.0
    assert res.od_series.matrix[1, a_idx, b_idx] == 8.0


def test_core_periphery_pool_with_attribute_map():
    nets = _toy_networks()
    attrs = {"A": "tech", "B": "tech", "C": "finance"}
    pooler = POOLER_REGISTRY.build(
        "core_periphery", n_core=1, periphery_attribute_map=attrs
    )
    res = pooler.pool(nets)
    # 1 core atom (highest-degree node B) + 2 periphery super-nodes by
    # attribute: A->tech, C->finance. So K = 1 + 2 = 3.
    assert res.od_series.K == 3
    mask = CorePeripheryPooler.get_core_mask(pooler, res.assignment)
    assert mask.sum() == 1  # exactly one core super-node
    # the two periphery super-nodes are tuple-labelled
    periph = [s for s in res.od_series.supernode_ids if isinstance(s, tuple)]
    assert len(periph) == 2


def test_core_periphery_pool_without_attribute_map():
    nets = _toy_networks()
    res = POOLER_REGISTRY.build("core_periphery", n_core=1).pool(nets)
    # 1 core + 1 catch-all periphery = K=2
    assert res.od_series.K == 2
    # periphery is the tuple-labelled super-node
    periph = [s for s in res.od_series.supernode_ids if isinstance(s, tuple)]
    assert len(periph) == 1


def test_semantic_pool():
    nets = _toy_networks()
    attrs = {"A": "tech", "B": "tech", "C": "finance"}
    res = POOLER_REGISTRY.build("semantic", attribute_map=attrs).pool(nets)
    assert res.od_series.K == 2  # {tech, finance}
    # total flow conserved per month (sum of OD matrix == sum of edge weights)
    for t in range(3):
        net_total = sum(w for _, _, w in nets[res.od_series.timestamps[t]].iter_edges())
        assert abs(res.od_series.matrix[t].sum() - net_total) < 1e-6


def test_flow_conservation_truncation():
    """For truncation (drops nodes), flow among retained nodes is conserved."""
    nets = _toy_networks()
    res = POOLER_REGISTRY.build("truncation", n_core=3).pool(nets)
    # all 3 nodes retained -> full flow conserved each month
    for t in range(3):
        net_total = sum(w for _, _, w in nets[res.od_series.timestamps[t]].iter_edges())
        assert abs(res.od_series.matrix[t].sum() - net_total) < 1e-6


def test_build_hard_assignment_validates():
    import pytest

    with pytest.raises(ValueError):
        build_hard_assignment(
            node_to_cluster={"a": "x"},
            original_node_ids=["a", "b"],  # b has no cluster
            supernode_ids=["x"],
        )
