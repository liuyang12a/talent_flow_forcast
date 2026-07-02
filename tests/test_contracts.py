"""Tests for core data contracts and the registry."""

import numpy as np
import pytest

from talent_flow.core import (
    FlowNetwork,
    ODMatrixSeries,
    AssignmentMatrix,
    ForecastResult,
    POOLER_REGISTRY,
    FORECASTER_REGISTRY,
    Registry,
)


def _sample_od(T=5, K=3):
    mat = np.zeros((T, K, K), dtype=float)
    mat[0, 0, 1] = 2.0
    return ODMatrixSeries(
        matrix=mat,
        timestamps=[f"2010-{i:02d}" for i in range(1, T + 1)],
        supernode_ids=list(range(K)),
    )


def test_od_matrix_series_shape_validation():
    with pytest.raises(ValueError):
        ODMatrixSeries(
            matrix=np.zeros((5, 3, 4)),
            timestamps=[f"2010-{i:02d}" for i in range(1, 6)],
            supernode_ids=[0, 1, 2],
        )


def test_od_matrix_series_timestamp_length():
    with pytest.raises(ValueError):
        ODMatrixSeries(
            matrix=np.zeros((5, 3, 3)),
            timestamps=["2010-01"],
            supernode_ids=[0, 1, 2],
        )


def test_od_matrix_series_properties():
    s = _sample_od()
    assert s.T == 5
    assert s.K == 3


def test_od_matrix_series_slice():
    s = _sample_od(T=6)
    sub = s.slice(1, 4)
    assert sub.T == 3
    assert sub.K == 3
    assert len(sub.timestamps) == 3


def test_assignment_matrix_validation():
    with pytest.raises(ValueError):
        AssignmentMatrix(
            node_super=np.array([0, 1]),  # length 2
            original_node_ids=["a", "b", "c"],  # length 3 -> mismatch
            supernode_ids=[0, 1],
        )


def test_assignment_matrix_index_out_of_range():
    with pytest.raises(ValueError):
        AssignmentMatrix(
            node_super=np.array([0, 1, 2]),  # 2 is out of range for K=2
            original_node_ids=["a", "b", "c"],
            supernode_ids=[0, 1],
        )


def test_assignment_matrix_index_storage():
    """node_super is a compact int index array; large N is cheap."""
    am = AssignmentMatrix(
        node_super=np.array([0, 2, 1, 0, 2]),
        original_node_ids=["a", "b", "c", "d", "e"],
        supernode_ids=["s0", "s1", "s2"],
    )
    assert am.N == 5
    assert am.K == 3
    assert am.node_super.dtype.kind in ("i", "u")
    assert am.node_super.tolist() == [0, 2, 1, 0, 2]


def test_forecast_result_shape_mismatch():
    with pytest.raises(ValueError):
        ForecastResult(
            predictions=np.zeros((3, 4, 4)),
            ground_truth=np.zeros((3, 4, 5)),
        )


def test_registry_register_and_build():
    reg = Registry("test")
    sentinel = {}

    @reg.register("foo")
    class Foo:
        def __init__(self, x=None):
            sentinel["x"] = x

    assert "foo" in reg
    assert reg.available() == ["foo"]
    reg.build("foo", x=42)
    assert sentinel["x"] == 42

    with pytest.raises(KeyError):
        reg.build("bar")


def test_flownetwork_roundtrip():
    net = FlowNetwork.empty()
    net.add_edge(1, 2, weight=3)
    net.add_edge(2, 1, weight=1)
    assert net.get_edge_weight(1, 2) == 3
    assert net.get_total_flow() == 4
    assert net.get_node_count() == 2
    mat, nodes = net.to_adjacency_matrix(node_order=[1, 2])
    assert mat[0][1] == 3 and mat[1][0] == 1
