"""Tests for the unified evaluation framework."""

import numpy as np

from talent_flow.core import (
    FlowNetwork,
    AssignmentMatrix,
    ODMatrixSeries,
    ForecastResult,
    PoolingQualityMetrics,
)
from talent_flow.evaluation import (
    calculate_metrics,
    ForecastEvaluator,
    PoolingQualityEvaluator,
    ProbabilisticEvaluator,
    SignificanceTester,
    ReportGenerator,
    EvaluationReport,
)


def test_metrics_argument_order():
    target = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 2.0, 3.0])
    assert calculate_metrics(target, pred, ["mae"])["mae"] == 0.0
    pred_off = np.array([2.0, 3.0, 4.0])
    assert calculate_metrics(target, pred_off, ["mae"])["mae"] == 1.0


def test_directional_accuracy():
    target = np.array([5.0, 3.0])
    pred = np.array([5.0, 3.0])
    prev = np.array([4.0, 4.0])
    res = calculate_metrics(target, pred, ["directional_accuracy"], prev=prev)
    assert res["directional_accuracy"] == 1.0


def test_forecast_evaluator_overall():
    rng = np.random.default_rng(0)
    gt = rng.random((4, 5, 5)) * 10
    pr = gt + rng.standard_normal((4, 5, 5))
    res = ForecastResult(predictions=pr, ground_truth=gt, forecaster_name="t")
    ev = ForecastEvaluator(metrics=["mae", "rmse"])
    rep = ev.evaluate(res)
    assert "overall" in rep
    assert rep["overall"]["mae"] >= 0


def test_forecast_evaluator_core_periphery_split():
    gt = np.ones((2, 4, 4))
    pr = np.ones((2, 4, 4)) * 2
    res = ForecastResult(predictions=pr, ground_truth=gt)
    core_mask = np.array([True, True, False, False])
    ev = ForecastEvaluator(metrics=["mae"])
    rep = ev.evaluate(res, core_mask=core_mask)
    assert "core" in rep and "periphery" in rep


def test_probabilistic_evaluator():
    target = np.array([0.5])
    intervals = {"lower": np.array([0.0]), "upper": np.array([1.0]), "level": 0.9}
    out = ProbabilisticEvaluator().evaluate(target, intervals)
    assert out["picp"] == 1.0


def test_significance_wilcoxon():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [1.5, 2.5, 3.5, 4.5]
    out = SignificanceTester("wilcoxon").compare(a, b)
    assert "p_value" in out and out["n"] == 4


def test_pooling_quality_evaluator():
    # build a small aggregated network and a hard assignment
    net = FlowNetwork.empty()
    net.add_edge(1, 2, 5)
    net.add_edge(2, 3, 3)
    net.add_edge(3, 1, 2)
    networks = {"2010-01": net}
    # assign nodes 1,2,3 -> supernodes; keep as identity (K=N=3)
    node_super = np.array([0, 1, 2])
    assignment = AssignmentMatrix(
        node_super=node_super, original_node_ids=[1, 2, 3], supernode_ids=[0, 1, 2]
    )
    od = ODMatrixSeries(
        matrix=np.array([[[0, 5, 0], [0, 0, 3], [2, 0, 0]]], dtype=float),
        timestamps=["2010-01"],
        supernode_ids=[0, 1, 2],
    )
    q = PoolingQualityEvaluator().evaluate(networks, assignment, od)
    assert isinstance(q, PoolingQualityMetrics)
    assert q.original_N == 3 and q.pooled_K == 3
    assert q.reconstruction_error is not None


def test_report_generator_markdown():
    reports = [
        EvaluationReport(
            method_name="dmd",
            method_type="forecasting",
            forecast_metrics={"overall": {"mae": 1.5, "rmse": 2.0}},
        ),
        EvaluationReport(
            method_name="arima",
            method_type="forecasting",
            forecast_metrics={"overall": {"mae": 2.5, "rmse": 3.0}},
        ),
    ]
    rows = ReportGenerator().generate_forecast_table(reports)
    md = ReportGenerator().to_markdown(rows)
    assert "dmd" in md and "arima" in md


def _two_community_network(n_per: int = 30, intra: float = 5.0, inter: float = 0.1):
    """Two dense communities loosely connected: nodes 0..n_per-1 and
    n_per..2n_per-1. Strong intra-community edges, weak inter edges."""
    net = FlowNetwork.empty()
    for c in (0, n_per):
        for i in range(c, c + n_per):
            for j in range(c, c + n_per):
                if i != j:
                    net.add_edge(i, j, intra)
    # a few weak inter edges
    for i in range(n_per):
        net.add_edge(i, n_per + i, inter)
    return net


def test_modularity_full_graph():
    """Sparse full-graph modularity should detect strong community structure."""
    net = _two_community_network(n_per=30)
    networks = {"2010-01": net}
    # assignment: each node -> its own community (2 communities)
    n = 60
    node_super = np.array([0] * 30 + [1] * 30)
    assignment = AssignmentMatrix(
        node_super=node_super, original_node_ids=list(range(n)), supernode_ids=[0, 1]
    )
    od = ODMatrixSeries(
        matrix=np.zeros((1, 2, 2)), timestamps=["2010-01"], supernode_ids=[0, 1]
    )
    q = PoolingQualityEvaluator().evaluate(networks, assignment, od)
    # strong two-community structure -> Q should be clearly positive
    assert q.modularity > 0.5


def test_stratified_sample_covers_core():
    """Stratified sample must include nodes from every super-node, including a
    tiny core super-node that a uniform sample would almost surely miss."""
    from talent_flow.evaluation.pooling_eval import _stratified_sample

    N, K = 10000, 3
    # super-node 0 (core): only 5 nodes; super-nodes 1 and 2: ~5000 each
    node_super = np.array([0] * 5 + [1] * 4995 + [2] * 5000)
    assignment = AssignmentMatrix(
        node_super=node_super, original_node_ids=list(range(N)), supernode_ids=[0, 1, 2]
    )
    keep = _stratified_sample(assignment, n_sample=200)
    sampled_communities = set(assignment.node_super[keep].tolist())
    # all three super-nodes (incl. the 5-node core) must be represented
    assert sampled_communities == {0, 1, 2}
    assert len(keep) <= 200


def test_spectral_no_nan_large_n():
    """On a large-N assignment, spectral_error must not be NaN (computed on
    the pooled K x K graph + stratified sample)."""
    net = _two_community_network(n_per=200)
    networks = {"2010-01": net}
    n = 400
    node_super = np.array([0] * 200 + [1] * 200)
    assignment = AssignmentMatrix(
        node_super=node_super, original_node_ids=list(range(n)), supernode_ids=[0, 1]
    )
    # non-zero pooled OD so the pooled spectrum is well-defined
    od = ODMatrixSeries(
        matrix=np.array([[[10.0, 0.5], [0.5, 10.0]]]),
        timestamps=["2010-01"],
        supernode_ids=[0, 1],
    )
    # force the sampling branch (N=400 > cap=100); small k for K=2
    q = PoolingQualityEvaluator(spectral_max_nodes=100, n_eigenvalues=1).evaluate(
        networks, assignment, od
    )
    assert q.spectral_error is not None
    assert not np.isnan(q.spectral_error)
