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
    S = np.eye(3)
    assignment = AssignmentMatrix(
        S=S, original_node_ids=[1, 2, 3], supernode_ids=[0, 1, 2]
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
