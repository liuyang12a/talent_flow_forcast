"""Integration tests for the two-stage pipeline and persistence."""

import numpy as np

from talent_flow.core import FlowNetwork, ODMatrixSeries
from talent_flow.pooling import POOLER_REGISTRY
from talent_flow.forecasting import FORECASTER_REGISTRY
from talent_flow.pipeline import (
    PoolingForecastPipeline,
    PoolingResultStore,
    ForecastResultStore,
    PipelineResult,
)
from talent_flow.pooling.core_periphery import CorePeripheryPooler


def _toy_networks(months=12):
    nets = {}
    rng = np.random.default_rng(0)
    for m in range(months):
        net = FlowNetwork.empty()
        # core hubs: A, B; periphery: C, D, E
        net.add_edge("A", "B", int(10 + m))
        net.add_edge("B", "A", int(5 + m))
        net.add_edge("A", "C", 2)
        net.add_edge("D", "B", 3)
        net.add_edge("E", "A", 1)
        nets[f"2010-{m+1:02d}"] = net
    return nets


def test_pipeline_core_periphery_dmd(tmp_path):
    nets = _toy_networks(18)
    pooler = POOLER_REGISTRY.build("core_periphery", n_core=2)
    fc = FORECASTER_REGISTRY.build("dmd", input_len=6, output_len=2, rank=3)
    pipe = PoolingForecastPipeline(pooler, fc)
    result = pipe.run(nets, metrics=["mae", "rmse"])
    assert isinstance(result, PipelineResult)
    assert "overall" in result.metrics
    assert result.forecast.predictions.shape == (2, result.pooling.od_series.K, result.pooling.od_series.K)
    # core_mask is available since pooler is CorePeripheryPooler
    assert result.core_mask is not None


def test_pooling_result_store_roundtrip(tmp_path):
    nets = _toy_networks(12)
    pooler = POOLER_REGISTRY.build("truncation", n_core=3)
    result = pooler.pool(nets)
    store = PoolingResultStore()
    d = store.save(result, tmp_path / "pooled")
    loaded = store.load(d)
    np.testing.assert_allclose(loaded.od_series.matrix, result.od_series.matrix)
    assert loaded.od_series.K == result.od_series.K
    assert loaded.pooler_name == result.pooler_name


def test_forecast_result_store_roundtrip(tmp_path):
    from talent_flow.core import ForecastResult
    res = ForecastResult(
        predictions=np.random.rand(2, 3, 3),
        ground_truth=np.random.rand(2, 3, 3),
        forecaster_name="dmd",
    )
    store = ForecastResultStore()
    d = store.save(res, tmp_path / "forecast")
    loaded = store.load(d)
    np.testing.assert_allclose(loaded.predictions, res.predictions)
    assert loaded.forecaster_name == "dmd"


def test_pipeline_from_config():
    cfg = {
        "pooling": {"name": "truncation", "params": {"n_core": 3}},
        "forecasting": {"name": "naive", "params": {"input_len": 4, "output_len": 1}},
        "split": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
    }
    pipe = PoolingForecastPipeline.from_config(cfg)
    assert pipe.pooler.name == "truncation"
    assert pipe.forecaster.name == "naive"
