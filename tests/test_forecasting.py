"""Tests for the forecasting stage."""

import numpy as np

from talent_flow.core import ODMatrixSeries
from talent_flow.forecasting import (
    FORECASTER_REGISTRY,
    SplitRatios,
    split_od_series,
    make_windows,
)


def _trend_od(T=30, K=4, seed=0):
    """OD series with a clear linear trend so DMD/DFM can learn it."""
    rng = np.random.default_rng(seed)
    base = np.linspace(1, 10, T)  # rising trend, shape [T]
    M = np.zeros((T, K, K))
    for i in range(K):
        for j in range(K):
            if i != j:
                M[:, i, j] = base + rng.normal(0, 0.1, T)
    return ODMatrixSeries(
        matrix=M,
        timestamps=[f"2010-{(m % 12) + 1:02d}" for m in range(T)],
        supernode_ids=list(range(K)),
    )


def test_split_and_windows():
    s = _trend_od(T=30, K=4)
    train, val, test = split_od_series(s, SplitRatios(0.7, 0.15, 0.15))
    assert train.T + val.T + test.T == 30
    X, y, starts = make_windows(train, input_len=6, output_len=1)
    assert X.shape[1:] == (6, 4, 4)
    assert y.shape[1:] == (1, 4, 4)


def test_all_forecasters_registered():
    for name in ["naive", "dmd", "dfm"]:
        assert name in FORECASTER_REGISTRY.available()


def test_naive_persistence():
    s = _trend_od(T=20, K=3)
    fc = FORECASTER_REGISTRY.build("naive", input_len=6, output_len=2, strategy="persistence")
    fc.fit(s)
    res = fc.predict(s)
    assert res.predictions.shape == (2, 3, 3)
    # persistence -> equals last matrix
    np.testing.assert_allclose(res.predictions[0], s.matrix[-1])


def test_dmd_fit_predict():
    s = _trend_od(T=30, K=4)
    fc = FORECASTER_REGISTRY.build("dmd", input_len=6, output_len=2, rank=5)
    fc.fit(s)
    res = fc.predict(s)
    assert res.predictions.shape == (2, 4, 4)
    assert np.isfinite(res.predictions).all()


def test_dfm_fit_predict():
    s = _trend_od(T=30, K=4)
    fc = FORECASTER_REGISTRY.build("dfm", input_len=6, output_len=2, n_factors=3, var_lag=1)
    fc.fit(s)
    res = fc.predict(s)
    assert res.predictions.shape == (2, 4, 4)
    assert np.isfinite(res.predictions).all()


def test_forecaster_evaluate_runs():
    s = _trend_od(T=30, K=3)
    fc = FORECASTER_REGISTRY.build("naive", input_len=6, output_len=2)
    fc.fit(s)
    metrics = fc.evaluate(s, metrics=["mae", "rmse"])
    assert "overall" in metrics
    assert metrics["overall"]["mae"] >= 0
