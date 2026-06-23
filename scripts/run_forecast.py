#!/usr/bin/env python3
"""Run the forecasting stage on a pre-pooled OD series.

Loads a PoolingResult from disk, fits a forecaster, predicts, evaluates, and
saves the ForecastResult + metrics.

Usage:
    python scripts/run_forecast.py --pooled datasets/pooled/core_periphery \
        --forecaster dmd --rank 20 --out ckpt/forecasts/dmd
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from talent_flow.forecasting import FORECASTER_REGISTRY, SplitRatios, split_od_series
from talent_flow.pipeline import PoolingResultStore, ForecastResultStore
from talent_flow.evaluation import ForecastEvaluator
from talent_flow.utils import get_logger, set_seed, save_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run forecasting stage")
    parser.add_argument("--pooled", type=str, required=True, help="pooled result dir")
    parser.add_argument("--forecaster", type=str, default="dmd")
    parser.add_argument("--input-len", type=int, default=12)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics", nargs="*", default=["mae", "rmse", "mape", "directional_accuracy"])
    # forecaster-specific overrides
    parser.add_argument("--rank", type=int, default=None, help="DMD rank")
    parser.add_argument("--n-factors", type=int, default=None, help="DFM factors")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    log = get_logger("forecast")

    pooling_result = PoolingResultStore().load(args.pooled)
    od = pooling_result.od_series
    log.info("loaded pooled OD: T=%d K=%d", od.T, od.K)

    params = {"input_len": args.input_len, "output_len": args.output_len}
    if args.rank is not None:
        params["rank"] = args.rank
    if args.n_factors is not None:
        params["n_factors"] = args.n_factors

    fc = FORECASTER_REGISTRY.build(args.forecaster, **params)
    train, val, _test = split_od_series(
        od, SplitRatios(args.train_ratio, args.val_ratio, args.test_ratio)
    )
    log.info("fitting %s on train T=%d", args.forecaster, train.T)
    fc.fit(train, val_series=val)
    forecast = fc.predict(od)
    h = args.output_len
    if forecast.ground_truth.shape[0] != h:
        forecast.ground_truth = od.matrix[-h:]

    prev = od.matrix[-h - 1]
    metrics = ForecastEvaluator(metrics=args.metrics).evaluate(forecast, prev_values=prev)
    log.info("metrics (overall): %s", metrics.get("overall"))

    out_dir = args.out or f"ckpt/forecasts/{args.forecaster}"
    path = ForecastResultStore().save(forecast, out_dir)
    save_json(metrics, Path(out_dir) / "metrics.json")
    log.info("saved ForecastResult + metrics to %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
