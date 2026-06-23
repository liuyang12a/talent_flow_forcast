#!/usr/bin/env python3
"""Cross experiment: poolers x forecasters full factorial.

Runs every (pooler, forecaster) combination on a shared time range and
collects metrics into a comparison table. Useful for the experiment matrix in
the refactoring plan (experiment 1 + 2).

Usage:
    python scripts/run_cross_experiment.py \
        --poolers truncation core_periphery \
        --forecasters naive dmd dfm \
        --start 2015-01 --end 2017-12 \
        --out ckpt/cross_experiment
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

from talent_flow.data import FlowNetworkStore
from talent_flow.evaluation import ReportGenerator, EvaluationReport
from talent_flow.pipeline import PoolingForecastPipeline, PoolingResultStore
from talent_flow.forecasting import SplitRatios
from talent_flow.utils import get_logger, set_seed, save_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cross experiment")
    parser.add_argument("--data-dir", type=str, default="datasets/flow_networks")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--poolers", nargs="+", default=["truncation", "core_periphery"])
    parser.add_argument("--forecasters", nargs="+", default=["naive", "dmd", "dfm"])
    parser.add_argument("--n-core", type=int, default=30)
    parser.add_argument("--input-len", type=int, default=12)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--out", type=str, default="ckpt/cross_experiment")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    set_seed(args.seed)
    log = get_logger("cross")

    store = FlowNetworkStore(args.data_dir)
    networks = store.load_range(args.start, args.end)
    if not networks:
        log.error("no networks found"); return 1
    log.info("loaded %d networks", len(networks))

    # cache pooling results per pooler (reuse across forecasters)
    pooled_cache = {}
    for pooler_name in args.poolers:
        params = {"n_core": args.n_core} if pooler_name in ("truncation", "core_periphery") else {}
        pooler = __build_pooler(pooler_name, params)
        result = pooler.pool(networks)
        out_dir = Path(args.out) / "pooled" / pooler_name
        PoolingResultStore().save(result, out_dir)
        pooled_cache[pooler_name] = result
        log.info("[pool] %s: K=%d density x%.1f", pooler_name, result.od_series.K,
                 result.quality.density_improvement_ratio)

    reports = []
    for pooler_name, forecaster_name in itertools.product(args.poolers, args.forecasters):
        pooling_result = pooled_cache[pooler_name]
        from talent_flow.pooling import POOLER_REGISTRY
        from talent_flow.forecasting import FORECASTER_REGISTRY
        pooler = POOLER_REGISTRY.get(pooler_name)(**({"n_core": args.n_core} if pooler_name in ("truncation","core_periphery") else {}))
        fc = FORECASTER_REGISTRY.build(
            forecaster_name, input_len=args.input_len, output_len=args.output_len
        )
        pipeline = PoolingForecastPipeline(pooler, fc)
        # reuse the already-pooled OD series
        from talent_flow.forecasting import split_od_series
        od = pooling_result.od_series
        train, val, _ = split_od_series(od, SplitRatios(0.7, 0.15, 0.15))
        fc.fit(train, val_series=val)
        forecast = fc.predict(od)
        h = args.output_len
        if forecast.ground_truth.shape[0] != h:
            forecast.ground_truth = od.matrix[-h:]
        from talent_flow.evaluation import ForecastEvaluator
        prev = od.matrix[-h - 1]
        metrics = ForecastEvaluator(metrics=["mae","rmse","mape"]).evaluate(forecast, prev_values=prev)
        combo = f"{pooler_name}+{forecaster_name}"
        log.info("[run] %s: %s", combo, metrics.get("overall"))
        reports.append(EvaluationReport(
            method_name=combo, method_type="pipeline",
            forecast_metrics=metrics,
            metadata={"pooler": pooler_name, "forecaster": forecaster_name},
        ))

    gen = ReportGenerator()
    rows = gen.generate_forecast_table(reports)
    md = gen.to_markdown(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cross_results.md").write_text(md, encoding="utf-8")
    save_json(rows, out / "cross_results.json")
    log.info("comparison table written to %s/cross_results.md", out)
    print("\n" + md)
    return 0


def __build_pooler(name, params):
    from talent_flow.pooling import POOLER_REGISTRY
    return POOLER_REGISTRY.build(name, **params)


if __name__ == "__main__":
    sys.exit(main())
