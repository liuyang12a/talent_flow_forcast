#!/usr/bin/env python3
"""Run the full two-stage pooling+forecasting pipeline from a YAML config.

Usage:
    python scripts/run_pipeline.py --config scripts/configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from talent_flow.data import FlowNetworkStore
from talent_flow.pipeline import PoolingForecastPipeline, PoolingResultStore, ForecastResultStore
from talent_flow.utils import ExperimentConfig, get_logger, set_seed, save_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run full two-stage pipeline")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--out", type=str, default=None, help="output root")
    args = parser.parse_args(argv)

    cfg = ExperimentConfig.from_file(args.config)
    set_seed(cfg.seed)
    log = get_logger("pipeline")
    log.info("experiment: %s", cfg.name)

    store = FlowNetworkStore(cfg.data.flow_networks_dir)
    networks = store.load_range(cfg.data.start_date, cfg.data.end_date)
    if not networks:
        log.error("no networks found"); return 1
    log.info("loaded %d monthly networks (%s..%s)", len(networks),
             cfg.data.start_date, cfg.data.end_date)

    pipeline = PoolingForecastPipeline.from_config(cfg.to_dict())
    result = pipeline.run(
        networks,
        metrics=cfg.evaluation.metrics,
    )

    out_root = Path(args.out or cfg.output_dir) / cfg.name
    PoolingResultStore().save(result.pooling, out_root / "pooling")
    ForecastResultStore().save(result.forecast, out_root / "forecast")
    save_json(result.metrics, out_root / "metrics.json")
    save_json(cfg.to_dict(), out_root / "config.json")

    log.info("=== %s ===", cfg.name)
    log.info("pooling: N=%d -> K=%d density x%.1f",
             result.pooling.quality.original_N, result.pooling.quality.pooled_K,
             result.pooling.quality.density_improvement_ratio)
    log.info("forecast overall: %s", result.metrics.get("overall"))
    if "core" in result.metrics:
        log.info("forecast core:     %s", result.metrics["core"])
        log.info("forecast periphery:%s", result.metrics["periphery"])
    log.info("outputs saved to %s", out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
