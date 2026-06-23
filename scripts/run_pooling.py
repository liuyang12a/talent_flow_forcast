#!/usr/bin/env python3
"""Run the pooling stage standalone.

Loads monthly FlowNetworks, runs a pooler, saves the PoolingResult to disk
and prints its intrinsic quality metrics.

Usage:
    python scripts/run_pooling.py --config scripts/configs/default.yaml
    python scripts/run_pooling.py --pooler core_periphery --n_core 50 \
        --start 2015-01 --end 2017-12 --out datasets/pooled/core_periphery
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from talent_flow.data import FlowNetworkStore
from talent_flow.pooling import POOLER_REGISTRY
from talent_flow.pipeline import PoolingResultStore
from talent_flow.utils import get_logger, set_seed


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run pooling stage")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--pooler", type=str, default=None, help="pooler name")
    parser.add_argument("--data-dir", type=str, default="datasets/flow_networks")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--out", type=str, default=None, help="output directory")
    parser.add_argument("--seed", type=int, default=42)
    # common pooler params (override)
    parser.add_argument("--n-core", type=int, default=None)
    args = parser.parse_args(argv)

    set_seed(args.seed)
    log = get_logger("pooling")

    pooler_name = "core_periphery"
    params = {}
    data_cfg = {"data_dir": args.data_dir, "start": args.start, "end": args.end}
    out_dir = args.out

    if args.config:
        from talent_flow.utils import ExperimentConfig

        cfg = ExperimentConfig.from_file(args.config)
        pooler_name = cfg.pooling.name
        params = dict(cfg.pooling.params)
        data_cfg = {
            "data_dir": cfg.data.flow_networks_dir,
            "start": cfg.data.start_date,
            "end": cfg.data.end_date,
        }
        out_dir = out_dir or f"datasets/pooled/{pooler_name}"
    if args.pooler:
        pooler_name = args.pooler
    if args.n_core is not None:
        params["n_core"] = args.n_core
    if out_dir is None:
        out_dir = f"datasets/pooled/{pooler_name}"

    log.info("loading flow networks from %s (%s..%s)", data_cfg["data_dir"],
             data_cfg["start"], data_cfg["end"])
    store = FlowNetworkStore(data_cfg["data_dir"])
    networks = store.load_range(data_cfg["start"], data_cfg["end"])
    if not networks:
        log.error("no networks found in the given range"); return 1
    log.info("loaded %d monthly networks", len(networks))

    log.info("running pooler '%s' with params %s", pooler_name, params)
    pooler = POOLER_REGISTRY.build(pooler_name, **params)
    result = pooler.pool(networks)

    q = result.quality
    log.info("pooling done: N=%d -> K=%d  density %.6f -> %.6f (x%.1f)  recon=%.4f  mod=%.4f",
             q.original_N, q.pooled_K, q.original_density, q.pooled_density,
             q.density_improvement_ratio, q.reconstruction_error, q.modularity)

    store_out = PoolingResultStore()
    path = store_out.save(result, out_dir)
    log.info("saved PoolingResult to %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
