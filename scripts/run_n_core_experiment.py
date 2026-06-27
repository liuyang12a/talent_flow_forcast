#!/usr/bin/env python3
"""n_core sensitivity experiment for core_periphery pooling.

Sweeps the ``n_core`` parameter of the core_periphery pooler over a list of
values, collects :class:`PoolingQualityMetrics` for each, and writes a
JSON + CSV summary plus a 2x2 visualization. Useful for analyzing how the
number of core (hub) nodes trades off densification against information
retention and structure preservation.

Usage:
    # default: degree method, 1000 points linearly spaced from 50 to 10000
    python scripts/run_n_core_experiment.py --start 2010-01 --end 2019-12

    # custom n_core list + k_core method
    python scripts/run_n_core_experiment.py --n-cores 10 30 50 80 120 \
        --core-method k_core --k-core-threshold 5

    # custom output dir
    python scripts/run_n_core_experiment.py --out ckpt/n_core_exp
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from talent_flow.data import FlowNetworkStore
from talent_flow.core.flow_network import merge_networks
from talent_flow.pooling import POOLER_REGISTRY
from talent_flow.utils import ensure_dir, get_logger, load_json, save_json, set_seed

# Eval metric columns (order is the CSV/table column order)
METRIC_COLUMNS: List[str] = [
    "n_core",
    "pooled_K",
    "original_N",
    "compression_ratio",
    "original_density",
    "pooled_density",
    "density_improvement_ratio",
    "zero_reduction",
    "reconstruction_error",
    "spectral_error",
    "modularity",
    "cluster_homogeneity",
]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="core_periphery n_core sensitivity experiment")
    p.add_argument("--data-dir", type=str, default="datasets/flow_networks")
    p.add_argument("--start", type=str, default="2010-01", help="start month YYYY-MM")
    p.add_argument("--end", type=str, default="2019-12", help="end month YYYY-MM")
    p.add_argument(
        "--n-cores",
        type=int,
        nargs="+",
        default=None,
        help="explicit list of n_core values to sweep (overrides --n-core-min/max/num)",
    )
    p.add_argument("--n-core-min", type=int, default=50, help="min n_core of the sweep range")
    p.add_argument("--n-core-max", type=int, default=5000, help="max n_core of the sweep range")
    p.add_argument(
        "--n-core-num",
        type=int,
        default=200,
        help="number of n_core points (linearly spaced, rounded to int)",
    )
    p.add_argument(
        "--core-method",
        type=str,
        default="degree",
        choices=["degree", "k_core"],
        help="core identification method (under k_core, n_core is a truncation cap)",
    )
    p.add_argument(
        "--k-core-threshold",
        type=int,
        default=5,
        help="minimum k-core number for the k_core method",
    )
    p.add_argument("--out", type=str, default="ckpt/n_core_experiment")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="skip visualization (only write JSON/CSV)",
    )
    p.add_argument(
        "--replot",
        action="store_true",
        help="skip pooling; re-read <out>/metrics.json and redraw the figure only",
    )
    p.add_argument(
        "--no-merge-time",
        action="store_true",
        help="disable time-merge optimization (pool month-by-month; uses ~Tx more memory)",
    )
    return p.parse_args(argv)


def run_single(
    networks,
    n_core: int,
    core_method: str,
    k_core_threshold: int,
) -> Dict[str, Any]:
    """Run pooling for a single n_core value; return a flat metrics dict.

    Uses ``mode="assignment"``: builds the assignment and evaluates quality
    from a time-summed K x K (no per-month ``[T,K,K]`` OD series materialized),
    which keeps memory at O(K^2) instead of O(T*K^2). All quality metrics
    remain available.

    ``networks`` is expected to already be either the original month dict or a
    single-month ``{timestamp: merged_net}`` dict (time-merged) — the pooler's
    quality metrics are identical in both cases because the evaluator only
    uses the time-summed OD matrix (linearity: sum_t S^T A_t S == S^T (sum_t
    A_t) S).
    """
    params: Dict[str, Any] = {"n_core": n_core, "core_method": core_method}
    if core_method == "k_core":
        params["k_core_threshold"] = k_core_threshold
    pooler = POOLER_REGISTRY.build("core_periphery", **params)
    result = pooler.pool(networks, mode="assignment")
    q = result.quality
    row: Dict[str, Any] = {"n_core": n_core}
    for col in METRIC_COLUMNS[1:]:
        row[col] = getattr(q, col, None)
    return row


def _fmt(v: Any) -> str:
    """Table-friendly value formatting (None -> N/A, nan -> N/A)."""
    if v is None:
        return "N/A"
    if isinstance(v, float):
        if np.isnan(v):
            return "N/A"
        if abs(v) >= 100:
            return f"{v:.1f}"
        return f"{v:.4f}"
    return str(v)


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in METRIC_COLUMNS})


def print_table(rows: List[Dict[str, Any]], log) -> None:
    """Print an aligned metrics table to the terminal.

    When there are many rows, only the first/last few are shown with an
    ellipsis in between; the full table is always in the CSV/JSON output.
    """
    head, tail = 8, 8
    n = len(rows)
    header = ["n_core", "K", "comp", "dens_x", "recon", "mod", "spec"]

    def _row(r: Dict[str, Any]) -> str:
        vals = [
            _fmt(r["n_core"]),
            _fmt(r["pooled_K"]),
            _fmt(r["compression_ratio"]),
            _fmt(r["density_improvement_ratio"]),
            _fmt(r["reconstruction_error"]),
            _fmt(r["modularity"]),
            _fmt(r["spectral_error"]),
        ]
        return " | ".join(f"{v:>8}" for v in vals)

    lines = ["\n" + " | ".join(f"{h:>8}" for h in header), "-" * (9 * len(header))]
    if n <= head + tail:
        for r in rows:
            lines.append(_row(r))
    else:
        for r in rows[:head]:
            lines.append(_row(r))
        lines.append(f"... ({n - head - tail} rows omitted, see metrics.csv) ...")
        for r in rows[-tail:]:
            lines.append(_row(r))
    log.info("\n".join(lines))


def plot_results(rows: List[Dict[str, Any]], out_path: Path, log) -> None:
    """Plot a 2x2 panel of metrics vs n_core."""
    import matplotlib

    matplotlib.use("Agg")  # headless mode: save straight to file
    import matplotlib.pyplot as plt

    def valid(key):
        """Filter out None/nan, returning (x, y) for plotting."""
        xs, ys = [], []
        for r in rows:
            v = r.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            xs.append(r["n_core"])
            ys.append(v)
        return xs, ys

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"core_periphery pooling quality vs n_core ({rows[0].get('core_method', 'degree')})",
        fontsize=14,
    )

    # (0,0) density improvement ratio (its own panel: magnitude differs from
    # zero_reduction)
    ax = axes[0, 0]
    xs, ys = valid("density_improvement_ratio")
    if xs:
        ax.plot(xs, ys, "o-", color="tab:blue", markersize=3)
    ax.set_xlabel("n_core")
    ax.set_ylabel("density_improvement_ratio")
    ax.set_title("Density improvement ratio (higher is better)")
    ax.grid(alpha=0.3)

    # (0,1) zero-element reduction + pooled_density (twin axis: the two have
    # very different scales, zero_reduction in [0,1], pooled_density small)
    ax = axes[0, 1]
    xs, ys = valid("zero_reduction")
    if xs:
        ax.plot(xs, ys, "o-", color="tab:purple", markersize=3, label="zero-element reduction")
    ax.set_xlabel("n_core")
    ax.set_ylabel("zero_reduction", color="tab:purple")
    ax.tick_params(axis="y", labelcolor="tab:purple")
    ax.set_title("Zero-element reduction & pooled density")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    xs, ys = valid("pooled_density")
    if xs:
        ax2.plot(xs, ys, "s--", color="tab:green", markersize=3, label="pooled_density")
    ax2.set_ylabel("pooled_density", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    # combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if h1 or h2:
        ax.legend(h1 + h2, l1 + l2, loc="best")

    # (1,0) information retention: reconstruction error (lower is better)
    ax = axes[1, 0]
    xs, ys = valid("reconstruction_error")
    if xs:
        ax.plot(xs, ys, "o-", color="tab:green", markersize=3)
    ax.set_xlabel("n_core")
    ax.set_ylabel("reconstruction_error")
    ax.set_title("Information retention: reconstruction error (lower is better)")
    ax.grid(alpha=0.3)

    # (1,1) structure preservation & clustering quality
    ax = axes[1, 1]
    xs, ys = valid("modularity")
    if xs:
        ax.plot(xs, ys, "o-", color="tab:purple", markersize=3, label="modularity")
    xs, ys = valid("spectral_error")
    if xs:
        ax.plot(xs, ys, "s-", color="tab:orange", markersize=3, label="spectral_error")
    else:
        ax.text(0.5, 0.5, "spectral_error all N/A\n(large graph subsampled / skipped)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="gray")
    ax.set_xlabel("n_core")
    ax.set_ylabel("metric value")
    ax.set_title("Structure preservation / clustering quality")
    if ax.has_data():
        ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("visualization saved to %s", out_path)


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    log = get_logger("n_core_exp")

    out_dir = ensure_dir(args.out)

    # --replot: skip pooling, re-read existing metrics.json and redraw only.
    if args.replot:
        metrics_path = Path(out_dir) / "metrics.json"
        if not metrics_path.exists():
            log.error("metrics.json not found at %s (run a full sweep first)", metrics_path)
            return 1
        rows = load_json(metrics_path)
        log.info("replot: loaded %d rows from %s", len(rows), metrics_path)
        print_table(rows, log)
        plot_results(rows, Path(out_dir) / "n_core_sensitivity.png", log)
        return 0

    log.info("loading monthly networks from %s (%s..%s)", args.data_dir, args.start, args.end)
    store = FlowNetworkStore(args.data_dir)
    networks = store.load_range(args.start, args.end)
    if not networks:
        log.error("no networks found in the given range"); return 1
    log.info("loaded %d monthly networks", len(networks))

    # Time-merge optimization (default ON): collapse all months into a single
    # merged network before pooling. The pooling quality metrics are identical
    # to month-by-month pooling (the evaluator only uses sum_t S^T A_t S, which
    # equals S^T (sum_t A_t) S by linearity), but this avoids materializing a
    # [T, K, K] OD array — critical for large K (e.g. K=10000 -> ~96GB at T=120
    # month-by-month, vs ~800MB merged).
    if not args.no_merge_time:
        n_months = len(networks)
        merged = merge_networks(list(networks.values()))
        networks = {"merged": merged}
        log.info("time-merged %d months into 1 network (quality metrics unchanged)", n_months)

    if args.n_cores:
        n_cores = sorted(set(args.n_cores))
    else:
        n_cores = sorted(
            set(
                int(round(v))
                for v in np.linspace(args.n_core_min, args.n_core_max, args.n_core_num)
            )
        )
    log.info(
        "sweeping %d n_core points from %d to %d (core_method=%s)",
        len(n_cores), n_cores[0], n_cores[-1], args.core_method,
    )

    rows: List[Dict[str, Any]] = []
    total = len(n_cores)
    # Report progress roughly every 1% (and on the last point) to avoid
    # flooding the log when sweeping many points.
    report_every = max(1, total // 1000)
    for i, n in enumerate(n_cores, 1):
        row = run_single(networks, n, args.core_method, args.k_core_threshold)
        row["core_method"] = args.core_method
        rows.append(row)
        if i == 1 or i == total or i % report_every == 0:
            log.info(
                "[%d/%d] n_core=%d -> K=%s comp=%s dens_x=%s recon=%s mod=%s",
                i, total, n,
                _fmt(row["pooled_K"]),
                _fmt(row["compression_ratio"]),
                _fmt(row["density_improvement_ratio"]),
                _fmt(row["reconstruction_error"]),
                _fmt(row["modularity"]),
            )

    save_json(rows, Path(out_dir) / "metrics.json")
    write_csv(rows, Path(out_dir) / "metrics.csv")
    log.info("metrics saved: %s/metrics.json, %s/metrics.csv", out_dir, out_dir)

    print_table(rows, log)

    if not args.no_plot:
        plot_results(rows, Path(out_dir) / "n_core_sensitivity.png", log)

    return 0


if __name__ == "__main__":
    sys.exit(main())
