#!/usr/bin/env python3
"""
Network Sparsity Diagnostics.

Analyse the raw flow-network sequence to characterise spatial and temporal
sparsity, supporting informed parameter selection for the dense-core
extraction pipeline.

Usage:
    python -m scripts.analysis.network_diagnostics
    python -m scripts.analysis.network_diagnostics --data_dir datasets/flow_networks
    python -m scripts.analysis.network_diagnostics --start 2015-01 --end 2019-12
"""

from __future__ import annotations

import sys
import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data import FlowNetworkDataLoader
from scripts.config import DATA_DIR

logger = logging.getLogger(__name__)


# ── result container ───────────────────────────────────────────────────────

@dataclass
class SparsityReport:
    """Aggregated sparsity diagnostics."""

    num_months: int = 0
    total_nodes: int = 0
    total_edges: int = 0
    total_flow: float = 0.0

    # degree distribution
    degree_percentiles: Dict[int, float] = field(default_factory=dict)
    power_law_alpha: float = 0.0          # estimated exponent

    # edge activity
    activity_percentiles: Dict[int, float] = field(default_factory=dict)
    fraction_high_activity: float = 0.0    # ρ ≥ 0.5
    fraction_low_activity: float = 0.0     # ρ < 0.1

    # coverage curves (for flow_core strategy)
    coverage_curve: List[Tuple[int, float]] = field(default_factory=list)
    # recommended thresholds
    recommended_max_nodes: int = 0
    recommended_target_coverage: float = 0.0
    recommended_min_activity: float = 0.0

    def summary(self) -> str:
        """Return a readable summary string."""
        lines = [
            "=" * 64,
            "  Network Sparsity Diagnostics Report",
            "=" * 64,
            f"  Months:          {self.num_months}",
            f"  Total nodes:     {self.total_nodes:,}",
            f"  Total edges:     {self.total_edges:,}",
            f"  Total flow:      {self.total_flow:,.0f}",
            "",
            "  --- Degree Distribution ---",
            f"  Power-law alpha: {self.power_law_alpha:.2f}",
        ]
        for pct, val in sorted(self.degree_percentiles.items()):
            lines.append(f"  P{pct:>4.1f} degree:     {val:,.1f}")

        lines.extend([
            "",
            "  --- Edge Activity (rho = non-zero months / T) ---",
            f"  rho >= 0.5:      {100*self.fraction_high_activity:.1f}% of edges",
            f"  rho < 0.1:       {100*self.fraction_low_activity:.1f}% of edges",
        ])
        for pct, val in sorted(self.activity_percentiles.items()):
            lines.append(f"  P{pct:>4.1f} activity:  {val:.3f}")

        lines.extend([
            "",
            "  --- Coverage Curve (flow_core strategy) ---",
            f"  Coverage @ 50 nodes:   {self._cov_at_50:.2%}" if self._cov_at_50 is not None else "  Coverage @ 50 nodes:   N/A",
            f"  Coverage @ 100 nodes:  {self._cov_at_100:.2%}" if self._cov_at_100 is not None else "  Coverage @ 100 nodes:  N/A",
            f"  Coverage @ 200 nodes:  {self._cov_at_200:.2%}" if self._cov_at_200 is not None else "  Coverage @ 200 nodes:  N/A",
            f"  Recommended max_nodes:          {self.recommended_max_nodes}",
            f"  Recommended target_coverage:    {self.recommended_target_coverage:.2f}",
            f"  Recommended min_activity_ratio: {self.recommended_min_activity:.2f}",
            "",
            "=" * 64,
        ])
        return "\n".join(lines)


# ── main analysis ──────────────────────────────────────────────────────────

def diagnose_network_sparsity(
    data_dir: str = None,
    start_date: str = "2010-01",
    end_date: str = "2020-12",
) -> SparsityReport:
    """Run full sparsity diagnostics on the flow-network sequence.

    Parameters
    ----------
    data_dir : str
        Path to the ``flow_networks/`` directory.
    start_date : str
    end_date : str

    Returns
    -------
    SparsityReport
    """
    if data_dir is None:
        data_dir = str(DATA_DIR)

    logger.info("Loading flow networks from %s [%s → %s] …",
                 data_dir, start_date, end_date)
    loader = FlowNetworkDataLoader(
        data_dir=data_dir,
        start_date=start_date,
        end_date=end_date,
    )
    networks = loader.load_networks()
    T = len(networks)
    logger.info("Loaded %d monthly networks.", T)

    if T == 0:
        logger.error("No networks found.")
        return SparsityReport()

    timestamps = sorted(networks.keys())

    report = SparsityReport(num_months=T)

    # ── accumulate statistics ──────────────────────────────────────────
    node_flow: Dict = defaultdict(float)
    node_partners: Dict = defaultdict(set)
    node_active_months: Dict = defaultdict(set)

    edge_weight: Dict = defaultdict(float)
    edge_active_months: Dict = defaultdict(set)

    total_flow = 0.0

    for t, ts in enumerate(timestamps):
        net = networks[ts]
        for (src, tgt), w in net.get_edges().items():
            if src == tgt:
                continue
            wf = float(w)

            node_flow[src] += wf
            node_flow[tgt] += wf
            node_partners[src].add(tgt)
            node_partners[tgt].add(src)
            node_active_months[src].add(t)
            node_active_months[tgt].add(t)

            e = (src, tgt)
            edge_weight[e] += wf
            edge_active_months[e].add(t)

            total_flow += wf

    report.total_nodes = len(node_flow)
    report.total_edges = len(edge_weight)
    report.total_flow = total_flow

    logger.info("Accumulated: %d nodes, %d edges, %.0f total flow.",
                 report.total_nodes, report.total_edges, total_flow)

    # ── degree distribution ────────────────────────────────────────────
    degrees = [len(partners) for partners in node_partners.values()]
    if degrees:
        report.degree_percentiles = {
            p: float(np.percentile(degrees, p))
            for p in [50, 80, 90, 95, 99, 99.9]
        }
        # crude power-law exponent estimate via log-log regression
        if len(degrees) >= 20:
            hist, bins = np.histogram(degrees, bins=50)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            mask = (hist > 0) & (bin_centers > 0)
            if mask.sum() >= 5:
                coeffs = np.polyfit(
                    np.log(bin_centers[mask]),
                    np.log(hist[mask]),
                    1,
                )
                report.power_law_alpha = -coeffs[0]

    # ── edge activity distribution ─────────────────────────────────────
    activity_ratios = [
        len(edge_active_months[e]) / T
        for e in edge_weight
    ]
    if activity_ratios:
        report.activity_percentiles = {
            p: float(np.percentile(activity_ratios, p))
            for p in [10, 25, 50, 75, 90]
        }
        report.fraction_high_activity = sum(
            1 for a in activity_ratios if a >= 0.5
        ) / len(activity_ratios)
        report.fraction_low_activity = sum(
            1 for a in activity_ratios if a < 0.1
        ) / len(activity_ratios)

    # ── coverage curve (flow_core simulation) ─────────────────────────
    node_scores = {
        n: node_flow[n] * (len(node_active_months[n]) / T)
        for n in node_flow
    }
    sorted_nodes = sorted(
        node_scores.items(), key=lambda kv: kv[1], reverse=True,
    )
    cumulative = 0.0
    coverage_curve = []
    for k, (node_id, score) in enumerate(sorted_nodes, 1):
        cumulative += node_flow[node_id]
        cov = cumulative / total_flow if total_flow > 0 else 0.0
        coverage_curve.append((k, cov))

    report.coverage_curve = coverage_curve

    # ── recommendations ────────────────────────────────────────────────
    # "elbow" heuristic: first k where coverage gain per added node < 0.1%
    # capped to keep the recommendation practical
    recommended_k = min(report.total_nodes, 200)
    for i in range(1, len(coverage_curve)):
        gain = coverage_curve[i][1] - coverage_curve[i - 1][1]
        if gain < 0.0005 and coverage_curve[i][0] >= 20:
            recommended_k = coverage_curve[i][0]
            break

    # also check fixed coverage targets as sanity bounds
    cov_at_50 = next((cov for k, cov in coverage_curve if k == 50), None)
    cov_at_100 = next((cov for k, cov in coverage_curve if k == 100), None)
    cov_at_200 = next((cov for k, cov in coverage_curve if k == 200), None)

    # clamp recommendation: if elbow gave < 20, use coverage-guided fallback
    if recommended_k < 20:
        for k_target in [50, 100, 200]:
            if k_target in dict(coverage_curve):
                recommended_k = k_target
                break

    recommended_cov = next(
        (cov for k, cov in coverage_curve if k == recommended_k),
        coverage_curve[-1][1] if coverage_curve else 0.80,
    )

    # activity threshold: use P30 of activity if that captures enough edges;
    # clamp to [0.01, 0.50] to stay meaningful under extreme sparsity
    if activity_ratios:
        # P40 captures the bottom 40% — edges below this are dropped
        recommended_activity = float(np.percentile(activity_ratios, 40))
        # if activity at P40 is vanishingly small, fall back to a sensible
        # fraction of T (e.g. 2 / T ≈ 0.015) so we get edges present in
        # at least 2 months
        recommended_activity = max(0.01, min(0.50, recommended_activity))
    else:
        recommended_activity = 0.30

    report.recommended_max_nodes = recommended_k
    report.recommended_target_coverage = recommended_cov
    report.recommended_min_activity = recommended_activity

    # attach extra diagnostics for extreme-sparsity cases
    report._cov_at_50 = cov_at_50
    report._cov_at_100 = cov_at_100
    report._cov_at_200 = cov_at_200

    return report


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse flow-network sparsity to guide dense-core extraction."
    )
    parser.add_argument(
        "--data_dir",
        default=str(DATA_DIR),
        help="Path to flow_networks/ directory.",
    )
    parser.add_argument(
        "--start", default="2010-01", help="Start date (YYYY-MM)."
    )
    parser.add_argument(
        "--end", default="2020-12", help="End date (YYYY-MM)."
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save the report as JSON.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    report = diagnose_network_sparsity(
        data_dir=args.data_dir,
        start_date=args.start,
        end_date=args.end,
    )

    print(report.summary())

    if args.save:
        import json
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # convert to serialisable dict
        serialisable = {
            "num_months": report.num_months,
            "total_nodes": report.total_nodes,
            "total_edges": report.total_edges,
            "total_flow": report.total_flow,
            "degree_percentiles": report.degree_percentiles,
            "power_law_alpha": report.power_law_alpha,
            "activity_percentiles": report.activity_percentiles,
            "fraction_high_activity": report.fraction_high_activity,
            "fraction_low_activity": report.fraction_low_activity,
            "coverage_curve": [
                {"nodes": k, "coverage": cov}
                for k, cov in report.coverage_curve
            ],
            "recommended": {
                "max_nodes": report.recommended_max_nodes,
                "target_coverage": report.recommended_target_coverage,
                "min_activity_ratio": report.recommended_min_activity,
            },
        }
        with open(out_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        logger.info("Report saved to %s", out_path)


if __name__ == "__main__":
    main()
