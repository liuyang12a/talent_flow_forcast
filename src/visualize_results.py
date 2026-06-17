"""
Visualization tools for forecasting results.

Provides plotting functions for:
- Time series with predictions
- Error distributions
- Metric comparisons
- Feature importance
"""

import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def load_results(results_dir: str = "src/output") -> tuple:
    """Load predictions and metrics from output files."""
    results_path = Path(results_dir)

    with open(results_path / "predictions.json", 'r') as f:
        predictions = json.load(f)

    with open(results_path / "metrics.json", 'r') as f:
        metrics = json.load(f)

    return predictions, metrics


def format_flow_key(source, target) -> str:
    """Format flow key for display (handle both int and str IDs)."""
    s = str(source)[:8]  # Truncate long IDs
    t = str(target)[:8]
    return f"{s} → {t}"


def plot_time_series_predictions(
    predictions: List[Dict],
    output_path: str = "src/output/forecast_plots.png",
    n_samples: int = 4
):
    """
    Plot actual vs predicted time series for sample flows.

    Args:
        predictions: List of prediction records
        output_path: Output file path
        n_samples: Number of flows to plot
    """
    # Group predictions by flow
    flows = {}
    for pred in predictions:
        key = (pred['source'], pred['target'])
        if key not in flows:
            flows[key] = []
        flows[key].append(pred)

    # Select top flows by average actual value
    flow_avg = {
        k: np.mean([p['actual'] for p in v])
        for k, v in flows.items()
    }
    top_flows = sorted(flow_avg.items(), key=lambda x: x[1], reverse=True)[:n_samples]

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (flow_key, _) in enumerate(top_flows):
        ax = axes[idx]
        flow_preds = flows[flow_key]

        timestamps = [p['timestamp'] for p in flow_preds]
        actual = [p['actual'] for p in flow_preds]
        predicted = [p['predicted'] for p in flow_preds]

        x = range(len(timestamps))

        ax.plot(x, actual, 'o-', label='Actual', linewidth=2, markersize=8, color='steelblue')
        ax.plot(x, predicted, 's--', label='Predicted', linewidth=2, markersize=6, color='coral')

        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Flow Count', fontsize=11)
        ax.set_title(format_flow_key(flow_key[0], flow_key[1]), fontsize=10, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(timestamps, rotation=45, ha='right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Time series plots saved to {output_path}")


def plot_error_distribution(
    predictions: List[Dict],
    output_path: str = "src/output/error_distribution.png"
):
    """
    Plot error distribution histogram.

    Args:
        predictions: List of prediction records
        output_path: Output file path
    """
    errors = [p['error'] for p in predictions]
    rel_errors = [
        p['error'] / (p['actual'] + 1e-8) * 100
        for p in predictions if p['actual'] > 0
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute errors
    ax1.hist(errors, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(np.mean(errors), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
    ax1.axvline(np.median(errors), color='green', linestyle='--',
                linewidth=2, label=f'Median: {np.median(errors):.2f}')
    ax1.set_xlabel('Absolute Error', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Distribution of Absolute Errors', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Relative errors (filter extreme values for visualization)
    rel_errors_filtered = [e for e in rel_errors if e <= 200]  # Filter >200%
    ax2.hist(rel_errors_filtered, bins=20, edgecolor='black', alpha=0.7, color='coral')
    ax2.axvline(np.mean(rel_errors), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(rel_errors):.1f}%')
    ax2.axvline(np.median(rel_errors), color='green', linestyle='--',
                linewidth=2, label=f'Median: {np.median(rel_errors):.1f}%')
    ax2.set_xlabel('Relative Error (%)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Distribution of Relative Errors', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Error distribution plots saved to {output_path}")


def plot_metrics_comparison(
    metrics: Dict[str, Dict],
    output_path: str = "src/output/metrics_comparison.png"
):
    """
    Plot comparison of metrics across flows.

    Args:
        metrics: Dictionary of metrics per flow
        output_path: Output file path
    """
    # Extract metrics
    flows = list(metrics.keys())
    rmse_values = [m['test_rmse'] for m in metrics.values()]
    mae_values = [m['test_mae'] for m in metrics.values()]
    r2_values = [m['test_r2'] for m in metrics.values()]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # RMSE comparison
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(flows)))
    bars = ax.barh(range(len(flows)), rmse_values, color=colors)
    ax.set_yticks(range(len(flows)))
    ax.set_yticklabels([f[:25] for f in flows], fontsize=8)
    ax.set_xlabel('RMSE', fontsize=11)
    ax.set_title('RMSE by Flow', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # MAE comparison
    ax = axes[0, 1]
    bars = ax.barh(range(len(flows)), mae_values, color=colors)
    ax.set_yticks(range(len(flows)))
    ax.set_yticklabels([f[:25] for f in flows], fontsize=8)
    ax.set_xlabel('MAE', fontsize=11)
    ax.set_title('MAE by Flow', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # R² comparison
    ax = axes[1, 0]
    bars = ax.barh(range(len(flows)), r2_values, color=colors)
    ax.set_yticks(range(len(flows)))
    ax.set_yticklabels([f[:25] for f in flows], fontsize=8)
    ax.set_xlabel('R² Score', fontsize=11)
    ax.set_title('R² by Flow', fontsize=12, fontweight='bold')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')

    # Scatter: RMSE vs R²
    ax = axes[1, 1]
    ax.scatter(rmse_values, r2_values, s=100, alpha=0.6, c=colors)
    for i, flow in enumerate(flows):
        ax.annotate(flow[:15],
                   (rmse_values[i], r2_values[i]),
                   fontsize=7, alpha=0.7)
    ax.set_xlabel('RMSE', fontsize=11)
    ax.set_ylabel('R² Score', fontsize=11)
    ax.set_title('RMSE vs R²', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Metrics comparison plots saved to {output_path}")


def plot_scatter_actual_vs_predicted(
    predictions: List[Dict],
    output_path: str = "src/output/scatter_plot.png"
):
    """
    Plot scatter plot of actual vs predicted values.

    Args:
        predictions: List of prediction records
        output_path: Output file path
    """
    actual = [p['actual'] for p in predictions]
    predicted = [p['predicted'] for p in predictions]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot
    ax.scatter(actual, predicted, alpha=0.6, s=80, c='steelblue', edgecolors='black')

    # Perfect prediction line
    max_val = max(max(actual), max(predicted))
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # Fit line
    z = np.polyfit(actual, predicted, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, max_val, 100)
    ax.plot(x_line, p(x_line), 'g-', linewidth=2,
            label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')

    ax.set_xlabel('Actual Flow Count', fontsize=12)
    ax.set_ylabel('Predicted Flow Count', fontsize=12)
    ax.set_title('Actual vs Predicted Values', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add correlation coefficient
    corr = np.corrcoef(actual, predicted)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Scatter plot saved to {output_path}")


def generate_summary_report(
    predictions: List[Dict],
    metrics: Dict[str, Dict],
    output_path: str = "src/output/summary_report.txt"
):
    """Generate text summary report."""

    lines = []
    lines.append("=" * 70)
    lines.append("TALENT FLOW FORECASTING - SUMMARY REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Overall statistics
    errors = [p['error'] for p in predictions]
    actual = [p['actual'] for p in predictions]
    predicted = [p['predicted'] for p in predictions]

    lines.append("OVERALL STATISTICS")
    lines.append("-" * 70)
    lines.append(f"Total Predictions: {len(predictions)}")
    lines.append(f"Total Flows Analyzed: {len(metrics)}")
    lines.append("")
    lines.append(f"Mean Absolute Error: {np.mean(errors):.4f}")
    lines.append(f"Median Absolute Error: {np.median(errors):.4f}")
    lines.append(f"Std Dev of Errors: {np.std(errors):.4f}")
    lines.append("")

    # Per-flow metrics
    lines.append("PER-FLOW METRICS")
    lines.append("-" * 70)
    lines.append(f"{'Flow':<35} {'RMSE':>8} {'MAE':>8} {'R2':>8}")
    lines.append("-" * 70)

    for flow, m in sorted(metrics.items(), key=lambda x: x[1]['test_rmse'])[:10]:
        flow_short = flow[:34]
        lines.append(f"{flow_short:<35} {m['test_rmse']:>8.3f} {m['test_mae']:>8.3f} {m['test_r2']:>8.3f}")

    lines.append("")
    lines.append("BEST PERFORMING FLOWS (by R2)")
    lines.append("-" * 70)
    best = sorted(metrics.items(), key=lambda x: x[1]['test_r2'], reverse=True)[:5]
    for i, (flow, m) in enumerate(best, 1):
        lines.append(f"{i}. {flow}: R2={m['test_r2']:.3f}, RMSE={m['test_rmse']:.3f}")

    lines.append("")
    lines.append("=" * 70)

    # Write to file
    report_text = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(report_text)

    print(f"Summary report saved to {output_path}")
    print()
    print(report_text)


def main():
    """Main visualization entry point."""
    print("=" * 60)
    print("Forecasting Results Visualization")
    print("=" * 60)
    print()

    # Load results
    print("Loading results...")
    try:
        predictions, metrics = load_results()
    except FileNotFoundError as e:
        print(f"Error: No results found. Please run forecasting first.")
        print(f"Details: {e}")
        return

    print(f"Loaded {len(predictions)} predictions for {len(metrics)} flows")
    print()

    # Generate plots
    print("Generating visualizations...")
    plot_time_series_predictions(predictions)
    plot_error_distribution(predictions)
    plot_metrics_comparison(metrics)
    plot_scatter_actual_vs_predicted(predictions)

    # Generate report
    print()
    generate_summary_report(predictions, metrics)

    print()
    print("=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
