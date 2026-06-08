#!/usr/bin/env python3
"""
Demo Runner for Time Series Forecasting Framework

This script provides a unified interface to run the forecasting pipeline
on existing flow network data (2017-01 to 2021-12).

Usage:
    uv run python demo/run_demo.py [command]

Commands:
    all         - Run complete demo (forecast + visualize)
    forecast    - Run forecasting only
    visualize   - Generate visualizations only (requires forecast results)
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Configuration
DATA_DIR = "data/flow_networks"
START_DATE = "2017-01"
END_DATE = "2020-12"
MODEL_TYPE = "ridge"
LOOKBACK_WINDOW = 1
FORECAST_HORIZON = 1
TEST_SIZE = 6


def run_forecast():
    """Run forecasting pipeline."""
    print("\n" + "=" * 60)
    print("Running Forecasting Pipeline")
    print("=" * 60)
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Model: {MODEL_TYPE}")
    print("=" * 60 + "\n")

    from ts_forecast_framework import ForecastingPipeline

    pipeline = ForecastingPipeline(
        data_dir=DATA_DIR,
        start_date=START_DATE,
        end_date=END_DATE,
        lookback_window=LOOKBACK_WINDOW,
        forecast_horizon=FORECAST_HORIZON,
        model_type=MODEL_TYPE
    )

    results = pipeline.run(
        test_size=TEST_SIZE,
        min_observations=2,  # Most edges appear only 1-2 times
        top_k=20  # Top 20 flows
    )

    if results:
        import numpy as np
        print("\n" + "=" * 60)
        print("FORECASTING RESULTS SUMMARY")
        print("=" * 60)

        avg_rmse = np.mean([r['test_rmse'] for r in results.values()])
        avg_mae = np.mean([r['test_mae'] for r in results.values()])
        avg_r2 = np.mean([r['test_r2'] for r in results.values()])

        print(f"Date Range: {START_DATE} to {END_DATE}")
        print(f"Model: {MODEL_TYPE}")
        print(f"Flows evaluated: {len(results)}")
        print(f"Average Test RMSE: {avg_rmse:.4f}")
        print(f"Average Test MAE: {avg_mae:.4f}")
        print(f"Average Test R2: {avg_r2:.4f}")
        print("=" * 60)

    return results


def run_visualize():
    """Generate visualizations."""
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60 + "\n")

    from visualize_results import main as viz_main
    viz_main()


def run_all():
    """Run complete demo pipeline."""
    results = run_forecast()

    if results:
        run_visualize()

        print("\n" + "=" * 60)
        print("DEMO COMPLETE!")
        print("=" * 60)
        print("\nGenerated files:")
        print("  - demo/output/metrics.json")
        print("  - demo/output/predictions.json")
        print("  - demo/output/*.png (visualizations)")
        print("  - demo/output/summary_report.txt")
        print("=" * 60)
    else:
        print("\nNo results generated. Please check data availability.")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable commands:")
        print("  all        - Run complete demo")
        print("  forecast   - Run forecasting")
        print("  visualize  - Generate visualizations")
        sys.exit(0)

    command = sys.argv[1].lower()

    commands = {
        'all': run_all,
        'forecast': run_forecast,
        'visualize': run_visualize,
    }

    if command in commands:
        commands[command]()
    else:
        print(f"Unknown command: {command}")
        print(f"Available: {', '.join(commands.keys())}")
        sys.exit(1)


if __name__ == "__main__":
    main()
