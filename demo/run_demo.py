#!/usr/bin/env python3
"""
Demo Runner for Time Series Forecasting Framework

This script provides a unified interface to:
1. Generate sample data
2. Run forecasting pipeline
3. Visualize results

Usage:
    uv run python demo/run_demo.py [command]

Commands:
    all         - Run complete demo (generate, forecast, visualize)
    generate    - Generate sample data only
    forecast    - Run forecasting only (requires existing data)
    visualize   - Generate visualizations only (requires forecast results)
"""

import sys
import subprocess
from pathlib import Path


def run_generate():
    """Generate sample data."""
    print("\n" + "=" * 60)
    print("STEP 1: Generating Sample Data")
    print("=" * 60 + "\n")

    from demo.generate_sample_data import generate_dataset

    generate_dataset(
        start_date="2022-01",
        end_date="2024-06",
        n_companies=25,
        output_dir="data/flow_networks"
    )


def run_forecast():
    """Run forecasting pipeline."""
    print("\n" + "=" * 60)
    print("STEP 2: Running Forecasting Pipeline")
    print("=" * 60 + "\n")

    from demo.ts_forecast_framework import ForecastingPipeline

    # Configuration
    pipeline = ForecastingPipeline(
        data_dir="data/flow_networks",
        lookback_window=6,
        forecast_horizon=1,
        model_type="ridge"
    )

    results = pipeline.run(
        test_size=3,
        min_observations=12,
        top_k=10
    )

    if results:
        print("\n" + "=" * 60)
        print("FORECASTING RESULTS SUMMARY")
        print("=" * 60)

        import numpy as np
        avg_rmse = np.mean([r['test_rmse'] for r in results.values()])
        avg_mae = np.mean([r['test_mae'] for r in results.values()])
        avg_r2 = np.mean([r['test_r2'] for r in results.values()])

        print(f"Model: Ridge Regression")
        print(f"Flows evaluated: {len(results)}")
        print(f"Average Test RMSE: {avg_rmse:.4f}")
        print(f"Average Test MAE: {avg_mae:.4f}")
        print(f"Average Test R²: {avg_r2:.4f}")
        print("=" * 60)


def run_visualize():
    """Generate visualizations."""
    print("\n" + "=" * 60)
    print("STEP 3: Generating Visualizations")
    print("=" * 60 + "\n")

    from demo.visualize_results import main as viz_main
    viz_main()


def run_all():
    """Run complete demo pipeline."""
    run_generate()
    run_forecast()
    run_visualize()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - data/flow_networks/flow_*.json.gz (sample data)")
    print("  - demo/output/metrics.json")
    print("  - demo/output/predictions.json")
    print("  - demo/output/*.png (visualizations)")
    print("  - demo/output/summary_report.txt")
    print("=" * 60)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable commands:")
        print("  all        - Run complete demo")
        print("  generate   - Generate sample data")
        print("  forecast   - Run forecasting")
        print("  visualize  - Generate visualizations")
        sys.exit(0)

    command = sys.argv[1].lower()

    commands = {
        'all': run_all,
        'generate': run_generate,
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
