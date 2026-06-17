#!/usr/bin/env python3
"""
Experiment script for ARIMA model on talent flow data.

This script demonstrates how to use the new data module and ARIMA model
to forecast talent flows.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import numpy as np
from datetime import datetime

from demo.data import FlowNetworkDataLoader, HighWeightSelector
from demo.data.transforms import ZScoreScaler
from demo.models.statistical import ARIMAModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run ARIMA experiment."""
    print("=" * 60)
    print("ARIMA Experiment on Talent Flow Data")
    print("=" * 60)

    # Configuration
    DATA_DIR = "datasets/flow_networks"
    START_DATE = "2017-01"
    END_DATE = "2020-12"
    INPUT_LEN = 6
    OUTPUT_LEN = 1
    TOP_K_EDGES = 50

    # Load data
    logger.info("Loading flow networks...")
    loader = FlowNetworkDataLoader(
        data_dir=DATA_DIR,
        start_date=START_DATE,
        end_date=END_DATE
    )
    networks = loader.load_networks()
    logger.info(f"Loaded {len(networks)} monthly networks")

    # Select high-weight edges
    logger.info(f"Selecting top {TOP_K_EDGES} edges by weight...")
    selector = HighWeightSelector(top_k=TOP_K_EDGES, min_months=6)
    edges = selector.select(networks)
    logger.info(f"Selected {len(edges)} edges")

    # Create datasets
    logger.info("Creating datasets...")
    scaler = ZScoreScaler()
    train_ds, val_ds, test_ds = loader.create_datasets(
        edges=edges,
        input_len=INPUT_LEN,
        output_len=OUTPUT_LEN,
        scaler=scaler,
        overlap=True
    )
    logger.info(f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Prepare data for ARIMA
    # ARIMA works on univariate series, so we train separate models
    logger.info("Training ARIMA models...")

    # Get raw data for all series
    train_data = train_ds.get_raw_series()  # [T, num_edges]

    # Train one ARIMA model per series
    models = []
    for i in range(len(edges)):
        series = train_data[:, i]

        # Create sliding windows
        X, y = [], []
        for t in range(len(series) - INPUT_LEN - OUTPUT_LEN + 1):
            X.append(series[t:t+INPUT_LEN])
            y.append(series[t+INPUT_LEN:t+INPUT_LEN+OUTPUT_LEN])
        X = np.array(X)
        y = np.array(y)

        if len(X) < 5:
            logger.warning(f"Not enough data for series {i}")
            models.append(None)
            continue

        # Train ARIMA
        model = ARIMAModel(
            input_len=INPUT_LEN,
            output_len=OUTPUT_LEN,
            order=(2, 1, 2),
            name=f"ARIMA_edge_{i}"
        )

        try:
            model.fit(X, y)
            models.append(model)
            logger.info(f"Trained ARIMA for series {i}")
        except Exception as e:
            logger.warning(f"Failed to train ARIMA for series {i}: {e}")
            models.append(None)

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_data = test_ds.get_raw_series()

    all_predictions = []
    all_targets = []

    for i, model in enumerate(models):
        if model is None:
            continue

        series = test_data[:, i]

        # Create test windows
        X_test, y_test = [], []
        for t in range(len(series) - INPUT_LEN - OUTPUT_LEN + 1):
            X_test.append(series[t:t+INPUT_LEN])
            y_test.append(series[t+INPUT_LEN:t+INPUT_LEN+OUTPUT_LEN])

        if len(X_test) == 0:
            continue

        X_test = np.array(X_test)
        y_test = np.array(y_test)

        try:
            predictions = model.predict(X_test)
            all_predictions.extend(predictions.flatten())
            all_targets.extend(y_test.flatten())
        except Exception as e:
            logger.warning(f"Prediction failed for series {i}: {e}")

    # Calculate metrics
    if all_predictions:
        from demo.utils.metrics import calculate_metrics

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        metrics = calculate_metrics(all_targets, all_predictions)

        print("\n" + "=" * 60)
        print("ARIMA Experiment Results")
        print("=" * 60)
        print(f"Number of series: {len([m for m in models if m is not None])}")
        print(f"Test samples: {len(all_predictions)}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
        print(f"R²: {metrics['r2']:.4f}")
        print("=" * 60)

        # Save results
        output_dir = Path("demo/output")
        output_dir.mkdir(parents=True, exist_ok=True)

        np.savez(
            output_dir / "arima_results.npz",
            predictions=all_predictions,
            targets=all_targets,
            metrics=metrics
        )
        logger.info(f"Results saved to {output_dir / 'arima_results.npz'}")


if __name__ == "__main__":
    main()
