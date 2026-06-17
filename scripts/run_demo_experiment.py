#!/usr/bin/env python3
"""
Quick Demo Experiment Runner

This script runs a quick demo experiment using isolated demo configuration.
Results are saved to demo_output/ instead of ckpt/ to avoid polluting
production experiment data.

Usage:
    python scripts/run_demo_experiment.py

The demo uses:
- Reduced dataset (30 edges instead of 300)
- Fewer epochs (20 instead of 100)
- Simplified model configs (single variant)
- Isolated output directory (demo_output/)
"""

import sys
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.demo_config import (
    DEMO_CONFIG, DEMO_PATHS, get_demo_config, init_demo_directories
)
from scripts.config import DATA_CONFIG
from src.models.statistical import ARIMAModel
from src.models.deep_learning import STGNNModel
from src.utils.metrics import calculate_metrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(DEMO_PATHS.ckpt_dir / 'demo_run.log')
    ]
)
logger = logging.getLogger(__name__)


def load_series_data_demo(selector_name: str = 'high_weight'):
    """Load or generate demo series data."""
    input_dir = DEMO_PATHS.get_series_path(selector_name)

    # Check if demo series already exists
    if input_dir.exists() and (input_dir / 'series_data.npz').exists():
        logger.info(f"Loading existing demo series from {input_dir}")
        data = np.load(input_dir / 'series_data.npz', allow_pickle=True)
        adj_matrix = np.load(input_dir / 'adjacency_matrix.npy')
        with open(input_dir / 'characteristics.json', 'r') as f:
            characteristics = json.load(f)
        return data['series_matrix'], data['edges'].tolist(), adj_matrix, characteristics

    # Generate demo series (simplified)
    logger.info("Generating demo series data...")
    input_dir.mkdir(parents=True, exist_ok=True)

    # Create synthetic data for demo
    n_timestamps = 48  # 4 years of monthly data
    n_edges = DEMO_CONFIG['selectors']['high_weight']['top_k']

    np.random.seed(DEMO_CONFIG['random_seed'])

    # Generate synthetic time series with some structure
    series_matrix = np.zeros((n_timestamps, n_edges))
    for i in range(n_edges):
        base = np.random.uniform(1, 10)
        trend = np.linspace(0, np.random.uniform(-2, 2), n_timestamps)
        seasonal = 2 * np.sin(2 * np.pi * np.arange(n_timestamps) / 12)
        noise = np.random.normal(0, 0.5, n_timestamps)
        series_matrix[:, i] = np.maximum(0, base + trend + seasonal + noise)

    # Create synthetic edges
    edges = [(f"company_{i}", f"company_{j}") for i, j in zip(range(n_edges), range(1, n_edges + 1))]

    # Create synthetic adjacency matrix
    adj_matrix = np.eye(n_edges) + np.random.rand(n_edges, n_edges) * 0.3
    adj_matrix = (adj_matrix + adj_matrix.T) / 2  # Make symmetric
    adj_matrix = (adj_matrix > 0.5).astype(np.float32)

    # Create synthetic characteristics
    characteristics = {}
    for i in range(n_edges):
        char = {
            'total_volume': float(np.sum(series_matrix[:, i])),
            'volatility': float(np.std(series_matrix[:, i])),
            'classification': {
                'volume': 'high' if np.sum(series_matrix[:, i]) > np.median(np.sum(series_matrix, axis=0)) else 'low',
                'volatility': 'high' if np.std(series_matrix[:, i]) > 1.0 else 'low'
            }
        }
        characteristics[f"high_weight_edge_{i}"] = char

    # Save demo series data
    np.savez(
        input_dir / 'series_data.npz',
        series_matrix=series_matrix,
        timestamps=np.array([f"2017-{m:02d}" for m in range(1, 13)] +
                           [f"2018-{m:02d}" for m in range(1, 13)] +
                           [f"2019-{m:02d}" for m in range(1, 13)] +
                           [f"2020-{m:02d}" for m in range(1, 13)]),
        edges=np.array(edges, dtype=object)
    )
    np.save(input_dir / 'adjacency_matrix.npy', adj_matrix)
    with open(input_dir / 'characteristics.json', 'w') as f:
        json.dump(characteristics, f, indent=2)

    logger.info(f"Demo series saved to {input_dir}")
    return series_matrix, edges, adj_matrix, characteristics


def create_windows(series: np.ndarray, input_len: int, output_len: int):
    """Create sliding windows from a time series."""
    X, y = [], []
    for t in range(len(series) - input_len - output_len + 1):
        X.append(series[t:t + input_len])
        y.append(series[t + input_len:t + input_len + output_len])
    return np.array(X), np.array(y)


def prepare_stgnn_data(input_data: np.ndarray, target_data: np.ndarray,
                       input_len: int, output_len: int):
    """Prepare data for STGNN model."""
    if len(input_data) < input_len + output_len:
        return None, None

    X_list, y_list = [], []
    for t in range(len(input_data) - input_len - output_len + 1):
        X = input_data[t:t + input_len, :, np.newaxis]
        y_end = t + input_len + output_len
        if y_end <= len(target_data):
            y = target_data[t + input_len:y_end, :, np.newaxis]
            X_list.append(X)
            y_list.append(y)

    if len(X_list) == 0:
        return None, None

    return np.array(X_list), np.array(y_list)


def run_demo_arima(series_matrix, edges, characteristics, input_len=6, output_len=1):
    """Run ARIMA demo experiment."""
    logger.info("=" * 60)
    logger.info("Running ARIMA Demo Experiment")
    logger.info("=" * 60)

    n_series = min(10, series_matrix.shape[1])  # Limit to 10 series for demo
    results = []

    for i in range(n_series):
        series = series_matrix[:, i]
        X, y = create_windows(series, input_len, output_len)

        if len(X) < 10:
            continue

        # Split
        train_size = int(len(X) * 0.7)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        try:
            # Try ARIMA(1,1,1) for demo speed
            model = ARIMAModel(input_len=input_len, output_len=output_len, order=(1, 1, 1))
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            metrics = calculate_metrics(y_test.flatten(), predictions.flatten())

            result = {
                'series_id': f"high_weight_edge_{i}",
                'model_type': 'arima',
                'selector_type': 'high_weight',
                'edge_idx': i,
                'input_len': input_len,
                'output_len': output_len,
                'metrics': metrics,
                'order': (1, 1, 1)
            }

            char = characteristics.get(f"high_weight_edge_{i}", {})
            if 'classification' in char:
                for key, value in char['classification'].items():
                    result[f'char_{key}'] = value

            results.append(result)
            logger.info(f"ARIMA series {i}: MAE={metrics['mae']:.4f}")

        except Exception as e:
            logger.warning(f"ARIMA failed for series {i}: {e}")

    logger.info(f"ARIMA completed: {len(results)} series")
    return results


def run_demo_stgnn(series_matrix, edges, adj_matrix, characteristics, input_len=6, output_len=1):
    """Run STGNN demo experiment."""
    logger.info("=" * 60)
    logger.info("Running STGNN Demo Experiment")
    logger.info("=" * 60)

    # Limit nodes for demo
    max_nodes = 10
    n_nodes = min(max_nodes, series_matrix.shape[1])
    series_matrix = series_matrix[:, :n_nodes]
    adj_matrix = adj_matrix[:n_nodes, :n_nodes]

    # Split data
    n_samples = series_matrix.shape[0]
    train_size = int(n_samples * 0.7)
    val_size = int(n_samples * 0.1)

    train_data = series_matrix[:train_size]
    val_data = series_matrix[train_size:train_size + val_size]
    test_data = series_matrix[train_size + val_size:]

    # Prepare data
    X_train, y_train = prepare_stgnn_data(train_data, train_data, input_len, output_len)
    X_val, y_val = prepare_stgnn_data(val_data, val_data, input_len, output_len)
    X_test, y_test = prepare_stgnn_data(test_data[:-output_len], test_data[input_len:], input_len, output_len)

    if X_train is None or len(X_train) < 5:
        logger.error("Not enough data for STGNN")
        return []

    logger.info(f"Data shapes: X_train={X_train.shape}, y_train={y_train.shape}")

    results = []
    try:
        model = STGNNModel(
            input_len=input_len, output_len=output_len, num_nodes=n_nodes,
            adjacency_matrix=adj_matrix, input_dim=1, hidden_dim=32, num_layers=2,
            spatial_type='gcn', temporal_type='gru', device='auto'
        )

        model.fit(X_train, y_train, X_val=X_val, y_val=y_val,
                  epochs=DEMO_CONFIG['models']['stgnn']['training']['epochs'],
                  batch_size=8, learning_rate=0.001, early_stopping_patience=5)

        predictions = model.predict(X_test, batch_size=8)

        # Evaluate per series
        for i in range(n_nodes):
            series_id = f"high_weight_edge_{i}"
            pred_series = predictions[:, :, i, 0].flatten()
            target_series = y_test[:, :, i, 0].flatten()
            metrics = calculate_metrics(target_series, pred_series)

            result = {
                'series_id': series_id,
                'model_type': 'stgnn',
                'selector_type': 'high_weight',
                'edge_idx': i,
                'input_len': input_len,
                'output_len': output_len,
                'metrics': metrics
            }

            char = characteristics.get(series_id, {})
            if 'classification' in char:
                for key, value in char['classification'].items():
                    result[f'char_{key}'] = value

            results.append(result)

        logger.info(f"STGNN completed: {len(results)} series")

    except Exception as e:
        logger.error(f"STGNN failed: {e}")
        import traceback
        traceback.print_exc()

    return results


def main():
    """Run demo experiment."""
    print("=" * 70)
    print("DEMO EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"Output directory: {DEMO_PATHS.ckpt_dir}")
    print("(Isolated from production ckpt/)")
    print("=" * 70)

    # Initialize demo directories
    init_demo_directories()

    # Load demo data
    logger.info("Loading demo series data...")
    series_matrix, edges, adj_matrix, characteristics = load_series_data_demo()
    logger.info(f"Loaded {len(edges)} edges, {series_matrix.shape[0]} timestamps")

    input_len = 6
    output_len = 1

    # Run ARIMA
    arima_results = run_demo_arima(series_matrix, edges, characteristics, input_len, output_len)

    # Run STGNN
    stgnn_results = run_demo_stgnn(series_matrix, edges, adj_matrix, characteristics, input_len, output_len)

    # Combine results
    all_results = arima_results + stgnn_results

    # Save results
    results_path = DEMO_PATHS.get_metrics_path('demo_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Save as CSV
    df = pd.DataFrame(all_results)
    if 'metrics' in df.columns:
        metrics_df = pd.json_normalize(df['metrics'])
        metrics_df.columns = [f'metric_{c}' for c in metrics_df.columns]
        df = pd.concat([df.drop('metrics', axis=1), metrics_df], axis=1)
    csv_path = DEMO_PATHS.get_metrics_path('demo_results.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"CSV saved to {csv_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("DEMO EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Total results: {len(all_results)}")

    if arima_results:
        arima_mae = np.mean([r['metrics']['mae'] for r in arima_results])
        print(f"ARIMA average MAE: {arima_mae:.4f}")

    if stgnn_results:
        stgnn_mae = np.mean([r['metrics']['mae'] for r in stgnn_results])
        print(f"STGNN average MAE: {stgnn_mae:.4f}")

    print(f"\nResults saved to: {DEMO_PATHS.ckpt_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
