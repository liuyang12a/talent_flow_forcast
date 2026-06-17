#!/usr/bin/env python3
"""
Quick experiment runner using pre-generated high_weight series data.

This script runs ARIMA and STGNN experiments on the high_weight series
without regenerating the data.
"""

import sys
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.config import (
    DATA_CONFIG, MODEL_CONFIG, CKPT_DIR, SERIES_OUTPUT_DIR, RANDOM_SEED
)
from scripts.analysis import (
    SeriesAnalyzer, ModelComparator, ReportGenerator, generate_experiment_report
)

from src.models.statistical import ARIMAModel
from src.models.deep_learning import STGNNModel
from src.utils.metrics import calculate_metrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(CKPT_DIR / 'quick_experiment.log')
    ]
)
logger = logging.getLogger(__name__)

# Set random seed
np.random.seed(RANDOM_SEED)


def load_series_data(selector_name: str = 'high_weight'):
    """Load pre-generated series data."""
    input_dir = SERIES_OUTPUT_DIR / selector_name

    # Load series data
    data = np.load(input_dir / 'series_data.npz', allow_pickle=True)
    series_data = {
        'series_matrix': data['series_matrix'],
        'timestamps': data['timestamps'].tolist(),
        'edges': data['edges'].tolist(),
    }

    # Load characteristics
    with open(input_dir / 'characteristics.json', 'r') as f:
        characteristics = json.load(f)

    # Load adjacency matrix
    adj_matrix = np.load(input_dir / 'adjacency_matrix.npy')

    return series_data, characteristics, adj_matrix


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


def run_arima_experiments(series_data, characteristics, input_len=12, output_len=1, max_series=50):
    """Run ARIMA experiments on series."""
    logger.info(f"Running ARIMA experiments (max {max_series} series)...")

    edges = series_data['edges']
    series_matrix = series_data['series_matrix']

    # Split data
    n_samples = series_matrix.shape[0]
    train_size = int(n_samples * DATA_CONFIG['train_ratio'])
    val_size = int(n_samples * DATA_CONFIG['val_ratio'])

    train_data = series_matrix[:train_size]
    test_data = series_matrix[train_size + val_size:]

    results = []

    # Limit number of series for faster execution
    n_series = min(max_series, len(edges))

    for i in range(n_series):
        series_id = f"high_weight_edge_{i}"
        series_train = train_data[:, i]
        series_test = test_data[:, i]

        # Skip if too many zeros
        if np.sum(series_train > 0) < 10:
            continue

        # Prepare windows
        X_train, y_train = create_windows(series_train, input_len, output_len)
        X_test, y_test = create_windows(series_test, input_len, output_len)

        if len(X_train) < 5 or len(X_test) < 1:
            continue

        # Try different ARIMA orders
        best_mae = float('inf')
        best_metrics = None

        for order in MODEL_CONFIG['arima']['orders']:
            try:
                model = ARIMAModel(
                    input_len=input_len,
                    output_len=output_len,
                    order=order,
                    name=f"ARIMA_{i}"
                )
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                metrics = calculate_metrics(y_test.flatten(), predictions.flatten())

                if metrics['mae'] < best_mae:
                    best_mae = metrics['mae']
                    best_metrics = metrics

            except Exception as e:
                logger.debug(f"ARIMA order {order} failed for series {i}: {e}")
                continue

        if best_metrics:
            result = {
                'series_id': series_id,
                'model_type': 'arima',
                'selector_type': 'high_weight',
                'edge_idx': i,
                'input_len': input_len,
                'output_len': output_len,
                'metrics': best_metrics
            }

            # Add characteristics
            char = characteristics.get(series_id, {})
            if 'classification' in char:
                for key, value in char['classification'].items():
                    result[f'char_{key}'] = value

            results.append(result)

        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i + 1}/{n_series} series")

    logger.info(f"ARIMA: Completed {len(results)} series")
    return results


def run_stgnn_experiments(series_data, characteristics, adj_matrix,
                          input_len=12, output_len=1, max_series=50):
    """Run STGNN experiments on series."""
    logger.info(f"Running STGNN experiments (max {max_series} nodes)...")

    edges = series_data['edges']
    series_matrix = series_data['series_matrix']

    # Limit nodes for STGNN
    n_nodes = min(max_series, len(edges))
    edges = edges[:n_nodes]
    series_matrix = series_matrix[:, :n_nodes]
    adj_matrix = adj_matrix[:n_nodes, :n_nodes]

    # Split data
    n_samples = series_matrix.shape[0]
    train_size = int(n_samples * DATA_CONFIG['train_ratio'])
    val_size = int(n_samples * DATA_CONFIG['val_ratio'])

    train_data = series_matrix[:train_size]
    val_data = series_matrix[train_size:train_size + val_size]
    test_data = series_matrix[train_size + val_size:]

    # Prepare data
    X_train, y_train = prepare_stgnn_data(train_data, train_data, input_len, output_len)
    X_val, y_val = prepare_stgnn_data(val_data, val_data, input_len, output_len)

    test_input = test_data[:-output_len]
    test_target = test_data[input_len:]
    X_test, y_test = prepare_stgnn_data(test_input, test_target, input_len, output_len)

    if X_train is None or len(X_train) < 5:
        logger.warning("Not enough data for STGNN")
        return []

    results = []

    try:
        logger.info("Training STGNN model...")
        model = STGNNModel(
            input_len=input_len,
            output_len=output_len,
            num_nodes=n_nodes,
            adjacency_matrix=adj_matrix,
            input_dim=1,
            hidden_dim=32,
            num_layers=2,
            spatial_type='gcn',
            temporal_type='gru',
            device='auto'
        )

        model.fit(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            epochs=50,
            batch_size=8,
            learning_rate=0.001,
            early_stopping_patience=10
        )

        logger.info("Generating predictions...")
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

            # Add characteristics
            char = characteristics.get(series_id, {})
            if 'classification' in char:
                for key, value in char['classification'].items():
                    result[f'char_{key}'] = value

            results.append(result)

        logger.info(f"STGNN: Completed {len(results)} series")

    except Exception as e:
        logger.error(f"STGNN experiment failed: {e}")
        import traceback
        traceback.print_exc()

    return results


def save_results(results):
    """Save experiment results."""
    if not results:
        logger.warning("No results to save")
        return

    # Save as JSON
    results_path = CKPT_DIR / 'metrics' / 'experiment_results.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save as CSV
    df = pd.DataFrame(results)

    # Expand metrics dictionary into columns
    if 'metrics' in df.columns:
        metrics_df = pd.json_normalize(df['metrics'])
        metrics_df.columns = [f'metric_{c}' for c in metrics_df.columns]
        df = pd.concat([df.drop('metrics', axis=1), metrics_df], axis=1)

    csv_path = CKPT_DIR / 'metrics' / 'experiment_results.csv'
    df.to_csv(csv_path, index=False)

    logger.info(f"Saved {len(results)} results to {results_path} and {csv_path}")


def analyze_and_visualize():
    """Generate analysis and visualizations."""
    logger.info("Generating analysis and visualizations...")

    csv_path = CKPT_DIR / 'metrics' / 'experiment_results.csv'
    if not csv_path.exists():
        logger.warning("No results CSV found")
        return

    # Generate reports
    from scripts.analysis import generate_experiment_report

    reports = generate_experiment_report(
        csv_path,
        CKPT_DIR,
        formats=['json', 'markdown', 'csv']
    )

    logger.info("Generated reports:")
    for fmt, path in reports.items():
        logger.info(f"  {fmt}: {path}")

    # Generate visualizations
    try:
        from scripts.analysis.visualizations import (
            plot_model_comparison, plot_predictability_analysis
        )

        results_df = pd.read_csv(csv_path)

        # Load characteristics
        char_path = SERIES_OUTPUT_DIR / 'high_weight' / 'characteristics.json'
        with open(char_path, 'r') as f:
            characteristics = json.load(f)

        plots_dir = CKPT_DIR / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Model comparison
        fig1 = plot_model_comparison(
            results_df, metric='mae',
            output_path=plots_dir / 'model_comparison_mae.png'
        )
        plt.close(fig1)

        fig2 = plot_model_comparison(
            results_df, metric='mape',
            output_path=plots_dir / 'model_comparison_mape.png'
        )
        plt.close(fig2)

        # Predictability analysis
        fig3 = plot_predictability_analysis(
            results_df, characteristics,
            output_path=plots_dir / 'predictability_analysis.png'
        )
        plt.close(fig3)

        logger.info(f"Saved visualizations to {plots_dir}")

    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Quick Experiment Runner")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading series data...")
    series_data, characteristics, adj_matrix = load_series_data('high_weight')
    logger.info(f"Loaded {len(series_data['edges'])} edges, {len(series_data['timestamps'])} timestamps")

    # Run experiments
    all_results = []

    # ARIMA experiments
    arima_results = run_arima_experiments(series_data, characteristics,
                                          input_len=12, output_len=1, max_series=50)
    all_results.extend(arima_results)

    # STGNN experiments
    stgnn_results = run_stgnn_experiments(series_data, characteristics, adj_matrix,
                                          input_len=12, output_len=1, max_series=50)
    all_results.extend(stgnn_results)

    # Save results
    save_results(all_results)

    # Analyze and visualize
    analyze_and_visualize()

    logger.info("=" * 60)
    logger.info("Experiment completed!")
    logger.info(f"Results saved to: {CKPT_DIR}")
    logger.info("=" * 60)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    main()
