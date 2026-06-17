#!/usr/bin/env python3
"""
Run STGNN experiment only.
"""

import sys
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.config import DATA_CONFIG, MODEL_CONFIG, CKPT_DIR, SERIES_OUTPUT_DIR, RANDOM_SEED
from src.models.deep_learning import STGNNModel
from src.utils.metrics import calculate_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
np.random.seed(RANDOM_SEED)

def load_series_data():
    input_dir = SERIES_OUTPUT_DIR / 'high_weight'
    data = np.load(input_dir / 'series_data.npz', allow_pickle=True)
    adj_matrix = np.load(input_dir / 'adjacency_matrix.npy')
    with open(input_dir / 'characteristics.json', 'r') as f:
        characteristics = json.load(f)
    return data['series_matrix'], data['edges'].tolist(), adj_matrix, characteristics

def prepare_stgnn_data(input_data, target_data, input_len, output_len):
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

def main():
    logger.info("Loading series data...")
    series_matrix, edges, adj_matrix, characteristics = load_series_data()
    logger.info(f"Loaded {len(edges)} edges, {series_matrix.shape[0]} timestamps")

    # Limit for faster execution
    max_nodes = 30
    n_nodes = min(max_nodes, len(edges))
    series_matrix = series_matrix[:, :n_nodes]
    adj_matrix = adj_matrix[:n_nodes, :n_nodes]
    logger.info(f"Using {n_nodes} nodes for STGNN")

    input_len, output_len = 12, 1

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
    X_test, y_test = prepare_stgnn_data(test_data[:-output_len], test_data[input_len:], input_len, output_len)

    if X_train is None or len(X_train) < 5:
        logger.error("Not enough data")
        return

    logger.info(f"Data shapes: X_train={X_train.shape}, y_train={y_train.shape}")

    # Train STGNN
    logger.info("Training STGNN...")
    try:
        model = STGNNModel(
            input_len=input_len, output_len=output_len, num_nodes=n_nodes,
            adjacency_matrix=adj_matrix, input_dim=1, hidden_dim=32, num_layers=2,
            spatial_type='gcn', temporal_type='gru', device='auto'
        )

        model.fit(X_train, y_train, X_val=X_val, y_val=y_val,
                  epochs=50, batch_size=8, learning_rate=0.001, early_stopping_patience=10)

        logger.info("Generating predictions...")
        predictions = model.predict(X_test, batch_size=8)

        # Evaluate per series
        results = []
        for i in range(n_nodes):
            series_id = f"high_weight_edge_{i}"
            pred_series = predictions[:, :, i, 0].flatten()
            target_series = y_test[:, :, i, 0].flatten()
            metrics = calculate_metrics(target_series, pred_series)

            result = {
                'series_id': series_id, 'model_type': 'stgnn',
                'selector_type': 'high_weight', 'edge_idx': i,
                'input_len': input_len, 'output_len': output_len,
                'metrics': metrics
            }

            char = characteristics.get(series_id, {})
            if 'classification' in char:
                for key, value in char['classification'].items():
                    result[f'char_{key}'] = value

            results.append(result)

        logger.info(f"STGNN completed {len(results)} series")

        # Load existing ARIMA results
        arima_path = CKPT_DIR / 'metrics' / 'experiment_results.json'
        if arima_path.exists():
            with open(arima_path, 'r') as f:
                arima_results = json.load(f)
            # Keep only ARIMA results (first 30)
            arima_results = [r for r in arima_results if r['model_type'] == 'arima'][:30]
            all_results = arima_results + results
        else:
            all_results = results

        # Save combined results
        results_path = CKPT_DIR / 'metrics' / 'experiment_results.json'
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        df = pd.DataFrame(all_results)
        if 'metrics' in df.columns:
            metrics_df = pd.json_normalize(df['metrics'])
            metrics_df.columns = [f'metric_{c}' for c in metrics_df.columns]
            df = pd.concat([df.drop('metrics', axis=1), metrics_df], axis=1)
        df.to_csv(CKPT_DIR / 'metrics' / 'experiment_results.csv', index=False)

        logger.info(f"Saved {len(all_results)} total results")

    except Exception as e:
        logger.error(f"STGNN failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
