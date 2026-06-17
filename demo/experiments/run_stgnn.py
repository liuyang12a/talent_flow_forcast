#!/usr/bin/env python3
"""
Experiment script for STGNN model on talent flow data.

This script demonstrates how to use the new data module and STGNN model
to forecast talent flows using spatial-temporal graph neural networks.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import numpy as np

from demo.data import FlowNetworkDataLoader, HighWeightSelector
from demo.data.transforms import ZScoreScaler
from demo.models.deep_learning import STGNNModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_adjacency_matrix(edges, networks):
    """
    Build adjacency matrix for the selected edges.

    For flow networks, we create a fully connected graph where
    edges represent the flow relationships.
    """
    n = len(edges)
    adj = np.eye(n)  # Start with self-connections

    # Aggregate all edges across time to determine connectivity
    edge_set = set(edges)
    all_connections = {}

    for network in networks.values():
        for (src, tgt), weight in network.get_edges().items():
            if (src, tgt) in edge_set:
                idx = edges.index((src, tgt))
                all_connections[idx] = all_connections.get(idx, 0) + weight

    # Create adjacency based on shared nodes
    for i, (src_i, tgt_i) in enumerate(edges):
        for j, (src_j, tgt_j) in enumerate(edges):
            if i != j:
                # Connect if they share source or target
                if src_i == src_j or tgt_i == tgt_j or src_i == tgt_j or tgt_i == src_j:
                    adj[i, j] = 1.0

    return adj


def main():
    """Run STGNN experiment."""
    print("=" * 60)
    print("STGNN Experiment on Talent Flow Data")
    print("=" * 60)

    # Configuration
    DATA_DIR = "datasets/flow_networks"
    START_DATE = "2017-01"
    END_DATE = "2020-12"
    INPUT_LEN = 12
    OUTPUT_LEN = 3
    TOP_K_EDGES = 30  # Smaller for STGNN due to memory constraints
    HIDDEN_DIM = 32
    NUM_LAYERS = 2
    EPOCHS = 100
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001

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
    selector = HighWeightSelector(top_k=TOP_K_EDGES, min_months=12)
    edges = selector.select(networks)
    logger.info(f"Selected {len(edges)} edges")

    # Build adjacency matrix
    logger.info("Building adjacency matrix...")
    adj_matrix = build_adjacency_matrix(edges, networks)
    logger.info(f"Adjacency matrix shape: {adj_matrix.shape}")

    # Create datasets
    logger.info("Creating datasets...")
    scaler = ZScoreScaler(axis=0)  # Normalize per edge
    train_ds, val_ds, test_ds = loader.create_datasets(
        edges=edges,
        input_len=INPUT_LEN,
        output_len=OUTPUT_LEN,
        scaler=scaler,
        overlap=True
    )
    logger.info(f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Prepare data for STGNN
    # STGNN expects [batch, time, nodes, features]
    def prepare_data(dataset):
        X_list, y_list = [], []
        for i in range(len(dataset)):
            sample = dataset[i]
            # Reshape: [time, nodes] -> [time, nodes, 1]
            X = sample['inputs'][:, :, np.newaxis]  # [input_len, nodes, 1]
            y = sample['target'][:, :, np.newaxis]  # [output_len, nodes, 1]
            X_list.append(X)
            y_list.append(y)
        return np.array(X_list), np.array(y_list)

    logger.info("Preparing training data...")
    X_train, y_train = prepare_data(train_ds)
    X_val, y_val = prepare_data(val_ds)
    X_test, y_test = prepare_data(test_ds)

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"y_train shape: {y_train.shape}")

    # Create and train STGNN
    logger.info("Creating STGNN model...")
    model = STGNNModel(
        input_len=INPUT_LEN,
        output_len=OUTPUT_LEN,
        num_nodes=len(edges),
        adjacency_matrix=adj_matrix,
        input_dim=1,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        spatial_type="gcn",
        temporal_type="gru",
        output_type="direct",
        dropout=0.1,
        device="auto"
    )

    logger.info("Training STGNN model...")
    model.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        early_stopping_patience=20
    )

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    predictions = model.predict(X_test, batch_size=BATCH_SIZE)

    # Flatten for metric calculation
    pred_flat = predictions.flatten()
    target_flat = y_test.flatten()

    from demo.utils.metrics import calculate_metrics
    metrics = calculate_metrics(target_flat, pred_flat)

    print("\n" + "=" * 60)
    print("STGNN Experiment Results")
    print("=" * 60)
    print(f"Number of edges: {len(edges)}")
    print(f"Hidden dimension: {HIDDEN_DIM}")
    print(f"Number of layers: {NUM_LAYERS}")
    print(f"Test samples: {len(pred_flat)}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"R²: {metrics['r2']:.4f}")
    print("=" * 60)

    # Save results
    output_dir = Path("demo/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / "stgnn_results.npz",
        predictions=predictions,
        targets=y_test,
        adjacency=adj_matrix,
        edges=edges,
        metrics=metrics
    )
    logger.info(f"Results saved to {output_dir / 'stgnn_results.npz'}")

    # Save model
    model.save(output_dir / "stgnn_model.pt")
    logger.info(f"Model saved to {output_dir / 'stgnn_model.pt'}")


if __name__ == "__main__":
    main()
