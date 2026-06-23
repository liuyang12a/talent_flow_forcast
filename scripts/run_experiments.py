#!/usr/bin/env python3
"""
Main experiment runner for ARIMA vs STGNN comparison.

This script orchestrates the complete experimental workflow:
1. Generate/select representative time series
2. Run ARIMA and STGNN models on each series
3. Collect and analyze results
4. Generate visualizations and reports

Usage:
    python scripts/run_experiments.py [--phase all|generate|analyze|visualize]
"""

import sys
import argparse
import logging
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.config import (
    DATA_CONFIG, SELECTOR_CONFIG, MODEL_CONFIG,
    PREDICTION_SETTINGS, EXPERIMENT_MATRIX, CKPT_DIR,
    DATA_DIR, SERIES_OUTPUT_DIR, RANDOM_SEED
)
from scripts.analysis import (
    SeriesAnalyzer, ModelComparator, ReportGenerator
)

from src.data import (
    FlowNetworkDataLoader, HighWeightSelector, HubNodeSelector, CommunitySelector,
    DenseSubgraphConfig, DenseSubgraphExtractor, BaseTensorBuilder,
)
from src.data.transforms import ZScoreScaler
from src.models.statistical import ARIMAModel
from src.models.deep_learning import STGNNModel
from src.utils.metrics import calculate_metrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(CKPT_DIR / 'experiment.log')
    ]
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)


class ExperimentRunner:
    """
    Main experiment runner class.

    Manages the complete experimental workflow from data preparation
to result analysis.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize the experiment runner.

        Args:
            config: Experiment configuration dictionary
        """
        self.config = config or {}
        self.series_data = {}  # {selector_type: {series_id: array}}
        self.series_characteristics = {}  # {selector_type: {series_id: dict}}
        self.results = []  # List of result dictionaries
        self.adjacency_matrices = {}  # {selector_type: adj_matrix}

    def phase1_generate_series(self) -> None:
        """
        Phase 1: Generate and save experiment time series.

        Loads flow networks, selects representative edges using different
strategies, and saves the time series data.
        """
        logger.info("=" * 60)
        logger.info("Phase 1: Generating Experiment Time Series")
        logger.info("=" * 60)

        # Load flow networks
        logger.info("Loading flow networks...")
        loader = FlowNetworkDataLoader(
            data_dir=str(DATA_DIR),
            start_date=DATA_CONFIG['date_range']['start'],
            end_date=DATA_CONFIG['date_range']['end']
        )
        networks = loader.load_networks()
        logger.info(f"Loaded {len(networks)} monthly networks")

        # Create scaler for normalization
        scaler = ZScoreScaler(axis=0)

        # Generate series for each selector type
        selectors = {
            'high_weight': (
                HighWeightSelector(**SELECTOR_CONFIG['high_weight']),
                SELECTOR_CONFIG['high_weight']['top_k']
            ),
            'hub_nodes': (
                HubNodeSelector(**{k: v for k, v in SELECTOR_CONFIG['hub_nodes'].items()
                                 if k != 'max_total'}),
                SELECTOR_CONFIG['hub_nodes']['max_total']
            ),
            'communities': (
                CommunitySelector(**{k: v for k, v in SELECTOR_CONFIG['communities'].items()
                                    if k != 'max_total'}),
                SELECTOR_CONFIG['communities']['max_total']
            )
        }

        for selector_name, (selector, max_edges) in selectors.items():
            logger.info(f"\nProcessing {selector_name} selector...")

            # Select edges
            edges = selector.select(networks)

            # Apply limit if needed
            if len(edges) > max_edges:
                logger.info(f"Limiting {selector_name} edges from {len(edges)} to {max_edges}")
                edges = edges[:max_edges]

            logger.info(f"Selected {len(edges)} edges for {selector_name}")

            if len(edges) == 0:
                logger.warning(f"No edges selected for {selector_name}, skipping")
                continue

            # Extract time series for these edges
            timestamps = sorted(networks.keys())
            series_matrix = np.zeros((len(timestamps), len(edges)))

            for t, timestamp in enumerate(timestamps):
                network = networks[timestamp]
                for e, (source, target) in enumerate(edges):
                    weight = network.get_edge_weight(source, target)
                    series_matrix[t, e] = weight

            # Store series data
            self.series_data[selector_name] = {
                'edges': edges,
                'timestamps': timestamps,
                'series_matrix': series_matrix,
                'series_dict': {f"edge_{i}": series_matrix[:, i] for i in range(len(edges))}
            }

            # Analyze characteristics
            logger.info(f"Analyzing series characteristics for {selector_name}...")
            analyzer = SeriesAnalyzer(period=12)
            characteristics = analyzer.analyze_series_collection(
                self.series_data[selector_name]['series_dict']
            )
            self.series_characteristics[selector_name] = characteristics

            # Build adjacency matrix for STGNN
            adj_matrix = self._build_adjacency_matrix(edges, networks)
            self.adjacency_matrices[selector_name] = adj_matrix

            # Save to disk
            self._save_series_data(selector_name)

        # ── dense_core: separate pipeline ──────────────────────────
        if 'dense_core' in SELECTOR_CONFIG:
            self._generate_dense_core_series(networks, scaler)

        logger.info("\nPhase 1 completed. Series data generated and saved.")

    def _generate_dense_core_series(
        self,
        networks: Dict,
        scaler: ZScoreScaler,
    ) -> None:
        """Generate series using the dense-core extraction pipeline.

        This is separate from the traditional selector loop because
        dense_core jointly optimises spatial and temporal density
        rather than simply ranking edges.
        """
        logger.info(f"\nProcessing dense_core selector...")

        cfg = SELECTOR_CONFIG['dense_core']
        selector_name = 'dense_core'

        # Build config and tensor builder from the selector config dict
        dense_config = DenseSubgraphConfig(
            spatial_strategy=cfg.get('spatial_strategy', 'flow_core'),
            max_nodes=cfg.get('max_nodes', 200),
            min_nodes=cfg.get('min_nodes', 20),
            target_coverage=cfg.get('target_coverage', 0.80),
            min_activity_ratio=cfg.get('min_activity_ratio', 0.30),
            max_allowed_gap=cfg.get('max_allowed_gap', 12),
            min_temporal_score=cfg.get('min_temporal_score', 0.25),
            tensor_type=cfg.get('tensor_type', 'edge_centric'),
            adj_type=cfg.get('adj_type', 'shared_node'),
            node_features=cfg.get('node_features'),
            exclude_self_loops=cfg.get('exclude_self_loops', True),
        )
        tensor_builder = BaseTensorBuilder.from_config(cfg)

        extractor = DenseSubgraphExtractor(dense_config, tensor_builder)
        result = extractor.extract(networks)

        if result.tensor is None or len(result.edges) == 0:
            logger.warning("dense_core produced no edges — skipping.")
            return

        timestamps = sorted(networks.keys())
        tensor_type = cfg.get('tensor_type', 'edge_centric')

        if tensor_type == 'node_centric':
            # tensor: [T, N, C]
            # Build series_dict: one 1-D series per (node, feature) pair
            N = result.tensor.shape[1]
            C = result.tensor.shape[2]
            features = cfg.get('node_features') or ['net_flow']
            series_dict = {}
            for n in range(N):
                for c in range(C):
                    feat_name = features[c] if c < len(features) else f"feat_{c}"
                    series_dict[f"node_{n}_{feat_name}"] = result.tensor[:, n, c]

            self.series_data[selector_name] = {
                'edges': list(result.nodes),       # recycled as "node list"
                'timestamps': timestamps,
                'series_matrix': result.tensor,     # [T, N, C]
                'series_dict': series_dict,
                'quality': result.quality,
                'metadata': result.metadata,
            }
        else:
            # edge_centric (default): tensor [T, E, 1]
            series_matrix_2d = result.tensor[:, :, 0]  # squeeze to [T, E]

            self.series_data[selector_name] = {
                'edges': result.edges,
                'timestamps': timestamps,
                'series_matrix': series_matrix_2d,
                'series_dict': {
                    f"edge_{i}": series_matrix_2d[:, i]
                    for i in range(len(result.edges))
                },
                'quality': result.quality,
                'metadata': result.metadata,
            }

        # Analyze characteristics (on the 1-D series)
        logger.info(f"Analyzing series characteristics for {selector_name}...")
        analyzer = SeriesAnalyzer(period=12)
        characteristics = analyzer.analyze_series_collection(
            self.series_data[selector_name]['series_dict']
        )
        self.series_characteristics[selector_name] = characteristics

        # Use the pre-built adjacency from the extractor
        self.adjacency_matrices[selector_name] = result.adjacency

        # Apply ZScore scaler to the 2-D matrix for downstream models
        if tensor_type == 'edge_centric':
            series_2d = self.series_data[selector_name]['series_matrix']  # [T, E]
            scaler.fit(series_2d)
            normalized = scaler.transform(series_2d)
            self.series_data[selector_name]['series_matrix'] = normalized
            # Update series_dict with normalized values
            self.series_data[selector_name]['series_dict'] = {
                f"edge_{i}": normalized[:, i]
                for i in range(normalized.shape[1])
            }

        # Save to disk
        self._save_series_data(selector_name)

        # Log quality report
        q = result.quality
        if q:
            logger.info(
                "dense_core quality: nodes=%d edges=%d density=%.4f "
                "coverage=%.2f%% mean_activity=%.3f median_activity=%.3f "
                "low_activity_pct=%.1f%% mean_max_gap=%.1f",
                q.node_count, q.edge_count, q.spatial_density,
                100 * q.flow_coverage, q.mean_activity, q.median_activity,
                100 * q.low_activity_pct, q.mean_max_gap,
            )

    def _build_adjacency_matrix(
        self,
        edges: List[Tuple],
        networks: Dict
    ) -> np.ndarray:
        """
        Build adjacency matrix for STGNN model.

        Args:
            edges: List of selected edges
            networks: Dictionary of flow networks

        Returns:
            Adjacency matrix
        """
        n = len(edges)
        adj = np.eye(n)  # Start with self-connections

        # Create adjacency based on shared nodes
        for i, (src_i, tgt_i) in enumerate(edges):
            for j, (src_j, tgt_j) in enumerate(edges):
                if i != j:
                    # Connect if they share source or target
                    if src_i == src_j or tgt_i == tgt_j or src_i == tgt_j or tgt_i == src_j:
                        adj[i, j] = 1.0

        return adj

    def _save_series_data(self, selector_name: str) -> None:
        """
        Save series data to disk.

        Args:
            selector_name: Name of the selector type
        """
        output_dir = SERIES_OUTPUT_DIR / selector_name
        output_dir.mkdir(parents=True, exist_ok=True)

        data = self.series_data[selector_name]

        # Save series matrix
        np.savez(
            output_dir / 'series_data.npz',
            series_matrix=data['series_matrix'],
            timestamps=data['timestamps'],
            edges=data['edges']
        )

        # Save characteristics
        from scripts.analysis.series_characteristics import _convert_to_serializable
        with open(output_dir / 'characteristics.json', 'w') as f:
            json.dump(_convert_to_serializable(self.series_characteristics[selector_name]), f, indent=2)

        # Save adjacency matrix
        np.save(output_dir / 'adjacency_matrix.npy', self.adjacency_matrices[selector_name])

        # Save quality metrics (dense_core only)
        if 'quality' in data and data['quality'] is not None:
            from dataclasses import asdict
            qual_dict = asdict(data['quality'])
            with open(output_dir / 'quality_metrics.json', 'w') as f:
                json.dump(qual_dict, f, indent=2)

        # Save metadata (dense_core only)
        if 'metadata' in data and data['metadata']:
            with open(output_dir / 'metadata.json', 'w') as f:
                json.dump(data['metadata'], f, indent=2)

        logger.info(f"Saved {selector_name} data to {output_dir}")

    def _load_series_data(self, selector_name: str) -> bool:
        """
        Load series data from disk.

        Args:
            selector_name: Name of the selector type

        Returns:
            True if loaded successfully
        """
        input_dir = SERIES_OUTPUT_DIR / selector_name

        if not input_dir.exists():
            return False

        try:
            # Load series data
            data = np.load(input_dir / 'series_data.npz', allow_pickle=True)
            self.series_data[selector_name] = {
                'series_matrix': data['series_matrix'],
                'timestamps': data['timestamps'].tolist(),
                'edges': data['edges'].tolist(),
                'series_dict': {f"edge_{i}": data['series_matrix'][:, i]
                               for i in range(data['series_matrix'].shape[1])}
            }

            # Load characteristics
            with open(input_dir / 'characteristics.json', 'r') as f:
                self.series_characteristics[selector_name] = json.load(f)

            # Load adjacency matrix
            self.adjacency_matrices[selector_name] = np.load(input_dir / 'adjacency_matrix.npy')

            return True

        except Exception as e:
            logger.warning(f"Failed to load {selector_name} data: {e}")
            return False

    def phase2_run_experiments(self) -> None:
        """
        Phase 2: Run ARIMA and STGNN experiments.

        Runs both models on all generated time series and collects results.
        """
        logger.info("=" * 60)
        logger.info("Phase 2: Running Experiments")
        logger.info("=" * 60)

        # Load series data if not already loaded
        if not self.series_data:
            for selector_name in ['high_weight', 'hub_nodes', 'communities', 'dense_core']:
                if self._load_series_data(selector_name):
                    logger.info(f"Loaded {selector_name} series data from disk")

        if not self.series_data:
            logger.error("No series data available. Run phase 1 first.")
            return

        # Run experiments for each selector type
        for selector_name, data in self.series_data.items():
            logger.info(f"\nRunning experiments for {selector_name}...")

            edges = data['edges']
            series_matrix = data['series_matrix']

            # Run experiments with different settings
            for setting in PREDICTION_SETTINGS[:1]:  # Use first setting for now
                input_len = setting['input_len']
                output_len = setting['output_len']

                logger.info(f"  Settings: input_len={input_len}, output_len={output_len}")

                # Run ARIMA experiments
                self._run_arima_experiments(
                    selector_name, edges, series_matrix,
                    input_len, output_len
                )

                # Run STGNN experiments
                self._run_stgnn_experiments(
                    selector_name, edges, series_matrix,
                    input_len, output_len
                )

        # Save results
        self._save_results()

        logger.info("\nPhase 2 completed. Experiments finished.")

    def _run_arima_experiments(
        self,
        selector_name: str,
        edges: List[Tuple],
        series_matrix: np.ndarray,
        input_len: int,
        output_len: int
    ) -> None:
        """
        Run ARIMA experiments for a set of series.

        Args:
            selector_name: Name of the selector type
            edges: List of edges (or nodes for node_centric tensors)
            series_matrix: Matrix of time series [T, E] or [T, N, C]
            input_len: Input sequence length
            output_len: Output sequence length
        """
        logger.info(f"  Running ARIMA experiments...")

        # If node_centric [T, N, C], flatten to [T, N*C]
        is_node_centric = series_matrix.ndim == 3
        if is_node_centric:
            N, C = series_matrix.shape[1], series_matrix.shape[2]
            series_2d = series_matrix.reshape(series_matrix.shape[0], N * C)
            logger.info(f"  Flattened node_centric [T,{N},{C}] → [T,{N*C}]")
        else:
            series_2d = series_matrix

        # Split data
        n_samples = series_2d.shape[0]
        train_size = int(n_samples * DATA_CONFIG['train_ratio'])
        val_size = int(n_samples * DATA_CONFIG['val_ratio'])

        train_data = series_2d[:train_size]
        val_data = series_2d[train_size:train_size + val_size]
        test_data = series_2d[train_size + val_size:]

        if len(test_data) < input_len + output_len:
            logger.warning(f"  Not enough test data for ARIMA")
            return

        # Train and evaluate for each series
        for i in range(series_2d.shape[1]):
            series_id = f"{selector_name}_edge_{i}"
            series_train = train_data[:, i]
            series_test = test_data[:, i]

            # Skip if too many zeros
            if np.sum(series_train > 0) < 10:
                continue

            # Prepare sliding windows
            X_train, y_train = self._create_windows(series_train, input_len, output_len)
            X_test, y_test = self._create_windows(series_test, input_len, output_len)

            if len(X_train) < 5 or len(X_test) < 1:
                continue

            # Try different ARIMA orders
            best_mae = float('inf')
            best_result = None

            for order in MODEL_CONFIG['arima']['orders']:
                try:
                    model = ARIMAModel(
                        input_len=input_len,
                        output_len=output_len,
                        order=order,
                        name=f"ARIMA_{selector_name}_{i}"
                    )
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)

                    metrics = calculate_metrics(y_test.flatten(), predictions.flatten())

                    if metrics['mae'] < best_mae:
                        best_mae = metrics['mae']
                        edge_label = (edges[i] if i < len(edges)
                                      else str(i))
                        best_result = {
                            'series_id': series_id,
                            'model_type': 'arima',
                            'selector_type': selector_name,
                            'edge_idx': i,
                            'edge': edge_label,
                            'input_len': input_len,
                            'output_len': output_len,
                            'config': {'order': order},
                            'metrics': metrics
                        }

                        # Add characteristics
                        if selector_name in self.series_characteristics:
                            char = self.series_characteristics[selector_name].get(series_id, {})
                            if 'classification' in char:
                                for key, value in char['classification'].items():
                                    best_result[f'char_{key}'] = value

                except Exception as e:
                    logger.debug(f"  ARIMA order {order} failed for series {i}: {e}")
                    continue

            if best_result:
                self.results.append(best_result)

        logger.info(f"  ARIMA: Completed {len([r for r in self.results if r['model_type'] == 'arima' and r['selector_type'] == selector_name])} series")

    def _run_stgnn_experiments(
        self,
        selector_name: str,
        edges: List[Tuple],
        series_matrix: np.ndarray,
        input_len: int,
        output_len: int
    ) -> None:
        """
        Run STGNN experiments for a set of series.

        Args:
            selector_name: Name of the selector type
            edges: List of edges
            series_matrix: Matrix of time series
            input_len: Input sequence length
            output_len: Output sequence length
        """
        logger.info(f"  Running STGNN experiments...")

        # If node_centric [T, N, C], flatten to [T, N*C] for uniform handling
        is_node_centric = series_matrix.ndim == 3
        if is_node_centric:
            N, C = series_matrix.shape[1], series_matrix.shape[2]
            series_2d = series_matrix.reshape(series_matrix.shape[0], N * C)
            effective_edges = [
                f"node_{n}_feat_{c}" for n in range(N) for c in range(C)
            ]
            # Adjacency: tile the N×N matrix to (N*C)×(N*C) by blocking
            adj_raw = self.adjacency_matrices.get(selector_name)
            if adj_raw is not None:
                adj_raw = adj_raw[:N, :N]
                adj_big = np.kron(adj_raw, np.ones((C, C), dtype=np.float32))
            else:
                adj_big = None
            logger.info(f"  Flattened node_centric [T,{N},{C}] → [T,{N*C}]")
        else:
            series_2d = series_matrix
            effective_edges = edges
            adj_raw = self.adjacency_matrices.get(selector_name)

        # Split data
        n_samples = series_2d.shape[0]
        train_size = int(n_samples * DATA_CONFIG['train_ratio'])
        val_size = int(n_samples * DATA_CONFIG['val_ratio'])

        train_data = series_2d[:train_size]
        val_data = series_2d[train_size:train_size + val_size]
        test_data = series_2d[train_size + val_size:]

        # Prepare data for STGNN: [samples, time, nodes, features]
        X_train, y_train = self._prepare_stgnn_data(train_data, val_data, input_len, output_len)
        X_val, y_val = self._prepare_stgnn_data(val_data, test_data[:len(val_data)], input_len, output_len)
        X_test, y_test = self._prepare_stgnn_data(test_data[:-output_len], test_data[input_len:], input_len, output_len)

        if X_train is None or len(X_train) < 5:
            logger.warning(f"  Not enough data for STGNN")
            return

        num_series = series_2d.shape[1]

        # Get adjacency matrix
        if is_node_centric and adj_big is not None:
            adj_matrix = adj_big[:num_series, :num_series]
        elif adj_raw is not None:
            adj_matrix = adj_raw[:num_series, :num_series]
        else:
            adj_matrix = np.eye(num_series)

        # Train STGNN
        try:
            model = STGNNModel(
                input_len=input_len,
                output_len=output_len,
                num_nodes=num_series,
                adjacency_matrix=adj_matrix,
                input_dim=1,
                hidden_dim=32,  # Use smaller hidden dim for faster training
                num_layers=2,
                spatial_type='gcn',
                temporal_type='gru',
                device='auto'
            )

            model.fit(
                X_train, y_train,
                X_val=X_val, y_val=y_val,
                epochs=50,  # Reduced epochs for faster execution
                batch_size=8,
                learning_rate=0.001,
                early_stopping_patience=10
            )

            predictions = model.predict(X_test, batch_size=8)

            # Evaluate per series
            for i in range(num_series):
                series_id = f"{selector_name}_edge_{i}"

                pred_series = predictions[:, :, i, 0].flatten()
                target_series = y_test[:, :, i, 0].flatten()

                metrics = calculate_metrics(target_series, pred_series)

                edge_label = (effective_edges[i] if i < len(effective_edges)
                              else str(i))
                result = {
                    'series_id': series_id,
                    'model_type': 'stgnn',
                    'selector_type': selector_name,
                    'edge_idx': i,
                    'edge': edge_label,
                    'input_len': input_len,
                    'output_len': output_len,
                    'config': {'hidden_dim': 32, 'num_layers': 2, 'spatial': 'gcn', 'temporal': 'gru'},
                    'metrics': metrics
                }

                # Add characteristics
                if selector_name in self.series_characteristics:
                    char = self.series_characteristics[selector_name].get(series_id, {})
                    if 'classification' in char:
                        for key, value in char['classification'].items():
                            result[f'char_{key}'] = value

                self.results.append(result)

            logger.info(f"  STGNN: Completed {num_series} series")

        except Exception as e:
            logger.error(f"  STGNN experiment failed: {e}")
            import traceback
            traceback.print_exc()

    def _create_windows(
        self,
        series: np.ndarray,
        input_len: int,
        output_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows from a time series.

        Args:
            series: Time series array
            input_len: Input sequence length
            output_len: Output sequence length

        Returns:
            Tuple of (X, y) arrays
        """
        X, y = [], []
        for t in range(len(series) - input_len - output_len + 1):
            X.append(series[t:t + input_len])
            y.append(series[t + input_len:t + input_len + output_len])
        return np.array(X), np.array(y)

    def _prepare_stgnn_data(
        self,
        input_data: np.ndarray,
        target_data: np.ndarray,
        input_len: int,
        output_len: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Prepare data for STGNN model.

        Args:
            input_data: Input time series data
            target_data: Target time series data
            input_len: Input sequence length
            output_len: Output sequence length

        Returns:
            Tuple of (X, y) arrays in STGNN format
        """
        if len(input_data) < input_len + output_len:
            return None, None

        X_list, y_list = [], []

        for t in range(len(input_data) - input_len - output_len + 1):
            # X: [input_len, num_nodes, 1]
            X = input_data[t:t + input_len, :, np.newaxis]
            # y: [output_len, num_nodes, 1]
            y_end = t + input_len + output_len
            if y_end <= len(target_data):
                y = target_data[t + input_len:y_end, :, np.newaxis]
                X_list.append(X)
                y_list.append(y)

        if len(X_list) == 0:
            return None, None

        return np.array(X_list), np.array(y_list)

    def _save_results(self) -> None:
        """Save experiment results to disk."""
        if not self.results:
            logger.warning("No results to save")
            return

        # Save as JSON
        results_path = CKPT_DIR / 'metrics' / 'experiment_results.json'
        results_path.parent.mkdir(parents=True, exist_ok=True)

        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        # Save as CSV
        df = pd.DataFrame(self.results)

        # Expand metrics dictionary into columns
        if 'metrics' in df.columns:
            metrics_df = pd.json_normalize(df['metrics'])
            metrics_df.columns = [f'metric_{c}' for c in metrics_df.columns]
            df = pd.concat([df.drop('metrics', axis=1), metrics_df], axis=1)

        csv_path = CKPT_DIR / 'metrics' / 'experiment_results.csv'
        df.to_csv(csv_path, index=False)

        logger.info(f"Saved {len(self.results)} results to {results_path} and {csv_path}")

    def phase3_analyze_results(self) -> None:
        """
        Phase 3: Analyze experiment results.

        Generates comparison tables, statistical tests, and analysis reports.
        """
        logger.info("=" * 60)
        logger.info("Phase 3: Analyzing Results")
        logger.info("=" * 60)

        # Load results if not already loaded
        if not self.results:
            results_path = CKPT_DIR / 'metrics' / 'experiment_results.json'
            if results_path.exists():
                with open(results_path, 'r') as f:
                    self.results = json.load(f)
                logger.info(f"Loaded {len(self.results)} results from {results_path}")
            else:
                logger.error("No results found. Run phase 2 first.")
                return

        # Generate reports
        csv_path = CKPT_DIR / 'metrics' / 'experiment_results.csv'
        if csv_path.exists():
            from scripts.analysis import generate_experiment_report

            reports = generate_experiment_report(
                csv_path,
                CKPT_DIR,
                formats=['json', 'markdown', 'csv']
            )

            logger.info("Generated reports:")
            for fmt, path in reports.items():
                logger.info(f"  {fmt}: {path}")

        logger.info("\nPhase 3 completed. Analysis finished.")

    def phase4_visualize(self) -> None:
        """
        Phase 4: Generate visualizations.

        Creates plots and charts for result presentation.
        """
        logger.info("=" * 60)
        logger.info("Phase 4: Generating Visualizations")
        logger.info("=" * 60)

        csv_path = CKPT_DIR / 'metrics' / 'experiment_results.csv'

        if not csv_path.exists():
            logger.error("No results CSV found. Run phase 2 and 3 first.")
            return

        # Load results
        results_df = pd.read_csv(csv_path)

        # Load characteristics
        all_characteristics = {}
        for selector_name in ['high_weight', 'hub_nodes', 'communities', 'dense_core']:
            char_path = SERIES_OUTPUT_DIR / selector_name / 'characteristics.json'
            if char_path.exists():
                with open(char_path, 'r') as f:
                    all_characteristics.update(json.load(f))

        # Generate visualizations
        plots_dir = CKPT_DIR / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)

        try:
            from scripts.analysis.visualizations import (
                plot_model_comparison,
                plot_series_characteristics,
                plot_predictability_analysis
            )

            # Model comparison
            fig1 = plot_model_comparison(
                results_df, metric='mae', group_by='selector_type',
                output_path=plots_dir / 'model_comparison_mae.png'
            )
            plt.close(fig1)

            fig2 = plot_model_comparison(
                results_df, metric='mape', group_by='selector_type',
                output_path=plots_dir / 'model_comparison_mape.png'
            )
            plt.close(fig2)

            # Series characteristics
            if all_characteristics:
                fig3 = plot_series_characteristics(
                    all_characteristics,
                    output_path=plots_dir / 'series_characteristics.png'
                )
                plt.close(fig3)

                # Predictability analysis
                fig4 = plot_predictability_analysis(
                    results_df, all_characteristics,
                    output_path=plots_dir / 'predictability_analysis.png'
                )
                plt.close(fig4)

            logger.info(f"Saved visualizations to {plots_dir}")

        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            import traceback
            traceback.print_exc()

        logger.info("\nPhase 4 completed. Visualizations generated.")

    def run_all_phases(self) -> None:
        """Run all experimental phases."""
        self.phase1_generate_series()
        self.phase2_run_experiments()
        self.phase3_analyze_results()
        self.phase4_visualize()

        logger.info("=" * 60)
        logger.info("All phases completed successfully!")
        logger.info(f"Results saved to: {CKPT_DIR}")
        logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run ARIMA vs STGNN comparison experiments'
    )
    parser.add_argument(
        '--phase',
        choices=['all', 'generate', 'run', 'analyze', 'visualize'],
        default='all',
        help='Which phase to run (default: all)'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom configuration file'
    )

    args = parser.parse_args()

    # Load custom config if provided
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)

    # Initialize runner
    runner = ExperimentRunner(config)

    # Run requested phase
    if args.phase == 'all':
        runner.run_all_phases()
    elif args.phase == 'generate':
        runner.phase1_generate_series()
    elif args.phase == 'run':
        runner.phase2_run_experiments()
    elif args.phase == 'analyze':
        runner.phase3_analyze_results()
    elif args.phase == 'visualize':
        runner.phase4_visualize()


if __name__ == '__main__':
    main()
