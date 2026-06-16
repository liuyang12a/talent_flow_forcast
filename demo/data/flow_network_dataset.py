"""
FlowNetwork-specific dataset implementation.

This module provides dataset classes for loading and processing
talent flow network data from monthly pickle files.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
from pathlib import Path
import pickle
import logging
import numpy as np

from demo.data.base_dataset import BaseDataset, DatasetConfig

logger = logging.getLogger(__name__)


class FlowNetworkDataset(BaseDataset):
    """
    Dataset for FlowNetwork time series data.

    This dataset loads monthly flow networks and constructs time series
    for specified edges (company pairs).

    Attributes:
        data_dir: Directory containing monthly .pkl files
        edges: List of (source, target) edges to model
        start_date: Start date in YYYY-MM format
        end_date: End date in YYYY-MM format
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        edges: List[Tuple[Union[int, str], Union[int, str]]],
        config: Union[DatasetConfig, Dict],
        mode: str = "train",
        start_date: str = "2017-01",
        end_date: str = "2020-12",
        scaler: Optional[Any] = None,
    ):
        """
        Initialize FlowNetwork dataset.

        Args:
            data_dir: Directory containing monthly .pkl files
            edges: List of (source, target) tuples to model
            config: Dataset configuration
            mode: Dataset mode ('train', 'val', or 'test')
            start_date: Start date in YYYY-MM format
            end_date: End date in YYYY-MM format
            scaler: Optional scaler for normalization
        """
        self.data_dir = Path(data_dir)
        self.edges = edges
        self.start_date = start_date
        self.end_date = end_date
        self.scaler = scaler

        # Will be populated by _load_data
        self.timestamps = []
        self.raw_series = None  # [T, E] where E is number of edges
        self._data = None  # Will be set after processing

        super().__init__(config, mode)

    def _load_description(self) -> None:
        """Load dataset description."""
        self._description = {
            'num_edges': len(self.edges),
            'start_date': self.start_date,
            'end_date': self.end_date,
            'data_dir': str(self.data_dir),
        }

    def _load_data(self) -> None:
        """
        Load monthly networks and build time series.

        This method:
        1. Loads all monthly FlowNetwork files
        2. Extracts time series for specified edges
        3. Constructs sliding windows
        """
        # Load monthly networks
        monthly_networks = self._load_monthly_networks()

        if not monthly_networks:
            raise ValueError(f"No data loaded from {self.data_dir}")

        # Build time series matrix [T, E]
        self.timestamps = sorted(monthly_networks.keys())
        self.raw_series = self._build_time_series(monthly_networks)

        # Apply scaler if provided
        if self.scaler is not None:
            if self.mode == "train" and hasattr(self.scaler, 'fit'):
                self._data = self.scaler.fit_transform(self.raw_series)
            else:
                self._data = self.scaler.transform(self.raw_series)
        else:
            self._data = self.raw_series.copy()

        logger.info(
            f"FlowNetworkDataset [{self.mode}]: Loaded {len(self.timestamps)} timestamps, "
            f"{len(self.edges)} edges, {len(self)} samples"
        )

    def _load_monthly_networks(self) -> Dict[str, "FlowNetwork"]:
        """
        Load all monthly FlowNetwork pickle files within date range.

        Returns:
            Dictionary mapping timestamp to FlowNetwork
        """
        from flow_network import FlowNetwork

        monthly_data = {}
        start_dt = datetime.strptime(self.start_date, "%Y-%m")
        end_dt = datetime.strptime(self.end_date, "%Y-%m")

        for file_path in sorted(self.data_dir.glob("*.pkl")):
            try:
                timestamp = file_path.stem
                file_dt = datetime.strptime(timestamp, "%Y-%m")

                if file_dt < start_dt or file_dt > end_dt:
                    continue

                with open(file_path, 'rb') as f:
                    network = pickle.load(f)
                    if isinstance(network, FlowNetwork):
                        monthly_data[timestamp] = network

            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")

        return monthly_data

    def _build_time_series(
        self,
        monthly_networks: Dict[str, "FlowNetwork"]
    ) -> np.ndarray:
        """
        Build time series matrix from networks.

        Args:
            monthly_networks: Dictionary of monthly FlowNetworks

        Returns:
            Time series matrix [T, E] where T is time steps, E is edges
        """
        series = np.zeros((len(self.timestamps), len(self.edges)))

        for t, timestamp in enumerate(self.timestamps):
            network = monthly_networks[timestamp]
            for e, (source, target) in enumerate(self.edges):
                weight = network.get_edge_weight(source, target)
                series[t, e] = weight

        return series

    def __len__(self) -> int:
        """Return number of samples (sliding windows)."""
        total_len = self.config.input_len + self.config.output_len
        if len(self._data) < total_len:
            return 0

        if self.config.overlap:
            return len(self._data) - total_len + 1
        else:
            return (len(self._data) - total_len) // self.config.output_len + 1

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """
        Get a single sample.

        Returns:
            Dictionary with:
                - 'inputs': [input_len, num_edges]
                - 'target': [output_len, num_edges]
                - 'metadata': timestamps and edge info
        """
        if self.config.overlap:
            start_idx = index
        else:
            start_idx = index * self.config.output_len

        end_idx = start_idx + self.config.input_len
        target_end_idx = end_idx + self.config.output_len

        inputs = self._data[start_idx:end_idx]  # [input_len, num_edges]
        target = self._data[end_idx:target_end_idx]  # [output_len, num_edges]

        return {
            'inputs': inputs,
            'target': target,
            'metadata': {
                'start_idx': start_idx,
                'timestamps': self.timestamps[start_idx:target_end_idx],
                'edges': self.edges,
            }
        }

    def get_raw_series(self) -> np.ndarray:
        """Return raw time series without scaling."""
        return self.raw_series

    def get_timestamps(self) -> List[str]:
        """Return list of timestamps."""
        return self.timestamps

    def get_edges(self) -> List[Tuple[Union[int, str], Union[int, str]]]:
        """Return list of edges."""
        return self.edges


class FlowNetworkDataLoader:
    """
    Helper class for loading and preprocessing FlowNetwork data.

    This class handles the common workflow of:
    1. Loading monthly networks
    2. Selecting representative edges
    3. Creating train/val/test datasets
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        start_date: str = "2017-01",
        end_date: str = "2020-12",
    ):
        """
        Initialize data loader.

        Args:
            data_dir: Directory containing monthly .pkl files
            start_date: Start date in YYYY-MM format
            end_date: End date in YYYY-MM format
        """
        self.data_dir = Path(data_dir)
        self.start_date = start_date
        self.end_date = end_date

    def load_networks(self) -> Dict[str, "FlowNetwork"]:
        """Load all monthly networks."""
        from flow_network import FlowNetwork

        networks = {}
        start_dt = datetime.strptime(self.start_date, "%Y-%m")
        end_dt = datetime.strptime(self.end_date, "%Y-%m")

        for file_path in sorted(self.data_dir.glob("*.pkl")):
            try:
                timestamp = file_path.stem
                file_dt = datetime.strptime(timestamp, "%Y-%m")

                if start_dt <= file_dt <= end_dt:
                    with open(file_path, 'rb') as f:
                        network = pickle.load(f)
                        if isinstance(network, FlowNetwork):
                            networks[timestamp] = network
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")

        return networks

    def create_datasets(
        self,
        edges: List[Tuple[Union[int, str], Union[int, str]]],
        input_len: int = 6,
        output_len: int = 1,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        overlap: bool = False,
        scaler: Optional[Any] = None,
    ) -> Tuple[FlowNetworkDataset, FlowNetworkDataset, FlowNetworkDataset]:
        """
        Create train/val/test datasets.

        Args:
            edges: List of edges to model
            input_len: Input sequence length
            output_len: Output sequence length
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            overlap: Whether to use overlapping windows
            scaler: Optional scaler for normalization

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        config = DatasetConfig(
            input_len=input_len,
            output_len=output_len,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            overlap=overlap,
        )

        train_dataset = FlowNetworkDataset(
            data_dir=self.data_dir,
            edges=edges,
            config=config,
            mode="train",
            start_date=self.start_date,
            end_date=self.end_date,
            scaler=scaler,
        )

        val_dataset = FlowNetworkDataset(
            data_dir=self.data_dir,
            edges=edges,
            config=config,
            mode="val",
            start_date=self.start_date,
            end_date=self.end_date,
            scaler=scaler,
        )

        test_dataset = FlowNetworkDataset(
            data_dir=self.data_dir,
            edges=edges,
            config=config,
            mode="test",
            start_date=self.start_date,
            end_date=self.end_date,
            scaler=scaler,
        )

        return train_dataset, val_dataset, test_dataset
