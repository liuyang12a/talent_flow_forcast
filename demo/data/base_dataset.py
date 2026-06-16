"""
Base dataset classes for time series forecasting.

This module provides abstract base classes for time series datasets,
designed to be compatible with different forecasting tasks including:
- Univariate time series forecasting
- Multivariate time series forecasting
- Spatial-temporal forecasting (graph-based)

Reference: Inspired by BasicTS framework (D:/experiments/tsf/basicts)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class DatasetConfig:
    """Configuration for time series datasets."""
    input_len: int  # Length of input sequence (history)
    output_len: int  # Length of output sequence (forecast horizon)
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    overlap: bool = False  # Whether to use overlapping windows


class BaseDataset(ABC):
    """
    Abstract base class for time series datasets.

    This class defines the interface that all datasets must implement.
    Subclasses should override the abstract methods to provide specific
    data loading and preprocessing logic.

    Attributes:
        config: DatasetConfig instance containing dataset parameters
        mode: Current mode ('train', 'val', or 'test')
    """

    def __init__(self, config: Union[DatasetConfig, Dict], mode: str = "train"):
        """
        Initialize the dataset.

        Args:
            config: Dataset configuration (DatasetConfig or dict)
            mode: Dataset mode - 'train', 'val', or 'test'
        """
        if isinstance(config, dict):
            self.config = DatasetConfig(**config)
        else:
            self.config = config

        self.mode = mode
        self._data = None
        self._description = None

        # Load data
        self._load_description()
        self._load_data()

    @abstractmethod
    def _load_description(self) -> None:
        """
        Load dataset description/metadata.

        This should include information about the dataset such as:
        - Number of samples
        - Number of features
        - Number of nodes (for spatial-temporal data)
        - Time range
        """
        pass

    @abstractmethod
    def _load_data(self) -> None:
        """
        Load the actual data into memory.

        This method should populate self._data with the preprocessed data
        ready for indexing.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.

        Args:
            index: Sample index

        Returns:
            Dictionary containing:
                - 'inputs': Input sequence [input_len, ...]
                - 'target': Target sequence [output_len, ...]
                - 'metadata': Optional metadata dict
        """
        pass

    def get_description(self) -> Dict[str, Any]:
        """Return dataset description."""
        return self._description or {}

    def set_mode(self, mode: str) -> None:
        """
        Switch dataset mode and reload data if necessary.

        Args:
            mode: New mode ('train', 'val', or 'test')
        """
        if mode not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train', 'val', or 'test'")
        self.mode = mode
        self._load_data()


class TimeSeriesDataset(BaseDataset):
    """
    Generic time series dataset for univariate or multivariate forecasting.

    This dataset handles time series data in the format:
    - Single series: [T] or [T, 1]
    - Multivariate: [T, N] where N is number of variables

    The dataset automatically creates sliding windows for training.
    """

    def __init__(
        self,
        data: np.ndarray,
        config: Union[DatasetConfig, Dict],
        mode: str = "train",
        scalers: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize time series dataset.

        Args:
            data: Raw time series data [T, ...]
            config: Dataset configuration
            mode: Dataset mode
            scalers: Optional dict of scalers for preprocessing
        """
        self.raw_data = data
        self.scalers = scalers or {}
        super().__init__(config, mode)

    def _load_description(self) -> None:
        """Load dataset description from raw data."""
        self._description = {
            'num_samples': len(self.raw_data),
            'shape': self.raw_data.shape,
            'dtype': str(self.raw_data.dtype),
        }

    def _load_data(self) -> None:
        """
        Load and split data according to mode.

        Data is split sequentially:
        - Train: first train_ratio portion
        - Val: next val_ratio portion
        - Test: remaining test_ratio portion
        """
        n = len(self.raw_data)
        train_end = int(n * self.config.train_ratio)
        val_end = train_end + int(n * self.config.val_ratio)

        if self.mode == "train":
            self._data = self.raw_data[:train_end]
        elif self.mode == "val":
            self._data = self.raw_data[train_end:val_end]
        else:  # test
            self._data = self.raw_data[val_end:]

        # Apply scalers if provided
        for scaler_name, scaler in self.scalers.items():
            if hasattr(scaler, 'transform'):
                if self.mode == "train" and hasattr(scaler, 'fit'):
                    self._data = scaler.fit_transform(self._data)
                else:
                    self._data = scaler.transform(self._data)

    def __len__(self) -> int:
        """
        Return number of samples.

        For non-overlapping windows: (len - input_len - output_len) // output_len + 1
        For overlapping windows: len - input_len - output_len + 1
        """
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
            Dict with 'inputs' and 'target' arrays
        """
        if self.config.overlap:
            start_idx = index
        else:
            start_idx = index * self.config.output_len

        end_idx = start_idx + self.config.input_len
        target_end_idx = end_idx + self.config.output_len

        inputs = self._data[start_idx:end_idx]
        target = self._data[end_idx:target_end_idx]

        return {
            'inputs': inputs,
            'target': target,
            'metadata': {
                'start_idx': start_idx,
                'end_idx': end_idx,
            }
        }


class SpatialTemporalDataset(BaseDataset):
    """
    Spatial-temporal dataset for graph-based forecasting.

    This dataset handles data in the format:
    - [T, N, C] where:
        T: Time steps
        N: Number of nodes
        C: Number of features per node

    The dataset can be used for traffic forecasting, weather forecasting,
    and other spatial-temporal prediction tasks.
    """

    def __init__(
        self,
        data: np.ndarray,
        adjacency_matrix: Optional[np.ndarray] = None,
        config: Optional[Union[DatasetConfig, Dict]] = None,
        mode: str = "train",
        scalers: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize spatial-temporal dataset.

        Args:
            data: Raw data [T, N, C]
            adjacency_matrix: Graph adjacency matrix [N, N]
            config: Dataset configuration
            mode: Dataset mode
            scalers: Optional scalers for preprocessing
        """
        self.raw_data = data
        self.adjacency_matrix = adjacency_matrix
        self.scalers = scalers or {}
        config = config or DatasetConfig(input_len=12, output_len=12)
        super().__init__(config, mode)

    def _load_description(self) -> None:
        """Load dataset description."""
        self._description = {
            'num_timestamps': self.raw_data.shape[0],
            'num_nodes': self.raw_data.shape[1],
            'num_features': self.raw_data.shape[2],
            'has_graph': self.adjacency_matrix is not None,
        }

    def _load_data(self) -> None:
        """Load and split data."""
        n = len(self.raw_data)
        train_end = int(n * self.config.train_ratio)
        val_end = train_end + int(n * self.config.val_ratio)

        if self.mode == "train":
            self._data = self.raw_data[:train_end]
        elif self.mode == "val":
            self._data = self.raw_data[train_end:val_end]
        else:
            self._data = self.raw_data[val_end:]

    def __len__(self) -> int:
        """Return number of samples."""
        total_len = self.config.input_len + self.config.output_len
        if len(self._data) < total_len:
            return 0
        return len(self._data) - total_len + 1

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """Get a single sample."""
        start_idx = index
        end_idx = start_idx + self.config.input_len
        target_end_idx = end_idx + self.config.output_len

        inputs = self._data[start_idx:end_idx]  # [input_len, N, C]
        target = self._data[end_idx:target_end_idx]  # [output_len, N, C]

        result = {
            'inputs': inputs,
            'target': target,
            'metadata': {
                'start_idx': start_idx,
                'timestamps': list(range(start_idx, target_end_idx)),
            }
        }

        if self.adjacency_matrix is not None:
            result['adjacency'] = self.adjacency_matrix

        return result
