"""
Data preprocessing transforms and scalers.
"""

from typing import Optional, Union, List
import numpy as np


class BaseScaler:
    """Base class for data scalers."""

    def fit(self, data: np.ndarray) -> "BaseScaler":
        """Fit the scaler to the data."""
        raise NotImplementedError

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform the data."""
        raise NotImplementedError

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform the data."""
        return self.fit(data).transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform the data."""
        raise NotImplementedError


class ZScoreScaler(BaseScaler):
    """
    Z-score normalization scaler.

    Transforms data to have zero mean and unit variance.
    """

    def __init__(self, axis: Optional[int] = None, eps: float = 1e-8):
        """
        Initialize Z-score scaler.

        Args:
            axis: Axis along which to compute statistics. None means all axes.
            eps: Small value to avoid division by zero.
        """
        self.axis = axis
        self.eps = eps
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray) -> "ZScoreScaler":
        """Compute mean and std from training data."""
        self.mean = np.mean(data, axis=self.axis, keepdims=True)
        self.std = np.std(data, axis=self.axis, keepdims=True)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply Z-score normalization."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler must be fitted before transform")
        return (data - self.mean) / (self.std + self.eps)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse Z-score normalization."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler must be fitted before inverse_transform")
        return data * (self.std + self.eps) + self.mean


class MinMaxScaler(BaseScaler):
    """
    Min-Max normalization scaler.

    Scales data to a specified range (default [0, 1]).
    """

    def __init__(
        self,
        feature_range: tuple = (0, 1),
        axis: Optional[int] = None,
        eps: float = 1e-8
    ):
        """
        Initialize Min-Max scaler.

        Args:
            feature_range: Desired range (min, max).
            axis: Axis along which to compute statistics.
            eps: Small value to avoid division by zero.
        """
        self.feature_range = feature_range
        self.axis = axis
        self.eps = eps
        self.min = None
        self.max = None
        self.scale = None

    def fit(self, data: np.ndarray) -> "MinMaxScaler":
        """Compute min and max from training data."""
        self.min = np.min(data, axis=self.axis, keepdims=True)
        self.max = np.max(data, axis=self.axis, keepdims=True)
        self.scale = (self.feature_range[1] - self.feature_range[0]) / (self.max - self.min + self.eps)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply Min-Max scaling."""
        if self.min is None or self.max is None:
            raise RuntimeError("Scaler must be fitted before transform")
        return self.feature_range[0] + (data - self.min) * self.scale

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse Min-Max scaling."""
        if self.min is None or self.max is None:
            raise RuntimeError("Scaler must be fitted before inverse_transform")
        return self.min + (data - self.feature_range[0]) / self.scale


class DifferenceTransform(BaseScaler):
    """
    Difference transformation for time series.

    Useful for making non-stationary series stationary.
    """

    def __init__(self, periods: int = 1):
        """
        Initialize difference transform.

        Args:
            periods: Number of periods to difference.
        """
        self.periods = periods
        self.last_values = None

    def fit(self, data: np.ndarray) -> "DifferenceTransform":
        """Store last values for inverse transform."""
        self.last_values = data[-self.periods:].copy()
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply difference transformation."""
        return np.diff(data, n=self.periods, axis=0, prepend=data[:self.periods])

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse difference transformation using cumulative sum."""
        if self.last_values is None:
            raise RuntimeError("Transform must be fitted before inverse_transform")
        return np.cumsum(data, axis=0) + self.last_values


class SlidingWindowTransform:
    """
    Transform time series into sliding windows.

    Converts a 1D or 2D time series into input/target pairs for supervised learning.
    """

    def __init__(
        self,
        input_len: int,
        output_len: int,
        stride: int = 1,
        axis: int = 0
    ):
        """
        Initialize sliding window transform.

        Args:
            input_len: Length of input window.
            output_len: Length of output window.
            stride: Step size between consecutive windows.
            axis: Time axis.
        """
        self.input_len = input_len
        self.output_len = output_len
        self.stride = stride
        self.axis = axis

    def transform(self, data: np.ndarray) -> tuple:
        """
        Transform data into sliding windows.

        Args:
            data: Input array [T, ...] or [...]

        Returns:
            Tuple of (inputs, targets) arrays.
        """
        # Move time axis to front for easier indexing
        if self.axis != 0:
            data = np.moveaxis(data, self.axis, 0)

        n_samples = (len(data) - self.input_len - self.output_len) // self.stride + 1

        inputs = []
        targets = []

        for i in range(n_samples):
            start = i * self.stride
            input_end = start + self.input_len
            target_end = input_end + self.output_len

            inputs.append(data[start:input_end])
            targets.append(data[input_end:target_end])

        return np.array(inputs), np.array(targets)


class TimeFeatureEncoder:
    """
    Encode time features from timestamps.

    Creates cyclical and categorical time features.
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        freq: str = "monthly"
    ):
        """
        Initialize time feature encoder.

        Args:
            features: List of features to encode ('year', 'month', 'day',
                     'hour', 'dayofweek', 'quarter', 'season').
            freq: Data frequency ('monthly', 'daily', 'hourly').
        """
        self.features = features or ['month', 'quarter']
        self.freq = freq

    def encode(self, timestamps: List) -> np.ndarray:
        """
        Encode timestamps into features.

        Args:
            timestamps: List of datetime objects or strings.

        Returns:
            Array of encoded features [len(timestamps), n_features].
        """
        from datetime import datetime

        features = []
        for ts in timestamps:
            if isinstance(ts, str):
                if self.freq == "monthly":
                    dt = datetime.strptime(ts, "%Y-%m")
                elif self.freq == "daily":
                    dt = datetime.strptime(ts, "%Y-%m-%d")
                else:
                    dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            else:
                dt = ts

            feat = []
            if 'year' in self.features:
                feat.append(dt.year)
            if 'month' in self.features:
                # Cyclical encoding for month
                month_sin = np.sin(2 * np.pi * dt.month / 12)
                month_cos = np.cos(2 * np.pi * dt.month / 12)
                feat.extend([month_sin, month_cos])
            if 'day' in self.features:
                feat.append(dt.day)
            if 'hour' in self.features:
                # Cyclical encoding for hour
                hour_sin = np.sin(2 * np.pi * dt.hour / 24)
                hour_cos = np.cos(2 * np.pi * dt.hour / 24)
                feat.extend([hour_sin, hour_cos])
            if 'dayofweek' in self.features:
                feat.append(dt.weekday())
            if 'quarter' in self.features:
                feat.append((dt.month - 1) // 3 + 1)
            if 'season' in self.features:
                season = (dt.month % 12 + 3) // 3  # 1=Spring, 2=Summer, 3=Autumn, 4=Winter
                feat.append(season)

            features.append(feat)

        return np.array(features, dtype=np.float32)
