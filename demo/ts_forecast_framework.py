"""
Time Series Forecasting Framework for Talent Flow Networks

This module provides a complete framework for forecasting monthly employee
flow between companies based on historical network data.

Author: Yang Liu
Date: 2024
"""

import json
import gzip
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FlowMetrics:
    """Time series metrics for a company pair flow."""
    source_company: str
    target_company: str
    timestamps: List[str] = field(default_factory=list)
    flow_counts: List[float] = field(default_factory=list)
    flow_weights: List[float] = field(default_factory=list)

    def to_array(self) -> np.ndarray:
        """Convert flow counts to numpy array."""
        return np.array(self.flow_counts)

    def get_time_features(self) -> np.ndarray:
        """Extract time-based features from timestamps."""
        features = []
        for ts in self.timestamps:
            dt = datetime.strptime(ts, "%Y-%m")
            features.append([
                dt.year,
                dt.month,
                dt.quarter,
                dt.month % 12 + 1,  # Season indicator
            ])
        return np.array(features)


class FlowDataLoader:
    """Load and preprocess monthly flow network data."""

    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
        self.flow_metrics: Dict[Tuple[str, str], FlowMetrics] = {}

    def load_monthly_data(self, pattern: str = "flow_*.json.gz") -> Dict[str, Dict]:
        """
        Load all monthly flow network files.

        Args:
            pattern: Glob pattern for data files

        Returns:
            Dictionary mapping timestamp to flow network data
        """
        monthly_data = {}

        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist")
            return monthly_data

        for file_path in sorted(self.data_dir.glob(pattern)):
            try:
                # Extract timestamp from filename (e.g., flow_2023-01.json.gz)
                timestamp = file_path.stm.replace('.json', '').replace('flow_', '')

                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
                    monthly_data[timestamp] = data
                    logger.info(f"Loaded data for {timestamp}")

            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        return monthly_data

    def build_time_series(
        self,
        monthly_data: Dict[str, Dict],
        min_observations: int = 6
    ) -> Dict[Tuple[str, str], FlowMetrics]:
        """
        Build time series for each company pair.

        Args:
            monthly_data: Dictionary of monthly flow data
            min_observations: Minimum number of observations required

        Returns:
            Dictionary mapping (source, target) to FlowMetrics
        """
        all_timestamps = sorted(monthly_data.keys())
        temp_data: Dict[Tuple[str, str], Dict[str, Dict]] = {}

        # Collect data for each company pair across all months
        for timestamp, data in monthly_data.items():
            edges = data.get('edges', [])

            for edge in edges:
                source = edge.get('source', '')
                target = edge.get('target', '')
                count = edge.get('count', 0)
                weight = edge.get('weight', 0.0)

                key = (source, target)
                if key not in temp_data:
                    temp_data[key] = {}
                temp_data[key][timestamp] = {'count': count, 'weight': weight}

        # Build complete time series with zero padding
        self.flow_metrics = {}
        for key, month_data in temp_data.items():
            if len(month_data) >= min_observations:
                metrics = FlowMetrics(
                    source_company=key[0],
                    target_company=key[1]
                )

                for ts in all_timestamps:
                    metrics.timestamps.append(ts)
                    if ts in month_data:
                        metrics.flow_counts.append(float(month_data[ts]['count']))
                        metrics.flow_weights.append(float(month_data[ts]['weight']))
                    else:
                        metrics.flow_counts.append(0.0)
                        metrics.flow_weights.append(0.0)

                self.flow_metrics[key] = metrics

        logger.info(f"Built {len(self.flow_metrics)} time series")
        return self.flow_metrics


class FeatureEngineer:
    """Feature engineering for time series forecasting."""

    def __init__(self, lookback_window: int = 6, forecast_horizon: int = 1):
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon

    def create_lag_features(
        self,
        series: np.ndarray,
        n_lags: Optional[int] = None
    ) -> np.ndarray:
        """
        Create lagged features.

        Args:
            series: Input time series
            n_lags: Number of lag features (default: lookback_window)

        Returns:
            Array of lag features [n_samples, n_lags]
        """
        if n_lags is None:
            n_lags = self.lookback_window

        n_samples = len(series) - n_lags - self.forecast_horizon + 1
        features = np.zeros((n_samples, n_lags))

        for i in range(n_lags):
            features[:, i] = series[i:i + n_samples]

        return features

    def create_rolling_features(
        self,
        series: np.ndarray,
        windows: List[int] = [3, 6]
    ) -> np.ndarray:
        """
        Create rolling window statistics.

        Args:
            series: Input time series
            windows: List of window sizes

        Returns:
            Array of rolling features
        """
        n_lags = self.lookback_window
        n_samples = len(series) - n_lags - self.forecast_horizon + 1

        rolling_features = []

        for window in windows:
            if window <= n_lags:
                # Calculate rolling mean
                rolling_mean = []
                for i in range(n_samples):
                    window_data = series[i + n_lags - window:i + n_lags]
                    rolling_mean.append(np.mean(window_data))
                rolling_features.append(np.array(rolling_mean))

                # Calculate rolling std
                rolling_std = []
                for i in range(n_samples):
                    window_data = series[i + n_lags - window:i + n_lags]
                    rolling_std.append(np.std(window_data))
                rolling_features.append(np.array(rolling_std))

        return np.column_stack(rolling_features) if rolling_features else np.zeros((n_samples, 0))

    def create_time_features(self, timestamps: List[str]) -> np.ndarray:
        """
        Create time-based features.

        Args:
            timestamps: List of timestamp strings (YYYY-MM format)

        Returns:
            Array of time features
        """
        features = []
        for ts in timestamps[self.lookback_window:]:
            dt = datetime.strptime(ts, "%Y-%m")
            features.append([
                dt.year,
                dt.month,
                dt.quarter,
                int(dt.month in [3, 4, 5]),    # Spring
                int(dt.month in [6, 7, 8]),    # Summer
                int(dt.month in [9, 10, 11]),  # Autumn
                int(dt.month in [12, 1, 2]),   # Winter
            ])
        return np.array(features)

    def prepare_dataset(
        self,
        flow_metrics: FlowMetrics,
        use_time_features: bool = True,
        use_rolling: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and target for a single flow time series.

        Args:
            flow_metrics: FlowMetrics object
            use_time_features: Whether to include time features
            use_rolling: Whether to include rolling features

        Returns:
            Tuple of (X, y, timestamps)
        """
        series = flow_metrics.to_array()
        timestamps = flow_metrics.timestamps

        # Create lag features
        X = self.create_lag_features(series)

        # Add rolling features
        if use_rolling:
            rolling_feats = self.create_rolling_features(series)
            if rolling_feats.shape[1] > 0:
                X = np.column_stack([X, rolling_feats])

        # Add time features
        if use_time_features:
            time_feats = self.create_time_features(timestamps)
            X = np.column_stack([X, time_feats])

        # Create target
        y = series[self.lookback_window + self.forecast_horizon - 1:]

        # Align timestamps
        aligned_timestamps = timestamps[self.lookback_window + self.forecast_horizon - 1:]

        return X, y, aligned_timestamps


class FlowForecaster:
    """Time series forecasting model for talent flow."""

    def __init__(self, model_type: str = "ridge", **model_params):
        self.model_type = model_type
        self.model_params = model_params
        self.model = None
        self.scaler = StandardScaler()
        self.feature_eng = FeatureEngineer()

        self._init_model()

    def _init_model(self):
        """Initialize the forecasting model."""
        if self.model_type == "ridge":
            self.model = Ridge(alpha=1.0, **self.model_params)
        elif self.model_type == "lasso":
            self.model = Lasso(alpha=0.1, **self.model_params)
        elif self.model_type == "rf":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                **self.model_params
            )
        elif self.model_type == "gb":
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                **self.model_params
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scale_features: bool = True
    ):
        """
        Fit the forecasting model.

        Args:
            X: Feature matrix
            y: Target vector
            scale_features: Whether to standardize features
        """
        if scale_features:
            X = self.scaler.fit_transform(X)

        self.model.fit(X, y)
        logger.info(f"Trained {self.model_type} model")

    def predict(self, X: np.ndarray, scale_features: bool = True) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature matrix
            scale_features: Whether features were scaled during training

        Returns:
            Predictions array
        """
        if scale_features:
            X = self.scaler.transform(X)
        return self.model.predict(X)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scale_features: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Feature matrix
            y: True target values
            scale_features: Whether features were scaled

        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X, scale_features)

        return {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'mape': np.mean(np.abs((y - y_pred) / (y + 1e-8))) * 100
        }

    def save(self, path: Union[str, Path]):
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type
            }, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: Union[str, Path]):
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.model_type = data['model_type']
        logger.info(f"Model loaded from {path}")


class ForecastingPipeline:
    """End-to-end forecasting pipeline."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        lookback_window: int = 6,
        forecast_horizon: int = 1,
        model_type: str = "ridge"
    ):
        self.data_loader = FlowDataLoader(data_dir)
        self.feature_eng = FeatureEngineer(lookback_window, forecast_horizon)
        self.forecaster = FlowForecaster(model_type)
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon

    def run(
        self,
        test_size: int = 3,
        min_observations: int = 12,
        top_k: Optional[int] = None
    ) -> Dict[Tuple[str, str], Dict[str, float]]:
        """
        Run the complete forecasting pipeline.

        Args:
            test_size: Number of months to use for testing
            min_observations: Minimum observations required
            top_k: Only evaluate top-k flows by volume (None for all)

        Returns:
            Dictionary of evaluation results per flow
        """
        # Load data
        logger.info("Loading monthly flow data...")
        monthly_data = self.data_loader.load_monthly_data()

        if not monthly_data:
            logger.error("No data loaded. Please check data directory.")
            return {}

        # Build time series
        logger.info("Building time series...")
        flow_metrics = self.data_loader.build_time_series(monthly_data, min_observations)

        if not flow_metrics:
            logger.error("No time series built. Check min_observations parameter.")
            return {}

        # Select top-k flows if specified
        if top_k:
            sorted_flows = sorted(
                flow_metrics.items(),
                key=lambda x: sum(x[1].flow_counts),
                reverse=True
            )
            flow_metrics = dict(sorted_flows[:top_k])

        # Train and evaluate for each flow
        results = {}
        all_predictions = []

        for key, metrics in flow_metrics.items():
            logger.info(f"Processing flow: {key[0]} -> {key[1]}")

            # Prepare dataset
            X, y, timestamps = self.feature_eng.prepare_dataset(metrics)

            if len(X) < test_size + 1:
                logger.warning(f"Insufficient data for {key}, skipping")
                continue

            # Train-test split
            X_train, X_test = X[:-test_size], X[-test_size:]
            y_train, y_test = y[:-test_size], y[-test_size:]
            timestamps_test = timestamps[-test_size:]

            # Train model
            self.forecaster.fit(X_train, y_train)

            # Evaluate
            train_metrics = self.forecaster.evaluate(X_train, y_train)
            test_metrics = self.forecaster.evaluate(X_test, y_test)

            # Store results
            results[key] = {
                'train_rmse': train_metrics['rmse'],
                'test_rmse': test_metrics['rmse'],
                'test_mae': test_metrics['mae'],
                'test_r2': test_metrics['r2'],
                'test_mape': test_metrics['mape'],
                'n_train': len(X_train),
                'n_test': len(X_test)
            }

            # Store predictions
            y_pred = self.forecaster.predict(X_test)
            for i, ts in enumerate(timestamps_test):
                all_predictions.append({
                    'source': key[0],
                    'target': key[1],
                    'timestamp': ts,
                    'actual': float(y_test[i]),
                    'predicted': float(y_pred[i]),
                    'error': float(abs(y_test[i] - y_pred[i]))
                })

        # Save results
        self._save_results(results, all_predictions)

        return results

    def _save_results(
        self,
        results: Dict,
        predictions: List[Dict],
        output_dir: str = "demo/output"
    ):
        """Save evaluation results and predictions."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save metrics
        with open(output_path / "metrics.json", 'w') as f:
            # Convert tuple keys to strings for JSON
            json_results = {f"{k[0]}->{k[1]}": v for k, v in results.items()}
            json.dump(json_results, f, indent=2)

        # Save predictions
        with open(output_path / "predictions.json", 'w') as f:
            json.dump(predictions, f, indent=2)

        logger.info(f"Results saved to {output_dir}")


def main():
    """Main entry point for demonstration."""
    # Configuration
    DATA_DIR = "data/flow_networks"
    MODEL_TYPE = "ridge"  # Options: ridge, lasso, rf, gb
    LOOKBACK_WINDOW = 6   # Use 6 months history
    FORECAST_HORIZON = 1  # Predict 1 month ahead
    TEST_SIZE = 3         # Use last 3 months for testing

    # Initialize pipeline
    pipeline = ForecastingPipeline(
        data_dir=DATA_DIR,
        lookback_window=LOOKBACK_WINDOW,
        forecast_horizon=FORECAST_HORIZON,
        model_type=MODEL_TYPE
    )

    # Run forecasting
    results = pipeline.run(
        test_size=TEST_SIZE,
        min_observations=12,
        top_k=10  # Evaluate top 10 flows
    )

    # Print summary
    if results:
        print("\n" + "="*60)
        print("FORECASTING RESULTS SUMMARY")
        print("="*60)

        avg_rmse = np.mean([r['test_rmse'] for r in results.values()])
        avg_mae = np.mean([r['test_mae'] for r in results.values()])
        avg_r2 = np.mean([r['test_r2'] for r in results.values()])

        print(f"Model: {MODEL_TYPE}")
        print(f"Flows evaluated: {len(results)}")
        print(f"Average Test RMSE: {avg_rmse:.4f}")
        print(f"Average Test MAE: {avg_mae:.4f}")
        print(f"Average Test R²: {avg_r2:.4f}")
        print("="*60)


if __name__ == "__main__":
    main()
