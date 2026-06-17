"""
ARIMA model for time series forecasting.

This module implements the ARIMA (AutoRegressive Integrated Moving Average)
model for univariate time series forecasting.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

from src.models.base_model import BaseStatisticalModel

logger = logging.getLogger(__name__)


class ARIMAModel(BaseStatisticalModel):
    """
    ARIMA (AutoRegressive Integrated Moving Average) model.

    ARIMA models are classical statistical models for time series forecasting
    that combine autoregression, differencing, and moving average components.

    The model is specified by three parameters (p, d, q):
    - p: Autoregressive order
    - d: Degree of differencing
    - q: Moving average order
    """

    def __init__(
        self,
        input_len: int = 12,
        output_len: int = 1,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        name: str = "ARIMA",
        **kwargs
    ):
        """
        Initialize ARIMA model.

        Args:
            input_len: Length of input sequence
            output_len: Length of output sequence (forecast horizon)
            order: ARIMA order (p, d, q)
            seasonal_order: Optional seasonal order (P, D, Q, s)
            name: Model name
            **kwargs: Additional parameters passed to statsmodels
        """
        super().__init__(input_len, output_len, name, **kwargs)
        self.order = order
        self.seasonal_order = seasonal_order
        self.models = {}  # One model per series
        self.statsmodels_kwargs = kwargs

    def _fit_impl(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Fit ARIMA models.

        For ARIMA, we fit a separate model for each time series.
        X contains historical data, y contains future values for training.
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA as StatsARIMA
        except ImportError:
            raise ImportError("statsmodels is required for ARIMA. Install with: uv pip install statsmodels")

        # Determine number of series
        if len(X.shape) == 1:
            n_series = 1
        elif len(X.shape) == 2:
            n_series = X.shape[1]
        else:
            n_series = X.shape[2] if X.shape[2] > 0 else 1

        logger.info(f"Fitting ARIMA{self.order} models for {n_series} series")

        # Fit a model for each series
        for i in range(n_series):
            try:
                if n_series == 1:
                    series_data = X.flatten()
                else:
                    series_data = X[:, i] if len(X.shape) == 2 else X[:, :, i].flatten()

                # Fit ARIMA model
                if self.seasonal_order:
                    model = StatsARIMA(
                        series_data,
                        order=self.order,
                        seasonal_order=self.seasonal_order,
                        **self.statsmodels_kwargs
                    )
                else:
                    model = StatsARIMA(
                        series_data,
                        order=self.order,
                        **self.statsmodels_kwargs
                    )

                fitted = model.fit()
                self.models[i] = fitted

            except Exception as e:
                logger.warning(f"Failed to fit ARIMA for series {i}: {e}")
                self.models[i] = None

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate predictions using fitted ARIMA models.

        Args:
            X: Input data [n_samples, input_len, n_features] or [n_samples, input_len]

        Returns:
            Predictions [n_samples, output_len, n_features] or [n_samples, output_len]
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        n_samples = X.shape[0]

        # Determine shape
        if len(X.shape) == 1:
            n_series = 1
            predictions = np.zeros((n_samples, self.output_len))
        elif len(X.shape) == 2:
            n_series = 1
            predictions = np.zeros((n_samples, self.output_len))
        else:
            n_series = X.shape[-1]
            predictions = np.zeros((n_samples, self.output_len, n_series))

        # Generate predictions for each sample and series
        for sample_idx in range(n_samples):
            for series_idx in range(n_series):
                try:
                    model = self.models.get(series_idx)
                    if model is None:
                        # Use last value as fallback
                        if n_series == 1:
                            last_value = X[sample_idx, -1]
                        else:
                            last_value = X[sample_idx, -1, series_idx]

                        pred = np.full(self.output_len, last_value)
                    else:
                        # Forecast using ARIMA
                        forecast = model.forecast(steps=self.output_len)
                        pred = forecast.values if hasattr(forecast, 'values') else forecast

                    if n_series == 1:
                        predictions[sample_idx] = pred
                    else:
                        predictions[sample_idx, :, series_idx] = pred

                except Exception as e:
                    logger.warning(f"Prediction failed for sample {sample_idx}, series {series_idx}: {e}")
                    # Use last known value as fallback
                    if n_series == 1:
                        predictions[sample_idx] = X[sample_idx, -1]
                    else:
                        predictions[sample_idx, :, series_idx] = X[sample_idx, -1, series_idx]

        return predictions

    def get_summary(self, series_idx: int = 0) -> str:
        """Get model summary for a specific series."""
        if series_idx not in self.models or self.models[series_idx] is None:
            return "No model fitted for this series"
        return str(self.models[series_idx].summary())


class AutoARIMAModel(BaseStatisticalModel):
    """
    Auto ARIMA model that automatically selects optimal parameters.

    Uses pmdarima's auto_arima to find the best (p, d, q) values.
    """

    def __init__(
        self,
        input_len: int = 12,
        output_len: int = 1,
        seasonal: bool = False,
        m: int = 1,
        name: str = "AutoARIMA",
        **kwargs
    ):
        """
        Initialize AutoARIMA model.

        Args:
            input_len: Length of input sequence
            output_len: Length of output sequence
            seasonal: Whether to include seasonal component
            m: Seasonal period
            name: Model name
            **kwargs: Additional parameters for auto_arima
        """
        super().__init__(input_len, output_len, name, **kwargs)
        self.seasonal = seasonal
        self.m = m
        self.models = {}
        self.auto_arima_kwargs = kwargs

    def _fit_impl(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Fit auto ARIMA models."""
        try:
            from pmdarima import auto_arima
        except ImportError:
            raise ImportError(
                "pmdarima is required for AutoARIMA. Install with: uv pip install pmdarima"
            )

        # Determine number of series
        if len(X.shape) == 1:
            n_series = 1
        elif len(X.shape) == 2:
            n_series = X.shape[1]
        else:
            n_series = X.shape[2] if X.shape[2] > 0 else 1

        logger.info(f"Fitting AutoARIMA models for {n_series} series")

        # Fit a model for each series
        for i in range(n_series):
            try:
                if n_series == 1:
                    series_data = X.flatten()
                else:
                    series_data = X[:, i] if len(X.shape) == 2 else X[:, :, i].flatten()

                # Fit auto ARIMA
                model = auto_arima(
                    series_data,
                    seasonal=self.seasonal,
                    m=self.m,
                    suppress_warnings=True,
                    **self.auto_arima_kwargs
                )

                self.models[i] = model

            except Exception as e:
                logger.warning(f"Failed to fit AutoARIMA for series {i}: {e}")
                self.models[i] = None

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Generate predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        n_samples = X.shape[0]

        # Determine shape
        if len(X.shape) == 1:
            n_series = 1
            predictions = np.zeros((n_samples, self.output_len))
        elif len(X.shape) == 2:
            n_series = 1
            predictions = np.zeros((n_samples, self.output_len))
        else:
            n_series = X.shape[-1]
            predictions = np.zeros((n_samples, self.output_len, n_series))

        # Generate predictions
        for sample_idx in range(n_samples):
            for series_idx in range(n_series):
                try:
                    model = self.models.get(series_idx)
                    if model is None:
                        # Use last value as fallback
                        if n_series == 1:
                            last_value = X[sample_idx, -1]
                        else:
                            last_value = X[sample_idx, -1, series_idx]
                        pred = np.full(self.output_len, last_value)
                    else:
                        pred = model.predict(n_periods=self.output_len)

                    if n_series == 1:
                        predictions[sample_idx] = pred
                    else:
                        predictions[sample_idx, :, series_idx] = pred

                except Exception as e:
                    logger.warning(f"Prediction failed: {e}")
                    if n_series == 1:
                        predictions[sample_idx] = X[sample_idx, -1]
                    else:
                        predictions[sample_idx, :, series_idx] = X[sample_idx, -1, series_idx]

        return predictions
