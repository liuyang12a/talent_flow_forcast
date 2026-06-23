"""
Time series characteristics analysis module.

This module provides tools for analyzing and classifying time series
based on various characteristics like trend, seasonality, volatility, etc.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from pathlib import Path
import json
import logging
import warnings

logger = logging.getLogger(__name__)


class SeriesAnalyzer:
    """
    Analyzer for time series characteristics.

    Provides methods to compute various statistical features and
c    classify series based on trend, seasonality, volatility, etc.
    """

    def __init__(self, period: int = 12):
        """
        Initialize the analyzer.

        Args:
            period: Seasonal period (default 12 for monthly data)
        """
        self.period = period

    @staticmethod
    def _is_analyzable(series: np.ndarray, min_length: int = 2) -> bool:
        """
        Check if a series has enough variation for statistical analysis.

        Args:
            series: Time series array
            min_length: Minimum number of finite values required

        Returns:
            True if the series can be meaningfully analyzed
        """
        series = np.asarray(series)
        finite = series[np.isfinite(series)]
        if len(finite) < min_length:
            return False
        if np.std(finite) < 1e-12:
            return False
        return True

    def calculate_statistics(self, series: np.ndarray) -> Dict[str, float]:
        """
        Calculate basic statistical features of a time series.

        Args:
            series: Time series array

        Returns:
            Dictionary of statistical features
        """
        series = np.asarray(series)
        series = series[np.isfinite(series)]  # Remove NaN/Inf

        if len(series) == 0:
            return {
                'mean': 0.0, 'std': 0.0, 'cv': 0.0,
                'skewness': 0.0, 'kurtosis': 0.0,
                'min': 0.0, 'max': 0.0, 'range': 0.0
            }

        mean = np.mean(series)
        std = np.std(series)

        # skew / kurtosis are undefined for constant series (std ≈ 0)
        if std < 1e-12:
            skewness = 0.0
            kurtosis = 0.0
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                skewness = float(stats.skew(series))
                kurtosis = float(stats.kurtosis(series))

        return {
            'mean': float(mean),
            'std': float(std),
            'cv': float(std / (abs(mean) + 1e-8)),  # Coefficient of variation
            'skewness': skewness,
            'kurtosis': kurtosis,
            'min': float(np.min(series)),
            'max': float(np.max(series)),
            'range': float(np.max(series) - np.min(series))
        }

    def test_stationarity(self, series: np.ndarray) -> Dict[str, any]:
        """
        Test stationarity using Augmented Dickey-Fuller test.

        Args:
            series: Time series array

        Returns:
            Dictionary with test results
        """
        defaults = {
            'adf_statistic': 0.0,
            'p_value': 1.0,
            'is_stationary': False,
            'critical_values': {},
            'trend_strength': 0.0
        }

        series = np.asarray(series)
        series = series[np.isfinite(series)]

        if not self._is_analyzable(series, min_length=10):
            return defaults

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            try:
                from statsmodels.tsa.stattools import adfuller

                result = adfuller(series, autolag='AIC')

                # Calculate trend strength based on first-order autocorrelation
                if np.std(series) > 1e-12:
                    trend_strength = np.corrcoef(series[:-1], series[1:])[0, 1]
                    if np.isnan(trend_strength):
                        trend_strength = 0.0
                else:
                    trend_strength = 0.0

                return {
                    'adf_statistic': float(result[0]),
                    'p_value': float(result[1]),
                    'is_stationary': result[1] < 0.05,
                    'critical_values': {k: float(v) for k, v in result[4].items()},
                    'trend_strength': float(abs(trend_strength))
                }

            except Exception as e:
                logger.debug(f"Stationarity test failed: {e}")
                return defaults

    def detect_seasonality(self, series: np.ndarray) -> Dict[str, float]:
        """
        Detect seasonality strength using autocorrelation.

        Args:
            series: Time series array

        Returns:
            Dictionary with seasonality metrics
        """
        defaults = {
            'seasonal_strength': 0.0,
            'seasonal_peak_lag': 0,
            'has_seasonality': False
        }

        series = np.asarray(series)
        series = series[np.isfinite(series)]

        if not self._is_analyzable(series, min_length=self.period * 2):
            return defaults

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            try:
                from statsmodels.tsa.stattools import acf

                # Compute autocorrelation
                nlags = min(len(series) // 2, self.period * 2)
                autocorr = acf(series, nlags=nlags, fft=True)

                # Seasonal strength at the period lag
                seasonal_strength = autocorr[self.period] if len(autocorr) > self.period else 0.0

                # Find peak in seasonal lags
                seasonal_lags = autocorr[self.period:min(len(autocorr), self.period * 2 + 1)]
                if len(seasonal_lags) > 0:
                    seasonal_peak_lag = np.argmax(seasonal_lags) + self.period
                else:
                    seasonal_peak_lag = self.period

                return {
                    'seasonal_strength': float(seasonal_strength),
                    'seasonal_peak_lag': int(seasonal_peak_lag),
                    'has_seasonality': seasonal_strength > 0.3
                }

            except Exception as e:
                logger.debug(f"Seasonality detection failed: {e}")
                return defaults

    def analyze_trend(self, series: np.ndarray) -> Dict[str, float]:
        """
        Analyze trend characteristics.

        Args:
            series: Time series array

        Returns:
            Dictionary with trend metrics
        """
        defaults = {
            'slope': 0.0,
            'trend_direction': 'flat',
            'trend_magnitude': 0.0,
            'r_squared': 0.0,
            'p_value': 1.0
        }

        series = np.asarray(series)
        series = series[np.isfinite(series)]

        if not self._is_analyzable(series, min_length=2):
            return defaults

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            # Simple linear regression for trend
            x = np.arange(len(series))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)

            # Determine trend direction
            if abs(slope) < 1e-6:
                trend_direction = 'flat'
            elif slope > 0:
                trend_direction = 'increasing'
            else:
                trend_direction = 'decreasing'

            # Normalize trend magnitude
            mean_val = np.mean(series)
            if abs(mean_val) > 1e-8:
                trend_magnitude = abs(slope * len(series) / mean_val)
            else:
                trend_magnitude = abs(slope * len(series))

            return {
                'slope': float(slope),
                'trend_direction': trend_direction,
                'trend_magnitude': float(trend_magnitude),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value)
            }

    def analyze_all(self, series: np.ndarray) -> Dict[str, any]:
        """
        Perform complete analysis of a time series.

        Args:
            series: Time series array

        Returns:
            Complete analysis results
        """
        return {
            'statistics': self.calculate_statistics(series),
            'stationarity': self.test_stationarity(series),
            'seasonality': self.detect_seasonality(series),
            'trend': self.analyze_trend(series)
        }

    def classify_series(self, series: np.ndarray, thresholds: Dict = None) -> Dict[str, str]:
        """
        Classify a time series based on its characteristics.

        Args:
            series: Time series array
            thresholds: Custom thresholds for classification

        Returns:
            Classification labels
        """
        if thresholds is None:
            thresholds = {
                'volatility_high': 0.5,
                'volatility_low': 0.2,
                'seasonality_strong': 0.3
            }

        analysis = self.analyze_all(series)

        classification = {}

        # Volume classification (based on mean flow)
        mean_flow = analysis['statistics']['mean']
        # Note: volume classification requires comparison across all series
        # This will be done separately
        classification['volume'] = 'unknown'

        # Volatility classification
        cv = analysis['statistics']['cv']
        if cv > thresholds['volatility_high']:
            classification['volatility'] = 'high'
        elif cv < thresholds['volatility_low']:
            classification['volatility'] = 'low'
        else:
            classification['volatility'] = 'medium'

        # Trend classification
        is_stationary = analysis['stationarity']['is_stationary']
        trend_magnitude = analysis['trend']['trend_magnitude']

        if not is_stationary and trend_magnitude > 0.1:
            classification['trend'] = 'strong'
        elif not is_stationary:
            classification['trend'] = 'weak'
        else:
            classification['trend'] = 'none'

        # Seasonality classification
        seasonal_strength = analysis['seasonality']['seasonal_strength']
        if seasonal_strength > thresholds['seasonality_strong']:
            classification['seasonality'] = 'strong'
        elif seasonal_strength > 0.1:
            classification['seasonality'] = 'weak'
        else:
            classification['seasonality'] = 'none'

        return classification

    def analyze_series_collection(
        self,
        series_dict: Dict[str, np.ndarray]
    ) -> Dict[str, Dict]:
        """
        Analyze a collection of time series.

        Args:
            series_dict: Dictionary mapping series names to arrays

        Returns:
            Dictionary of analysis results
        """
        results = {}

        # First pass: collect all statistics
        all_means = []
        for name, series in series_dict.items():
            stats = self.calculate_statistics(series)
            all_means.append(stats['mean'])
            results[name] = {'statistics': stats}

        # Calculate volume thresholds
        if len(all_means) > 0:
            high_threshold = np.percentile(all_means, 80)
            low_threshold = np.percentile(all_means, 20)
        else:
            high_threshold = float('inf')
            low_threshold = 0

        # Second pass: complete analysis and classification
        for name, series in series_dict.items():
            analysis = self.analyze_all(series)

            # Add volume classification
            mean_flow = analysis['statistics']['mean']
            if mean_flow >= high_threshold:
                analysis['classification'] = {'volume': 'high'}
            elif mean_flow <= low_threshold:
                analysis['classification'] = {'volume': 'low'}
            else:
                analysis['classification'] = {'volume': 'medium'}

            # Add other classifications
            other_class = self.classify_series(series)
            analysis['classification'].update(other_class)

            results[name] = analysis

        return results


def _convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [_convert_to_serializable(v) for v in obj]
    return obj


def save_characteristics(
    characteristics: Dict,
    output_path: Path
) -> None:
    """
    Save series characteristics to a JSON file.

    Args:
        characteristics: Characteristics dictionary
        output_path: Path to save the file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to native Python types
    serializable_characteristics = _convert_to_serializable(characteristics)

    with open(output_path, 'w') as f:
        json.dump(serializable_characteristics, f, indent=2)

    logger.info(f"Characteristics saved to {output_path}")


def load_characteristics(input_path: Path) -> Dict:
    """
    Load series characteristics from a JSON file.

    Args:
        input_path: Path to the JSON file

    Returns:
        Characteristics dictionary
    """
    with open(input_path, 'r') as f:
        return json.load(f)
