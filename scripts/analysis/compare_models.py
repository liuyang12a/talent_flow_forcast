"""
Model comparison module.

This module provides tools for comparing ARIMA and STGNN model performances
across different time series and characteristics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class ModelComparator:
    """
    Comparator for ARIMA and STGNN model performances.

    Provides methods to compare models across different metrics,
    series types, and characteristics.
    """

    def __init__(self, results_df: Optional[pd.DataFrame] = None):
        """
        Initialize the comparator.

        Args:
            results_df: DataFrame containing experiment results
        """
        self.results = results_df

    def load_results(self, results_path: Path) -> "ModelComparator":
        """
        Load results from a CSV or JSON file.

        Args:
            results_path: Path to the results file

        Returns:
            Self for method chaining
        """
        results_path = Path(results_path)

        if results_path.suffix == '.csv':
            self.results = pd.read_csv(results_path)
        elif results_path.suffix == '.json':
            with open(results_path, 'r') as f:
                data = json.load(f)
            self.results = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported file format: {results_path.suffix}")

        logger.info(f"Loaded {len(self.results)} results from {results_path}")
        return self

    def compare_by_metric(
        self,
        metric: str = 'mae',
        group_by: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Compare models by a specific metric.

        Args:
            metric: Metric name to compare
            group_by: Optional column to group by

        Returns:
            Comparison results dictionary
        """
        if self.results is None:
            raise ValueError("No results loaded. Call load_results() first.")

        if metric not in self.results.columns:
            raise ValueError(f"Metric '{metric}' not found in results")

        # Filter valid results
        valid_results = self.results[
            (self.results[metric].notna()) &
            (self.results[metric] != float('inf'))
        ]

        if group_by:
            comparison = {}
            for group in valid_results[group_by].unique():
                group_data = valid_results[valid_results[group_by] == group]
                comparison[group] = self._compute_comparison_stats(
                    group_data, metric
                )
            return comparison
        else:
            return self._compute_comparison_stats(valid_results, metric)

    def _compute_comparison_stats(
        self,
        data: pd.DataFrame,
        metric: str
    ) -> Dict[str, any]:
        """
        Compute comparison statistics for a dataset.

        Args:
            data: DataFrame subset
            metric: Metric column name

        Returns:
            Comparison statistics
        """
        arima_data = data[data['model_type'] == 'arima'][metric]
        stgnn_data = data[data['model_type'] == 'stgnn'][metric]

        if len(arima_data) == 0 or len(stgnn_data) == 0:
            return {
                'error': 'Insufficient data for comparison',
                'arima_count': len(arima_data),
                'stgnn_count': len(stgnn_data)
            }

        # Basic statistics
        arima_mean = arima_data.mean()
        stgnn_mean = stgnn_data.mean()
        arima_std = arima_data.std()
        stgnn_std = stgnn_data.std()

        # Relative improvement
        if arima_mean != 0:
            improvement = (arima_mean - stgnn_mean) / arima_mean * 100
        else:
            improvement = 0.0

        # Statistical test (Wilcoxon signed-rank test)
        # Need paired data for proper comparison
        try:
            if len(arima_data) == len(stgnn_data):
                statistic, p_value = stats.wilcoxon(arima_data, stgnn_data)
            else:
                statistic, p_value = stats.mannwhitneyu(
                    arima_data, stgnn_data, alternative='two-sided'
                )
            is_significant = p_value < 0.05
        except Exception as e:
            statistic, p_value, is_significant = 0, 1.0, False

        # Win counts (which model is better for each series)
        # This requires pairing by series_id
        if 'series_id' in data.columns:
            paired_data = data.pivot_table(
                index='series_id',
                columns='model_type',
                values=metric,
                aggfunc='mean'
            ).dropna()

            if len(paired_data) > 0:
                arima_wins = (paired_data['arima'] < paired_data['stgnn']).sum()
                stgnn_wins = (paired_data['stgnn'] < paired_data['arima']).sum()
                ties = (paired_data['arima'] == paired_data['stgnn']).sum()
            else:
                arima_wins = stgnn_wins = ties = 0
        else:
            arima_wins = stgnn_wins = ties = 0

        return {
            'arima_mean': float(arima_mean),
            'arima_std': float(arima_std),
            'stgnn_mean': float(stgnn_mean),
            'stgnn_std': float(stgnn_std),
            'improvement_pct': float(improvement),
            'statistical_test': {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'is_significant': is_significant
            },
            'win_counts': {
                'arima': int(arima_wins),
                'stgnn': int(stgnn_wins),
                'ties': int(ties)
            },
            'sample_counts': {
                'arima': len(arima_data),
                'stgnn': len(stgnn_data)
            }
        }

    def compare_by_series_type(self) -> Dict[str, Dict]:
        """
        Compare models grouped by series selector type.

        Returns:
            Comparison results by selector type
        """
        if self.results is None:
            raise ValueError("No results loaded.")

        if 'selector_type' not in self.results.columns:
            logger.warning("No 'selector_type' column in results")
            return {}

        comparison = {}
        for metric in ['mae', 'rmse', 'mape', 'r2']:
            if metric in self.results.columns:
                comparison[metric] = self.compare_by_metric(
                    metric, group_by='selector_type'
                )

        return comparison

    def compare_by_characteristic(
        self,
        characteristic: str
    ) -> Dict[str, Dict]:
        """
        Compare models grouped by series characteristic.

        Args:
            characteristic: Characteristic name (e.g., 'volatility', 'trend')

        Returns:
            Comparison results by characteristic level
        """
        if self.results is None:
            raise ValueError("No results loaded.")

        col_name = f'char_{characteristic}'
        if col_name not in self.results.columns:
            logger.warning(f"No '{col_name}' column in results")
            return {}

        comparison = {}
        for metric in ['mae', 'rmse', 'mape']:
            if metric in self.results.columns:
                comparison[metric] = self.compare_by_metric(
                    metric, group_by=col_name
                )

        return comparison

    def get_best_model_per_series(self) -> pd.DataFrame:
        """
        Determine the best model for each series.

        Returns:
            DataFrame with best model assignments
        """
        if self.results is None:
            raise ValueError("No results loaded.")

        if 'series_id' not in self.results.columns:
            raise ValueError("No 'series_id' column in results")

        # Group by series and model, compute mean metrics
        grouped = self.results.groupby(['series_id', 'model_type']).agg({
            'mae': 'mean',
            'rmse': 'mean',
            'mape': 'mean',
            'r2': 'mean'
        }).reset_index()

        # Pivot to compare models
        best_models = []
        for series_id in grouped['series_id'].unique():
            series_data = grouped[grouped['series_id'] == series_id]

            if len(series_data) < 2:
                continue

            arima_row = series_data[series_data['model_type'] == 'arima']
            stgnn_row = series_data[series_data['model_type'] == 'stgnn']

            if len(arima_row) == 0 or len(stgnn_row) == 0:
                continue

            arima_mae = arima_row['mae'].values[0]
            stgnn_mae = stgnn_row['mae'].values[0]

            if stgnn_mae < arima_mae:
                best_model = 'stgnn'
                mae_diff = arima_mae - stgnn_mae
            else:
                best_model = 'arima'
                mae_diff = stgnn_mae - arima_mae

            best_models.append({
                'series_id': series_id,
                'best_model': best_model,
                'mae_diff': mae_diff,
                'arima_mae': arima_mae,
                'stgnn_mae': stgnn_mae
            })

        return pd.DataFrame(best_models)

    def analyze_predictability_factors(self) -> Dict[str, any]:
        """
        Analyze which factors affect predictability.

        Returns:
            Analysis results
        """
        if self.results is None:
            raise ValueError("No results loaded.")

        analysis = {}

        # Correlate characteristics with error metrics
        characteristics = ['char_volume', 'char_volatility', 'char_trend', 'char_seasonality']
        metrics = ['mae', 'rmse', 'mape']

        for metric in metrics:
            if metric not in self.results.columns:
                continue

            analysis[metric] = {}

            for char in characteristics:
                if char not in self.results.columns:
                    continue

                # Compute correlation if characteristic is numeric
                if self.results[char].dtype in ['float64', 'int64']:
                    corr = self.results[char].corr(self.results[metric])
                    analysis[metric][char] = {'correlation': float(corr) if not pd.isna(corr) else 0.0}

                # Compute mean error by category if categorical
                else:
                    grouped = self.results.groupby(char)[metric].mean()
                    analysis[metric][char] = {
                        k: float(v) for k, v in grouped.to_dict().items()
                    }

        return analysis

    def generate_summary_table(self) -> pd.DataFrame:
        """
        Generate a summary comparison table.

        Returns:
            Summary DataFrame
        """
        if self.results is None:
            raise ValueError("No results loaded.")

        summary = []

        for metric in ['mae', 'rmse', 'mape', 'r2']:
            if metric not in self.results.columns:
                continue

            comp = self.compare_by_metric(metric)
            if 'error' not in comp:
                summary.append({
                    'metric': metric,
                    'arima_mean': comp['arima_mean'],
                    'stgnn_mean': comp['stgnn_mean'],
                    'improvement_pct': comp['improvement_pct'],
                    'significant': comp['statistical_test']['is_significant'],
                    'arima_wins': comp['win_counts']['arima'],
                    'stgnn_wins': comp['win_counts']['stgnn']
                })

        return pd.DataFrame(summary)


def compare_models_by_selector(
    results_path: Path,
    output_path: Optional[Path] = None
) -> Dict:
    """
    Convenience function to compare models by selector type.

    Args:
        results_path: Path to results file
        output_path: Optional path to save comparison

    Returns:
        Comparison dictionary
    """
    comparator = ModelComparator().load_results(results_path)
    comparison = comparator.compare_by_series_type()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Comparison saved to {output_path}")

    return comparison
