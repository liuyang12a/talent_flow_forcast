"""
Visualization module for experiment results.

This module provides functions to create various plots and charts
for analyzing and presenting experiment results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set default style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_model_comparison(
    results: pd.DataFrame,
    metric: str = 'mae',
    group_by: str = 'selector_type',
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Plot model comparison by a grouping variable.

    Args:
        results: DataFrame with experiment results
        metric: Metric to compare
        group_by: Column to group by
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Filter valid results
    valid_results = results[
        (results[metric].notna()) &
        (results[metric] != float('inf')) &
        (results[metric] < results[metric].quantile(0.99))  # Remove outliers
    ]

    if len(valid_results) == 0:
        logger.warning(f"No valid data for {metric}")
        return fig

    # Box plot
    if group_by in valid_results.columns:
        sns.boxplot(
            data=valid_results,
            x=group_by,
            y=metric,
            hue='model_type',
            ax=axes[0]
        )
        axes[0].set_title(f'{metric.upper()} by {group_by.replace("_", " ").title()}')
        axes[0].tick_params(axis='x', rotation=45)
    else:
        sns.boxplot(
            data=valid_results,
            x='model_type',
            y=metric,
            ax=axes[0]
        )
        axes[0].set_title(f'{metric.upper()} Distribution')

    # Bar plot of means with error bars
    if group_by in valid_results.columns:
        grouped = valid_results.groupby([group_by, 'model_type'])[metric].agg(['mean', 'std']).reset_index()
        pivot_mean = grouped.pivot(index=group_by, columns='model_type', values='mean')
        pivot_std = grouped.pivot(index=group_by, columns='model_type', values='std')

        pivot_mean.plot(kind='bar', yerr=pivot_std, ax=axes[1], capsize=4)
        axes[1].set_title(f'Mean {metric.upper()} Comparison')
        axes[1].tick_params(axis='x', rotation=45)
    else:
        grouped = valid_results.groupby('model_type')[metric].agg(['mean', 'std'])
        grouped['mean'].plot(kind='bar', yerr=grouped['std'], ax=axes[1], capsize=4)
        axes[1].set_title(f'Mean {metric.upper()} Comparison')

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")

    return fig


def plot_series_characteristics(
    characteristics: Dict[str, Dict],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot distribution of series characteristics.

    Args:
        characteristics: Dictionary of series characteristics
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Extract data
    means = [c['statistics']['mean'] for c in characteristics.values()]
    cvs = [c['statistics']['cv'] for c in characteristics.values()]
    trend_strengths = [c['stationarity']['trend_strength'] for c in characteristics.values()]
    seasonal_strengths = [c['seasonality']['seasonal_strength'] for c in characteristics.values()]

    # Mean flow distribution
    axes[0, 0].hist(means, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Mean Flow Distribution')
    axes[0, 0].set_xlabel('Mean Flow')
    axes[0, 0].set_ylabel('Count')

    # Coefficient of variation
    axes[0, 1].hist(cvs, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].set_title('Coefficient of Variation')
    axes[0, 1].set_xlabel('CV')
    axes[0, 1].set_ylabel('Count')

    # Trend strength
    axes[0, 2].hist(trend_strengths, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[0, 2].set_title('Trend Strength Distribution')
    axes[0, 2].set_xlabel('Trend Strength')
    axes[0, 2].set_ylabel('Count')

    # Seasonal strength
    axes[1, 0].hist(seasonal_strengths, bins=30, edgecolor='black', alpha=0.7, color='red')
    axes[1, 0].set_title('Seasonal Strength Distribution')
    axes[1, 0].set_xlabel('Seasonal Strength')
    axes[1, 0].set_ylabel('Count')

    # Scatter: Trend vs Seasonal
    axes[1, 1].scatter(trend_strengths, seasonal_strengths, alpha=0.5)
    axes[1, 1].set_xlabel('Trend Strength')
    axes[1, 1].set_ylabel('Seasonal Strength')
    axes[1, 1].set_title('Trend vs Seasonality')

    # Classification pie chart
    classifications = [c['classification'] for c in characteristics.values()]
    vol_counts = pd.Series([c['volatility'] for c in classifications]).value_counts()
    axes[1, 2].pie(vol_counts.values, labels=vol_counts.index, autopct='%1.1f%%')
    axes[1, 2].set_title('Volatility Classification')

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")

    return fig


def plot_predictability_analysis(
    results: pd.DataFrame,
    characteristics: Dict[str, Dict],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot relationship between series characteristics and predictability.

    Args:
        results: DataFrame with experiment results
        characteristics: Dictionary of series characteristics
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Merge results with characteristics
    char_df = pd.DataFrame.from_dict(characteristics, orient='index')
    char_df['series_id'] = char_df.index

    # Calculate mean error per series for each model
    error_by_series = results.groupby(['series_id', 'model_type'])['mae'].mean().reset_index()
    error_pivot = error_by_series.pivot(index='series_id', columns='model_type', values='mae')

    # Merge with characteristics
    merged = char_df.merge(error_pivot, left_on='series_id', right_index=True, how='left')

    if len(merged) == 0:
        logger.warning("No merged data for predictability analysis")
        return fig

    # Extract values
    cv_values = [c['statistics']['cv'] for c in characteristics.values()]
    trend_values = [c['stationarity']['trend_strength'] for c in characteristics.values()]
    seasonal_values = [c['seasonality']['seasonal_strength'] for c in characteristics.values()]

    # Get mean MAE for each series
    series_ids = list(characteristics.keys())
    mae_values = []
    for sid in series_ids:
        series_errors = error_by_series[error_by_series['series_id'] == sid]['mae']
        if len(series_errors) > 0:
            mae_values.append(series_errors.mean())
        else:
            mae_values.append(np.nan)

    mae_values = np.array(mae_values)
    valid_mask = ~np.isnan(mae_values)

    if valid_mask.sum() == 0:
        logger.warning("No valid MAE values for predictability analysis")
        return fig

    # CV vs MAE
    axes[0, 0].scatter(np.array(cv_values)[valid_mask], mae_values[valid_mask], alpha=0.5)
    axes[0, 0].set_xlabel('Coefficient of Variation')
    axes[0, 0].set_ylabel('MAE')
    axes[0, 0].set_title('Volatility vs Predictability')

    # Trend vs MAE
    axes[0, 1].scatter(np.array(trend_values)[valid_mask], mae_values[valid_mask], alpha=0.5, color='green')
    axes[0, 1].set_xlabel('Trend Strength')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Trend vs Predictability')

    # Seasonality vs MAE
    axes[1, 0].scatter(np.array(seasonal_values)[valid_mask], mae_values[valid_mask], alpha=0.5, color='red')
    axes[1, 0].set_xlabel('Seasonal Strength')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].set_title('Seasonality vs Predictability')

    # Model comparison scatter
    if 'arima' in merged.columns and 'stgnn' in merged.columns:
        arima_mae = merged['arima'].dropna()
        stgnn_mae = merged['stgnn'].dropna()

        if len(arima_mae) > 0 and len(stgnn_mae) > 0:
            min_len = min(len(arima_mae), len(stgnn_mae))
            axes[1, 1].scatter(arima_mae[:min_len], stgnn_mae[:min_len], alpha=0.5, color='purple')
            axes[1, 1].plot([0, max(arima_mae.max(), stgnn_mae.max())],
                          [0, max(arima_mae.max(), stgnn_mae.max())],
                          'r--', label='Equal performance')
            axes[1, 1].set_xlabel('ARIMA MAE')
            axes[1, 1].set_ylabel('STGNN MAE')
            axes[1, 1].set_title('ARIMA vs STGNN Performance')
            axes[1, 1].legend()

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")

    return fig


def plot_prediction_horizon_analysis(
    results: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot how model performance changes with prediction horizon.

    Args:
        results: DataFrame with experiment results
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if 'output_len' not in results.columns:
        logger.warning("No 'output_len' column in results")
        return fig

    metrics = ['mae', 'mape']

    for idx, metric in enumerate(metrics):
        if metric not in results.columns:
            continue

        # Group by output_len and model_type
        grouped = results.groupby(['output_len', 'model_type'])[metric].mean().reset_index()
        pivot = grouped.pivot(index='output_len', columns='model_type', values=metric)

        pivot.plot(kind='line', marker='o', ax=axes[idx])
        axes[idx].set_xlabel('Prediction Horizon (months)')
        axes[idx].set_ylabel(metric.upper())
        axes[idx].set_title(f'{metric.upper()} vs Prediction Horizon')
        axes[idx].legend(title='Model')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")

    return fig


def plot_series_samples(
    series_dict: Dict[str, np.ndarray],
    predictions: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    n_samples: int = 4,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 10)
) -> plt.Figure:
    """
    Plot sample time series with predictions.

    Args:
        series_dict: Dictionary of time series
        predictions: Optional dictionary of predictions by model
        n_samples: Number of series to plot
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n_samples = min(n_samples, len(series_dict))
    sample_ids = np.random.choice(list(series_dict.keys()), n_samples, replace=False)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, series_id in enumerate(sample_ids):
        series = series_dict[series_id]
        ax = axes[idx]

        # Plot actual series
        ax.plot(series, label='Actual', linewidth=2)

        # Plot predictions if available
        if predictions and series_id in predictions:
            for model_name, pred in predictions[series_id].items():
                ax.plot(pred, label=f'{model_name} Pred', linestyle='--', alpha=0.7)

        ax.set_title(f'Series: {series_id}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Flow')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")

    return fig


def create_summary_figure(
    results: pd.DataFrame,
    characteristics: Dict[str, Dict],
    output_dir: Path
) -> List[Path]:
    """
    Create all summary figures for the experiment.

    Args:
        results: DataFrame with experiment results
        characteristics: Dictionary of series characteristics
        output_dir: Directory to save figures

    Returns:
        List of saved figure paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    # Model comparison by selector type
    fig1 = plot_model_comparison(
        results, metric='mae', group_by='selector_type',
        output_path=output_dir / 'model_comparison_mae.png'
    )
    saved_paths.append(output_dir / 'model_comparison_mae.png')
    plt.close(fig1)

    # Model comparison by MAPE
    fig2 = plot_model_comparison(
        results, metric='mape', group_by='selector_type',
        output_path=output_dir / 'model_comparison_mape.png'
    )
    saved_paths.append(output_dir / 'model_comparison_mape.png')
    plt.close(fig2)

    # Series characteristics
    fig3 = plot_series_characteristics(
        characteristics,
        output_path=output_dir / 'series_characteristics.png'
    )
    saved_paths.append(output_dir / 'series_characteristics.png')
    plt.close(fig3)

    # Predictability analysis
    fig4 = plot_predictability_analysis(
        results, characteristics,
        output_path=output_dir / 'predictability_analysis.png'
    )
    saved_paths.append(output_dir / 'predictability_analysis.png')
    plt.close(fig4)

    # Prediction horizon analysis
    if 'output_len' in results.columns and len(results['output_len'].unique()) > 1:
        fig5 = plot_prediction_horizon_analysis(
            results,
            output_path=output_dir / 'prediction_horizon_analysis.png'
        )
        saved_paths.append(output_dir / 'prediction_horizon_analysis.png')
        plt.close(fig5)

    logger.info(f"Created {len(saved_paths)} summary figures")
    return saved_paths
