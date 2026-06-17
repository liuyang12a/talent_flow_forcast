"""
Analysis toolkit for time series forecasting experiments.

This package provides tools for:
- Analyzing time series characteristics
- Comparing model performances
- Generating visualizations
- Creating experiment reports
"""

from .series_characteristics import SeriesAnalyzer
from .compare_models import ModelComparator
from .visualizations import (
    plot_model_comparison,
    plot_series_characteristics,
    plot_predictability_analysis,
    plot_prediction_horizon_analysis
)
from .reports import ReportGenerator, generate_experiment_report

__all__ = [
    'SeriesAnalyzer',
    'ModelComparator',
    'plot_model_comparison',
    'plot_series_characteristics',
    'plot_predictability_analysis',
    'plot_prediction_horizon_analysis',
    'ReportGenerator'
]
