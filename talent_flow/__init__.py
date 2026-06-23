"""Talent flow forecasting pipeline package.

Two-stage, loosely-coupled pipeline:

    FlowNetwork dict --pool()--> ODMatrixSeries --fit/predict--> ForecastResult
                                                            |
                                                   evaluation framework

Subpackages
-----------
core        : FlowNetwork data structure + data contracts + registries
data        : raw data loading, preprocessing, FlowNetwork store
pooling     : pluggable network pooling methods
forecasting : pluggable OD-matrix forecasting methods
evaluation  : unified evaluation (pooling quality + forecast accuracy)
pipeline    : two-stage orchestration and persistence
viz         : visualization utilities
utils       : config, io, seeding, logging
"""

__version__ = "0.3.0"
