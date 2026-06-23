"""Forecasting stage (pluggable OD-matrix forecasters).

Implemented in stage 3. Methods will self-register via
``@FORECASTER_REGISTRY.register("name")``.
"""

from talent_flow.core.registry import FORECASTER_REGISTRY  # noqa: F401

__all__ = ["FORECASTER_REGISTRY"]
