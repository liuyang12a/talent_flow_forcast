"""Forecasting stage: pluggable OD-matrix forecasters.

Every forecaster subclasses :class:`BaseForecaster` and implements
:meth:`fit`/:meth:`predict`; methods self-register via
``@FORECASTER_REGISTRY.register("name")`` on import.
"""

from talent_flow.core.registry import FORECASTER_REGISTRY
from .base import BaseForecaster
from .windowing import SplitRatios, split_od_series, make_windows

# importing these registers their forecasters
from . import naive  # noqa: F401
from . import dmd  # noqa: F401
from . import factor  # noqa: F401

# ARIMA / STGNN require optional deps; register only if importable.
try:
    from . import arima  # noqa: F401
except ImportError:
    pass
try:
    from . import stgnn  # noqa: F401
except ImportError:
    pass

__all__ = [
    "FORECASTER_REGISTRY",
    "BaseForecaster",
    "SplitRatios",
    "split_od_series",
    "make_windows",
]
