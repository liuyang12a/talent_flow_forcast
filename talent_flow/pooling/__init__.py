"""Pluggable network pooling methods.

Every pooler subclasses :class:`BasePooler` and implements
:meth:`build_assignment`; the generic :meth:`pool` flow (aggregation via
``S^T A S`` + intrinsic quality evaluation) is shared. Methods self-register
via ``@POOLER_REGISTRY.register("name")`` on import.
"""

from talent_flow.core.registry import POOLER_REGISTRY
from .base import BasePooler
from .assignment import build_hard_assignment, aggregated_adjacency

# Importing the modules below triggers registration of their poolers.
from . import semantic  # noqa: F401
from . import truncation  # noqa: F401
from . import community  # noqa: F401
from . import core_periphery  # noqa: F401
from . import dense_subgraph  # noqa: F401

__all__ = [
    "POOLER_REGISTRY",
    "BasePooler",
    "build_hard_assignment",
    "aggregated_adjacency",
]
