#!/usr/bin/env python3
"""
Plugin registries for poolers, forecasters, and evaluators.

A method is registered via the ``@<REGISTRY>.register("name")`` decorator and
constructed via ``<REGISTRY>.build("name", **params)``. Adding a method only
requires writing a new module that imports the registry and decorates its
class — no edits to existing core code.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Type


class Registry:
    """A simple name -> class registry with decorator and factory APIs."""

    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Type[Any]] = {}

    @property
    def name(self) -> str:
        return self._name

    def register(self, alias: str) -> Callable[[Type[Any]], Type[Any]]:
        """Class decorator registering ``cls`` under ``alias``."""

        def _decorator(cls: Type[Any]) -> Type[Any]:
            if alias in self._registry:
                raise ValueError(
                    f"{self._name}: '{alias}' already registered to "
                    f"{self._registry[alias].__name__}"
                )
            self._registry[alias] = cls
            return cls

        return _decorator

    def build(self, alias: str, **params: Any) -> Any:
        """Instantiate the registered class for ``alias`` with ``params``."""
        if alias not in self._registry:
            available = ", ".join(sorted(self._registry)) or "(empty)"
            raise KeyError(
                f"{self._name}: '{alias}' not registered. Available: {available}"
            )
        return self._registry[alias](**params)

    def get(self, alias: str) -> Type[Any]:
        """Return the registered class (without instantiating)."""
        if alias not in self._registry:
            raise KeyError(f"{self._name}: '{alias}' not registered")
        return self._registry[alias]

    def available(self) -> list[str]:
        """List of registered aliases."""
        return sorted(self._registry.keys())

    def __contains__(self, alias: str) -> bool:
        return alias in self._registry


# Module-level singletons. Importing these submodules (pooling.*, forecasting.*)
# triggers registration of their classes via the decorators.
POOLER_REGISTRY = Registry("pooler")
FORECASTER_REGISTRY = Registry("forecaster")
