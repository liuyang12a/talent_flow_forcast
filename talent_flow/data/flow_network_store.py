#!/usr/bin/env python3
"""
Disk store for monthly :class:`FlowNetwork` objects.

Reads pickled ``FlowNetwork`` instances from ``datasets/flow_networks/*.pkl``
(filenames ``YYYY-MM.pkl``), with date-range filtering and caching.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Dict, Optional

from talent_flow.core import FlowNetwork

_MONTH_RE = re.compile(r"^(\d{4})-(\d{2})\.pkl$")


def _month_key(month_str: str) -> tuple[int, int]:
    """``"2017-03"`` -> ``(2017, 3)`` for chronological comparison."""
    year, month = month_str.split("-")
    return int(year), int(month)


def _in_range(month_str: str, start: Optional[str], end: Optional[str]) -> bool:
    k = _month_key(month_str)
    if start is not None and k < _month_key(start):
        return False
    if end is not None and k > _month_key(end):
        return False
    return True


class FlowNetworkStore:
    """Load and cache monthly :class:`FlowNetwork` objects from disk."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"flow_networks dir not found: {self.data_dir}")
        self._cache: Dict[str, FlowNetwork] = {}

    def list_months(self, start: Optional[str] = None, end: Optional[str] = None) -> list[str]:
        """Return chronologically sorted month strings within ``[start, end]``."""
        months = []
        for p in self.data_dir.glob("*.pkl"):
            m = _MONTH_RE.match(p.name)
            if m is None:
                continue
            month_str = f"{m.group(1)}-{m.group(2)}"
            if _in_range(month_str, start, end):
                months.append(month_str)
        months.sort(key=_month_key)
        return months

    def load(self, month: str) -> FlowNetwork:
        """Load a single month's network (cached)."""
        if month in self._cache:
            return self._cache[month]
        path = self.data_dir / f"{month}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"network file not found: {path}")
        with path.open("rb") as f:
            net = pickle.load(f)
        if not isinstance(net, FlowNetwork):
            raise TypeError(
                f"{path} contains {type(net).__name__}, expected FlowNetwork"
            )
        self._cache[month] = net
        return net

    def load_range(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Dict[str, FlowNetwork]:
        """Load all months in ``[start, end]`` as an ordered ``{month: net}`` dict."""
        return {m: self.load(m) for m in self.list_months(start, end)}

    def aggregate(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> FlowNetwork:
        """Return the time-aggregated network (sum of all months in range)."""
        from talent_flow.core.flow_network import merge_networks

        networks = list(self.load_range(start, end).values())
        return merge_networks(networks)

    def clear_cache(self) -> None:
        self._cache.clear()
