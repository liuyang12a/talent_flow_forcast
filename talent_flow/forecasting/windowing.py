#!/usr/bin/env python3
"""Windowing / split helpers for OD-matrix time series.

Shared by all forecasters so that train/val/test partitioning is consistent
across methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from talent_flow.core import ODMatrixSeries


@dataclass
class SplitRatios:
    train: float = 0.7
    val: float = 0.15
    test: float = 0.15

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if not np.isclose(total, 1.0):
            raise ValueError(f"split ratios must sum to 1.0, got {total}")


def split_od_series(
    series: ODMatrixSeries,
    ratios: Optional[SplitRatios] = None,
) -> Tuple[ODMatrixSeries, ODMatrixSeries, ODMatrixSeries]:
    """Split an :class:`ODMatrixSeries` into (train, val, test) by time."""
    ratios = ratios or SplitRatios()
    T = series.T
    n_train = int(round(T * ratios.train))
    n_val = int(round(T * ratios.val))
    # remainder -> test
    n_test = T - n_train - n_val
    if n_test <= 0:
        raise ValueError(f"non-positive test size: train={n_train} val={n_val} T={T}")
    train = series.slice(0, n_train)
    val = series.slice(n_train, n_train + n_val)
    test = series.slice(n_train + n_val, T)
    return train, val, test


def make_windows(
    series: ODMatrixSeries,
    input_len: int,
    output_len: int,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Build sliding-window (X, y) pairs from an OD-matrix series.

    Returns:
        X: ``[n_windows, input_len, K, K]``
        y: ``[n_windows, output_len, K, K]``
        start_indices: list of window start time indices
    """
    M = series.matrix  # [T, K, K]
    T, K, _ = M.shape
    X, y, starts = [], [], []
    for start in range(0, T - input_len - output_len + 1, stride):
        X.append(M[start : start + input_len])
        y.append(M[start + input_len : start + input_len + output_len])
        starts.append(start)
    if not X:
        return (
            np.empty((0, input_len, K, K)),
            np.empty((0, output_len, K, K)),
            [],
        )
    return np.stack(X), np.stack(y), starts
