#!/usr/bin/env python3
"""Statistical significance tests for comparing methods."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def paired_t_test(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Paired two-sided t-test on per-sample metric values ``a`` vs ``b``."""
    from scipy import stats

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    t_stat, p_value = stats.ttest_rel(a[:n], b[:n])
    return {"t_statistic": float(t_stat), "p_value": float(p_value), "n": int(n)}


def wilcoxon(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Wilcoxon signed-rank test on paired samples."""
    from scipy import stats

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    diff = a[:n] - b[:n]
    diff = diff[diff != 0]
    if len(diff) < 1:
        return {"statistic": 0.0, "p_value": 1.0, "n": int(n)}
    stat, p_value = stats.wilcoxon(diff)
    return {"statistic": float(stat), "p_value": float(p_value), "n": int(n)}


class SignificanceTester:
    """Convenience wrapper dispatching by test name."""

    def __init__(self, test: str = "wilcoxon"):
        if test not in ("wilcoxon", "paired_t"):
            raise ValueError(f"unknown test: {test}")
        self.test = test

    def compare(self, a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
        if self.test == "wilcoxon":
            return wilcoxon(a, b)
        return paired_t_test(a, b)
