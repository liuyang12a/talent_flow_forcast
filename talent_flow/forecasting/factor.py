#!/usr/bin/env python3
"""Dynamic Factor Model (DFM) forecaster.

Flattens the OD matrix to a ``[T, K*K]`` panel, extracts ``r`` common
factors via PCA, fits a low-order VAR on the factors, forecasts, and
reconstructs the OD matrix from the factor loadings.

Purpose-built for "large N (K*K), small T" regimes — exactly the short
OD-matrix forecasting problem. Optionally a BVAR-style ridge can be applied
to the factor VAR; here we use a regularized least-squares VAR (ridge) which
behaves like a shrinkage BVAR point estimate without requiring PyMC.

Refs: Stock & Watson (2002); Bai & Ng (2002).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from talent_flow.core import ForecastResult, ODMatrixSeries
from talent_flow.core.registry import FORECASTER_REGISTRY
from .base import BaseForecaster


@FORECASTER_REGISTRY.register("dfm")
class DFMForecaster(BaseForecaster):
    """Dynamic Factor Model: PCA factors + ridge-regularized VAR.

    Config:
        n_factors: number of common factors ``r``.
        var_lag: VAR order ``p``.
        ridge: L2 ridge penalty on the VAR coefficients (shrinkage).
    """

    name = "dfm"

    def __init__(
        self,
        input_len: int = 12,
        output_len: int = 1,
        n_factors: int = 8,
        var_lag: int = 1,
        ridge: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            input_len,
            output_len,
            n_factors=n_factors,
            var_lag=var_lag,
            ridge=ridge,
            **kwargs,
        )
        self.n_factors = n_factors
        self.var_lag = var_lag
        self.ridge = ridge
        self._mean = None
        self._components = None  # [K*K, r] loadings
        self._var_coef = None  # VAR coefficient matrix
        self._intercept = None
        self._shape = None
        self._last_factors = None

    def fit(self, od_series: ODMatrixSeries, val_series=None):
        M = od_series.matrix  # [T, K, K]
        T, K, _ = M.shape
        self._shape = (K, K)
        panel = M.reshape(T, K * K)  # [T, N]
        # center
        self._mean = panel.mean(axis=0)
        panel_c = panel - self._mean
        # PCA via SVD: panel_c = U S Vt ; factors = U[:, :r] * S[:r], loadings = Vt[:r].T
        U, s, Vt = np.linalg.svd(panel_c, full_matrices=False)
        r = min(self.n_factors, len(s))
        self._components = Vt[:r].T  # [N, r]
        factors = U[:, :r] * s[:r]  # [T, r]
        # fit VAR(p) on factors with ridge
        self._var_coef, self._intercept = self._fit_ridge_var(factors)
        self._last_factors = factors
        self.is_fitted = True
        return self

    def _fit_ridge_var(self, factors: np.ndarray):
        p = self.var_lag
        T, r = factors.shape
        if T <= p:
            # fall back to mean model
            return np.zeros((r, r * p)), factors.mean(axis=0)
        # build design: rows = [1, f_{t-1}, ..., f_{t-p}]
        Y = factors[p:]  # [T-p, r]
        X = np.ones((T - p, 1 + r * p))
        for t in range(p, T):
            row = []
            for lag in range(1, p + 1):
                row.append(factors[t - lag])
            X[t - p, 1:] = np.concatenate(row)
        # ridge: (X^T X + ridge*I) beta = X^T Y ; don't penalize intercept
        XtX = X.T @ X
        reg = self.ridge * np.eye(1 + r * p)
        reg[0, 0] = 0.0
        beta = np.linalg.solve(XtX + reg, X.T @ Y)  # [1+r*p, r]
        intercept = beta[0]
        coef = beta[1:].T  # [r, r*p]
        return coef, intercept

    def predict(self, od_series: ODMatrixSeries) -> ForecastResult:
        if not self.is_fitted:
            raise RuntimeError("DFMForecaster not fitted")
        K, _ = self._shape
        M = od_series.matrix
        T = M.shape[0]
        # project the observed series onto the loadings to get factors
        panel = M.reshape(T, K * K)
        factors = (panel - self._mean) @ self._components  # [T, r]
        # iterative forecast
        p = self.var_lag
        r = factors.shape[1]
        history = list(factors)
        preds_factors = []
        for _ in range(self.output_len):
            x = np.concatenate([history[-lag] for lag in range(1, p + 1)])
            f_next = self._intercept + self._var_coef @ x
            preds_factors.append(f_next)
            history.append(f_next)
        F_pred = np.stack(preds_factors)  # [h, r]
        # reconstruct
        panel_pred = F_pred @ self._components.T + self._mean  # [h, N]
        preds = panel_pred.reshape(self.output_len, K, K)
        gt = M[-self.output_len :] if T >= self.output_len else M
        return ForecastResult(
            predictions=preds,
            ground_truth=gt,
            forecaster_name=self.name,
            metadata={"n_factors": r, "var_lag": p},
        )
