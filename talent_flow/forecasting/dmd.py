#!/usr/bin/env python3
"""Dynamic Mode Decomposition (DMD) forecaster.

Recommended first baseline for short, high-dimensional OD-matrix series
(T ~ 120, K ~ 50-100). Flattens the ``[T, K, K]`` OD tensor to snapshot
vectors of size ``K*K`` and fits a best-fit linear operator ``A`` via SVD.
Forecasts are produced by modal extrapolation
``x_{t+h} = sum_j b_j lambda_j^h w_j``.

Refs: Schmid (2010); Kutz et al. (2016).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from talent_flow.core import ForecastResult, ODMatrixSeries
from talent_flow.core.registry import FORECASTER_REGISTRY
from .base import BaseForecaster


@FORECASTER_REGISTRY.register("dmd")
class DMDForecaster(BaseForecaster):
    """DMD on flattened OD snapshots.

    Config:
        rank: truncation rank for the SVD (number of DMD modes). ``None`` or
            ``0`` keeps all modes.
        exact: whether to use exact DMD modes (True) or projected (False).
    """

    name = "dmd"

    def __init__(
        self,
        input_len: int = 12,
        output_len: int = 1,
        rank: Optional[int] = 20,
        exact: bool = True,
        **kwargs,
    ):
        super().__init__(input_len, output_len, rank=rank, exact=exact, **kwargs)
        self.rank = rank
        self.exact = exact
        # fitted state
        self._modes: Optional[np.ndarray] = None
        self._eigenvalues: Optional[np.ndarray] = None
        self._amplitudes: Optional[np.ndarray] = None
        self._shape: Optional[tuple] = None

    def fit(self, od_series: ODMatrixSeries, val_series=None):
        M = od_series.matrix  # [T, K, K]
        T, K, _ = M.shape
        self._shape = (K, K)
        X = M[:-1].reshape(T - 1, K * K).T  # [N, T-1]
        Xp = M[1:].reshape(T - 1, K * K).T  # [N, T-1]
        # SVD of X
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        r = self.rank if (self.rank and self.rank > 0) else len(s)
        r = min(r, len(s))
        U_r, s_r, Vt_r = U[:, :r], s[:r], Vt[:r]
        # reduced operator Atilde = U_r^T Xp Vr Sigma_r^-1  (r x r)
        s_inv = np.diag(1.0 / (s_r + 1e-12))
        Atilde = U_r.T @ Xp @ Vt_r.T @ s_inv
        # eigen-decomposition
        eigvals, W = np.linalg.eig(Atilde)  # (r,), (r,r)
        # modes
        if self.exact:
            modes = Xp @ Vt_r.T @ s_inv @ W  # [N, r]
        else:
            modes = U_r @ W  # [N, r]
        # amplitudes: least-squares fit to first snapshot X[:, 0]
        b, *_ = np.linalg.lstsq(modes, X[:, 0], rcond=None)
        self._modes = modes
        self._eigenvalues = eigvals
        self._amplitudes = b
        self.is_fitted = True
        return self

    def predict(self, od_series: ODMatrixSeries) -> ForecastResult:
        if not self.is_fitted:
            raise RuntimeError("DMDForecaster not fitted")
        K, _ = self._shape
        # evolve from the last observed snapshot
        M = od_series.matrix
        x_last = M[-1].reshape(K * K)
        # re-fit amplitudes to the last snapshot for the forecast origin
        b, *_ = np.linalg.lstsq(self._modes, x_last, rcond=None)
        preds = np.zeros((self.output_len, K, K), dtype=float)
        for h in range(1, self.output_len + 1):
            evolved = self._modes @ (b * (self._eigenvalues ** h))
            preds[h - 1] = evolved.real.reshape(K, K)
        gt = M[-self.output_len :] if M.shape[0] >= self.output_len else M
        return ForecastResult(
            predictions=preds,
            ground_truth=gt,
            forecaster_name=self.name,
            metadata={"rank": int(self._modes.shape[1])},
        )
