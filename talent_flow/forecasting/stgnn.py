#!/usr/bin/env python3
"""Spatial-Temporal GNN forecaster for OD-matrix series.

Adapts the legacy STGNN architecture (graph convolution + GRU temporal) to
the new OD-matrix contract. Each super-node is represented by its (in-flow,
out-flow) features at each time step; the (time-summed) OD matrix serves as
the graph adjacency. Fixes the legacy ``self.model = self.model`` no-op bug
by having the forecaster *be* the ``nn.Module`` and using a clean fit/predict
loop.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from talent_flow.core import ForecastResult, ODMatrixSeries
from talent_flow.core.registry import FORECASTER_REGISTRY
from .base import BaseForecaster
from .windowing import make_windows


@FORECASTER_REGISTRY.register("stgnn")
class STGNNForecaster(BaseForecaster):
    """STGNN over super-nodes with (in-flow, out-flow) node features.

    Config:
        hidden_dim, num_layers, spatial_type, temporal_type, dropout: model
            hyperparameters.
        epochs, batch_size, learning_rate, patience: training hyperparameters.
        device: ``"auto"`` / ``"cuda"`` / ``"cpu"``.
    """

    name = "stgnn"

    def __init__(
        self,
        input_len: int = 12,
        output_len: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 2,
        spatial_type: str = "gcn",
        temporal_type: str = "gru",
        dropout: float = 0.1,
        epochs: int = 100,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        patience: int = 10,
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(
            input_len,
            output_len,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            spatial_type=spatial_type,
            temporal_type=temporal_type,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            **kwargs,
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.spatial_type = spatial_type
        self.temporal_type = temporal_type
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.device = device
        self._net = None
        self._adj = None
        self._K = None

    def _resolve_device(self):
        import torch

        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    @staticmethod
    def _od_to_node_features(M: np.ndarray) -> np.ndarray:
        """``[T, K, K]`` OD -> ``[T, K, 2]`` node features (in-flow, out-flow)."""
        out_flow = M.sum(axis=2)  # [T, K]
        in_flow = M.sum(axis=1)  # [T, K]
        return np.stack([out_flow, in_flow], axis=-1)  # [T, K, 2]

    def fit(self, od_series: ODMatrixSeries, val_series=None):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        from talent_flow.forecasting.layers import (
            GraphConvolution,
            TemporalConvolution,
            TemporalAttention,
        )

        M = od_series.matrix
        T, K, _ = M.shape
        self._K = K
        # adjacency = time-summed OD (symmetric-normalized inside the layer)
        adj = M.sum(axis=0)
        # symmetrize (undirected graph convolution)
        adj = (adj + adj.T) * 0.5
        self._adj = adj

        # Build windows of OD matrices, then derive node features for inputs
        # and keep OD as targets.
        X_od, y_od, _ = make_windows(od_series, self.input_len, self.output_len)
        if X_od.shape[0] == 0:
            raise ValueError("not enough timesteps to build training windows")
        # X node features: [n, in_len, K, 2]
        n = X_od.shape[0]
        X_feat = np.zeros((n, self.input_len, K, 2), dtype=np.float32)
        for i in range(n):
            X_feat[i] = self._od_to_node_features(X_od[i])  # [in_len, K, 2]
        y_t_np = y_od.astype(np.float32)  # [n, out_len, K, K]

        device = self._resolve_device()

        net = _build_stgnn_net_class()(
            input_dim=2,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_nodes=K,
            output_len=self.output_len,
            spatial_type=self.spatial_type,
            temporal_type=self.temporal_type,
            dropout=self.dropout,
        ).to(device)
        adj_t = torch.from_numpy(adj).float().to(device)

        X_t = torch.from_numpy(X_feat).float().to(device)
        y_t = torch.from_numpy(y_t_np).float().to(device)
        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        opt = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        crit = nn.MSELoss()
        best_state = None
        best_loss = float("inf")
        patience = 0
        for epoch in range(self.epochs):
            net.train()
            total = 0.0
            for bx, by in loader:
                opt.zero_grad()
                out = net(bx, adj_t)  # [n, out, K, K]
                loss = crit(out, by)
                loss.backward()
                opt.step()
                total += loss.item()
            avg = total / max(len(loader), 1)
            if avg < best_loss - 1e-6:
                best_loss = avg
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self._net = net
        self.is_fitted = True
        return self

    def predict(self, od_series: ODMatrixSeries) -> ForecastResult:
        import torch

        if not self.is_fitted:
            raise RuntimeError("STGNNForecaster not fitted")
        device = self._resolve_device()
        M = od_series.matrix
        x = M[-self.input_len :]  # [in, K, K]
        feats = self._od_to_node_features(x)[None]  # [1, in, K, 2]
        x_t = torch.from_numpy(feats).float().to(device)
        adj_t = torch.from_numpy(self._adj).float().to(device)
        with torch.no_grad():
            out = self._net(x_t, adj_t)  # [1, out, K, K]
        preds = out.squeeze(0).cpu().numpy()
        gt = M[-self.output_len :] if M.shape[0] >= self.output_len else M
        return ForecastResult(
            predictions=preds,
            ground_truth=gt,
            forecaster_name=self.name,
        )


# ---- the torch Module (built lazily so torch is optional) ----

def _build_stgnn_net_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from talent_flow.forecasting.layers import (
        GraphConvolution,
        ChebyshevGraphConvolution,
        TemporalConvolution,
        TemporalAttention,
    )

    class STGNNNet(nn.Module):
        def __init__(
            self,
            input_dim,
            hidden_dim,
            num_layers,
            num_nodes,
            output_len,
            spatial_type="gcn",
            temporal_type="gru",
            dropout=0.0,
        ):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.output_len = output_len
            self.num_nodes = num_nodes
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.spatial_layers = nn.ModuleList()
            self.temporal_layers = nn.ModuleList()
            self.norms = nn.ModuleList()
            for _ in range(num_layers):
                if spatial_type == "gcn":
                    self.spatial_layers.append(GraphConvolution(hidden_dim, hidden_dim))
                elif spatial_type == "chebyshev":
                    self.spatial_layers.append(
                        ChebyshevGraphConvolution(hidden_dim, hidden_dim, k=3)
                    )
                else:
                    raise ValueError(spatial_type)
                if temporal_type == "gru":
                    self.temporal_layers.append(
                        nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
                    )
                elif temporal_type == "lstm":
                    self.temporal_layers.append(
                        nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
                    )
                elif temporal_type == "tcn":
                    self.temporal_layers.append(
                        TemporalConvolution(hidden_dim, hidden_dim, kernel_size=3)
                    )
                elif temporal_type == "attention":
                    self.temporal_layers.append(TemporalAttention(hidden_dim))
                else:
                    raise ValueError(temporal_type)
                self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropout = nn.Dropout(dropout)
            # Bilinear output head producing the OD matrix per output step.
            self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, x, adj):
            # x: [batch, time, nodes, features]
            b, t_in, n, _ = x.shape
            h = self.input_proj(x)  # [b, t, n, hid]
            for spatial, temporal, norm in zip(
                self.spatial_layers, self.temporal_layers, self.norms
            ):
                hs = []
                for t in range(t_in):
                    hs.append(spatial(h[:, t, :, :], adj))
                h = torch.stack(hs, dim=1)
                h = F.relu(h)
                h = self.dropout(h)
                if isinstance(temporal, (nn.GRU, nn.LSTM)):
                    hf = h.permute(0, 2, 1, 3).reshape(b * n, t_in, self.hidden_dim)
                    out, _ = temporal(hf)
                    h = out.reshape(b, n, t_in, self.hidden_dim).permute(0, 2, 1, 3)
                else:
                    outs = []
                    for nn_i in range(n):
                        outs.append(temporal(h[:, :, nn_i, :]))
                    h = torch.stack(outs, dim=2)
                h = norm(h)
            last = self.out_proj(h[:, -1, :, :])  # [b, n, hid]
            gram = torch.matmul(last, last.transpose(1, 2))  # [b, n, n]
            gram = gram.unsqueeze(1).expand(b, self.output_len, n, n)
            return gram

    return STGNNNet
