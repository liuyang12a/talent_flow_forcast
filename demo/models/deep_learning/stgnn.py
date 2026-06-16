"""
Spatial-Temporal Graph Neural Network (STGNN) for time series forecasting.

This module implements a STGNN that combines graph convolution for spatial
dependencies and temporal modeling for temporal patterns.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from demo.models.base_model import BaseDeepLearningModel
from demo.models.deep_learning.layers import (
    GraphConvolution,
    ChebyshevGraphConvolution,
    TemporalConvolution,
    TemporalAttention,
)

logger = logging.getLogger(__name__)


class STGraphEncoder(nn.Module):
    """
    Spatial-Temporal Graph Encoder.

    Stacks multiple spatial-temporal layers, each containing:
    1. Spatial graph convolution (GCN or Chebyshev)
    2. Temporal modeling (TCN or GRU)
    3. Optional attention mechanism
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_nodes: int = None,
        spatial_type: str = "gcn",
        temporal_type: str = "gru",
        chebyshev_k: int = 3,
        dropout: float = 0.0
    ):
        """
        Initialize ST encoder.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of ST layers
            num_nodes: Number of graph nodes
            spatial_type: Type of spatial convolution ('gcn', 'chebyshev')
            temporal_type: Type of temporal model ('gru', 'lstm', 'tcn', 'attention')
            chebyshev_k: Order for Chebyshev convolution
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.spatial_type = spatial_type
        self.temporal_type = temporal_type

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Spatial-temporal layers
        self.spatial_layers = nn.ModuleList()
        self.temporal_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for i in range(num_layers):
            # Spatial layer
            if spatial_type == "gcn":
                spatial = GraphConvolution(hidden_dim, hidden_dim)
            elif spatial_type == "chebyshev":
                spatial = ChebyshevGraphConvolution(
                    hidden_dim, hidden_dim, k=chebyshev_k
                )
            else:
                raise ValueError(f"Unknown spatial type: {spatial_type}")

            self.spatial_layers.append(spatial)

            # Temporal layer
            if temporal_type == "gru":
                temporal = nn.GRU(
                    hidden_dim, hidden_dim,
                    num_layers=1, batch_first=True
                )
            elif temporal_type == "lstm":
                temporal = nn.LSTM(
                    hidden_dim, hidden_dim,
                    num_layers=1, batch_first=True
                )
            elif temporal_type == "tcn":
                temporal = TemporalConvolution(
                    hidden_dim, hidden_dim, kernel_size=3
                )
            elif temporal_type == "attention":
                temporal = TemporalAttention(hidden_dim)
            else:
                raise ValueError(f"Unknown temporal type: {temporal_type}")

            self.temporal_layers.append(temporal)

            # Normalization
            self.norm_layers.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input [batch, time, nodes, features]
            adj: Adjacency matrix [nodes, nodes]

        Returns:
            Output [batch, time, nodes, hidden_dim]
        """
        batch_size, time_steps, num_nodes, _ = x.shape

        # Project input
        x = self.input_proj(x)  # [batch, time, nodes, hidden]

        # Apply ST layers
        for i, (spatial, temporal, norm) in enumerate(
            zip(self.spatial_layers, self.temporal_layers, self.norm_layers)
        ):
            # Spatial convolution (applied per time step)
            x_spatial = []
            for t in range(time_steps):
                h = spatial(x[:, t, :, :], adj)  # [batch, nodes, hidden]
                x_spatial.append(h)
            x = torch.stack(x_spatial, dim=1)  # [batch, time, nodes, hidden]
            x = F.relu(x)
            x = self.dropout(x)

            # Temporal modeling
            if self.temporal_type in ["gru", "lstm"]:
                # Reshape for RNN: [batch*nodes, time, hidden]
                x_flat = x.permute(0, 2, 1, 3).reshape(
                    batch_size * num_nodes, time_steps, self.hidden_dim
                )
                x_temp, _ = temporal(x_flat)
                x = x_temp.reshape(
                    batch_size, num_nodes, time_steps, self.hidden_dim
                ).permute(0, 2, 1, 3)
            elif self.temporal_type == "tcn":
                x_temp = []
                for n in range(num_nodes):
                    h = temporal(x[:, :, n, :])  # [batch, time, hidden]
                    x_temp.append(h)
                x = torch.stack(x_temp, dim=2)
            else:  # attention
                x_temp = []
                for n in range(num_nodes):
                    h = temporal(x[:, :, n, :])  # [batch, time, hidden]
                    x_temp.append(h)
                x = torch.stack(x_temp, dim=2)

            # Residual connection and normalization
            x = norm(x)

        return x


class STGNNModel(BaseDeepLearningModel):
    """
    Spatial-Temporal Graph Neural Network for time series forecasting.

    This model combines graph convolution for capturing spatial relationships
    between nodes with temporal modeling for time series patterns.

    Architecture:
        Input -> ST Encoder -> Output Projection -> Predictions
    """

    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_nodes: int,
        adjacency_matrix: np.ndarray,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        spatial_type: str = "gcn",
        temporal_type: str = "gru",
        output_type: str = "direct",
        dropout: float = 0.0,
        device: str = "auto",
        name: str = "STGNN",
        **kwargs
    ):
        """
        Initialize STGNN model.

        Args:
            input_len: Length of input sequence (history)
            output_len: Length of output sequence (forecast horizon)
            num_nodes: Number of nodes in the graph
            adjacency_matrix: Graph adjacency matrix [num_nodes, num_nodes]
            input_dim: Input feature dimension per node
            hidden_dim: Hidden dimension
            num_layers: Number of ST layers
            spatial_type: Type of spatial convolution ('gcn', 'chebyshev')
            temporal_type: Type of temporal model ('gru', 'lstm', 'tcn', 'attention')
            output_type: How to generate output ('direct' or 'recursive')
            dropout: Dropout rate
            device: Device to use
            name: Model name
            **kwargs: Additional parameters
        """
        super().__init__(input_len, output_len, name, device, **kwargs)

        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_type = output_type

        # Convert and normalize adjacency matrix
        self.adj = self._preprocess_adjacency(adjacency_matrix)

        # Build model
        self.encoder = STGraphEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_nodes=num_nodes,
            spatial_type=spatial_type,
            temporal_type=temporal_type,
            dropout=dropout
        )

        # Output projection
        if output_type == "direct":
            # Direct multi-step prediction
            # Input: [batch, hidden*time, nodes] -> Output: [batch, output*dim, nodes]
            self.output_proj = nn.Conv1d(
                hidden_dim * input_len,
                output_len * input_dim,
                kernel_size=1
            )
        else:
            # Single-step prediction (for recursive forecasting)
            # Input: [batch, hidden, nodes] -> Output: [batch, input_dim, nodes]
            self.output_proj = nn.Conv1d(
                hidden_dim,
                input_dim,
                kernel_size=1
            )

        # Move to device
        self.adj = self.adj.to(self.device)
        self.to(self.device)

        logger.info(
            f"STGNN initialized: {num_nodes} nodes, "
            f"{input_len}->{output_len}, hidden={hidden_dim}"
        )

    def _preprocess_adjacency(self, adj: np.ndarray) -> torch.Tensor:
        """
        Preprocess adjacency matrix for graph convolution.

        Applies symmetric normalization: D^(-1/2) A D^(-1/2)
        """
        # Convert to tensor
        adj = torch.from_numpy(adj).float()

        # Add self-loops
        adj = adj + torch.eye(adj.size(0), device=adj.device)

        # Symmetric normalization
        rowsum = adj.sum(1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)

        adj_normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

        return adj_normalized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input [batch, input_len, num_nodes, input_dim]

        Returns:
            Predictions [batch, output_len, num_nodes, input_dim]
        """
        batch_size = x.size(0)

        # Encode spatial-temporal patterns
        encoded = self.encoder(x, self.adj)  # [batch, input_len, nodes, hidden]

        # Generate predictions
        if self.output_type == "direct":
            # Flatten temporal dimension and permute for Conv1d
            # encoded: [batch, input_len, nodes, hidden]
            encoded_flat = encoded.permute(0, 3, 1, 2)  # [batch, hidden, input_len, nodes]
            encoded_flat = encoded_flat.reshape(batch_size, -1, self.num_nodes)  # [batch, hidden*time, nodes]

            # Project to output
            output = self.output_proj(encoded_flat)  # [batch, output*dim, nodes]

            # Reshape to [batch, output_len, nodes, input_dim]
            output = output.reshape(
                batch_size, self.output_len, self.input_dim, self.num_nodes
            ).permute(0, 1, 3, 2)

        else:
            # Use last time step for recursive prediction
            # last_step: [batch, nodes, hidden]
            last_step = encoded[:, -1, :, :].permute(0, 2, 1)  # [batch, hidden, nodes]
            output = self.output_proj(last_step)  # [batch, input_dim, nodes]
            output = output.permute(0, 2, 1).unsqueeze(1)  # [batch, 1, nodes, input_dim]

        return output

    def _train_loop(
        self,
        train_loader: "DataLoader",
        val_loader: Optional["DataLoader"],
        epochs: int,
        learning_rate: float,
        early_stopping_patience: int,
        **kwargs
    ) -> None:
        """
        Training loop for STGNN.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            learning_rate: Learning rate
            early_stopping_patience: Patience for early stopping
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)

                        output = self.model(batch_x)
                        loss = criterion(output, batch_y)
                        val_loss += loss.item()

                val_loss /= len(val_loader)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs}: "
                        f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                    )
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}")

        self.model = self.model
