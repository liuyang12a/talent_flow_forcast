"""
Graph convolution layers for STGNN.

This module implements various graph convolution operations used in
spatial-temporal graph neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphConvolution(nn.Module):
    """
    Basic Graph Convolution Layer (GCN).

    Implements the propagation rule:
        H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))

    where A is the adjacency matrix, D is the degree matrix,
    H is the node features, and W is the learnable weight matrix.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True
    ):
        """
        Initialize GCN layer.

        Args:
            in_features: Size of input features
            out_features: Size of output features
            bias: Whether to use bias term
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [batch, nodes, in_features]
            adj: Adjacency matrix [nodes, nodes] or [batch, nodes, nodes]

        Returns:
            Output features [batch, nodes, out_features]
        """
        # Linear transformation
        support = torch.matmul(x, self.weight)

        # Graph convolution: adj @ support
        if adj.dim() == 2:
            output = torch.matmul(adj, support)
        else:
            output = torch.bmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        return output


class ChebyshevGraphConvolution(nn.Module):
    """
    Chebyshev Graph Convolution.

    Uses Chebyshev polynomials of the first kind to approximate
    spectral graph convolutions without explicit eigendecomposition.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        k: int = 3,
        bias: bool = True
    ):
        """
        Initialize Chebyshev convolution.

        Args:
            in_features: Size of input features
            out_features: Size of output features
            k: Order of Chebyshev polynomial (number of hops)
            bias: Whether to use bias term
        """
        super().__init__()
        self.k = k
        self.weights = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(in_features, out_features))
            for _ in range(k)
        ])

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        for weight in self.weights:
            stdv = 1. / np.sqrt(weight.size(1))
            weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            stdv = 1. / np.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using Chebyshev polynomials.

        Args:
            x: Node features [batch, nodes, in_features]
            adj: Scaled Laplacian [nodes, nodes]

        Returns:
            Output features [batch, nodes, out_features]
        """
        batch_size, num_nodes, _ = x.shape

        # Chebyshev polynomial terms
        cheb_polynomials = [torch.eye(num_nodes, device=x.device), adj]

        for i in range(2, self.k):
            cheb_next = 2 * torch.matmul(adj, cheb_polynomials[-1]) - cheb_polynomials[-2]
            cheb_polynomials.append(cheb_next)

        # Weighted sum
        output = torch.matmul(cheb_polynomials[0], x) @ self.weights[0]
        for i in range(1, self.k):
            output += torch.matmul(cheb_polynomials[i], x) @ self.weights[i]

        if self.bias is not None:
            output += self.bias

        return output


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT).

    Implements attention mechanism over graph neighbors to weight
    their contributions differently.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.0,
        alpha: float = 0.2,
        concat: bool = True
    ):
        """
        Initialize GAT layer.

        Args:
            in_features: Size of input features
            out_features: Size of output features
            dropout: Dropout rate
            alpha: LeakyReLU negative slope
            concat: Whether to concatenate multi-head outputs
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.a = nn.Parameter(torch.FloatTensor(2 * out_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [batch, nodes, in_features]
            adj: Adjacency matrix [nodes, nodes]

        Returns:
            Output features [batch, nodes, out_features]
        """
        batch_size, num_nodes, _ = x.shape

        # Linear transformation
        h = torch.matmul(x, self.W)  # [batch, nodes, out_features]

        # Compute attention coefficients
        # Expand for pairwise attention
        h_i = h.unsqueeze(2).expand(-1, -1, num_nodes, -1)  # [batch, nodes, nodes, out]
        h_j = h.unsqueeze(1).expand(-1, num_nodes, -1, -1)  # [batch, nodes, nodes, out]

        # Concatenate and compute attention
        a_input = torch.cat([h_i, h_j], dim=-1)  # [batch, nodes, nodes, 2*out]
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(-1), self.alpha)

        # Mask attention with adjacency matrix
        mask = adj.unsqueeze(0).expand(batch_size, -1, -1)
        e = e.masked_fill(mask == 0, float('-inf'))

        # Softmax normalization
        attention = F.softmax(e, dim=-1)
        attention = F.dropout(attention, p=self.dropout, training=self.training)

        # Apply attention
        output = torch.bmm(attention, h)

        return output


class TemporalConvolution(nn.Module):
    """
    Temporal Convolution for time series modeling.

    Can use either 1D convolutions or recurrent layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1
    ):
        """
        Initialize temporal convolution.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            dilation: Dilation rate for dilated convolutions
        """
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input [batch, time, features]

        Returns:
            Output [batch, time, out_channels]
        """
        # Conv1d expects [batch, channels, time]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)

        # Remove extra padding
        if self.conv.padding[0] > 0:
            x = x[:, :-self.conv.padding[0], :]

        return x


class TemporalAttention(nn.Module):
    """
    Temporal self-attention for capturing long-range temporal dependencies.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize temporal attention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input [batch, time, features]
            mask: Optional attention mask

        Returns:
            Output [batch, time, features]
        """
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm(x + self.dropout(attn_output))
        return x
