"""PyTorch Geometric GNN forecaster model.

Uses PyG's GCNConv layers for message passing and a GRU temporal encoder
for sequence-to-sequence prediction on graph-structured data.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GNNForecasterPyG(nn.Module):
    """PyG-based GNN + temporal GRU forecaster for next-window prediction.

    Accepts a PyG Batch produced by torch_geometric.loader.DataLoader.
    For each time step in the history, applies stacked GCN layers to aggregate
    spatial node features, then runs a GRU across time to predict the next window.

    Expected Data object fields (per graph):
        x:          (N, 1 + T*frame_feat_dim) — channel_idx followed by T frames of F features
        y:          (N, target_dim) — prediction target
        edge_index: (2, E) sparse edges in COO format

    Args:
        frame_feat_dim: feature dimension per time step F (excluding channel index)
        target_dim: output feature dimension per node
        gnn_hidden: hidden dimension for GCN layers
        gnn_layers: number of stacked GCN layers
        gru_hidden: hidden dimension for the GRU
        gru_layers: number of stacked GRU layers (dropout applied between layers when > 1)
        history: number of past time steps T
        dropout: dropout probability applied after each GCN layer
    """

    def __init__(
        self,
        frame_feat_dim: int,
        target_dim: int,
        gnn_hidden: int = 64,
        gnn_layers: int = 2,
        gru_hidden: int = 128,
        gru_layers: int = 1,
        history: int = 4,
        dropout: float = 0.1,
        norm_type: str = "none",
    ) -> None:
        super().__init__()
        self.frame_feat_dim = frame_feat_dim
        self.gnn_hidden = gnn_hidden
        self.gnn_layers = gnn_layers
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self.history = history
        self.dropout = dropout
        self.target_dim = int(target_dim)
        self.norm_type = norm_type

        gcn_in = 1 + frame_feat_dim
        if norm_type == "batch":
            self.input_norm: nn.Module | None = nn.BatchNorm1d(gcn_in)
        elif norm_type == "layer":
            self.input_norm = nn.LayerNorm(gcn_in)
        else:
            self.input_norm = None

        # GCN input per time step: channel_idx (1) + per-frame features
        gcn_layers = []
        for _ in range(gnn_layers):
            gcn_layers.append(GCNConv(gcn_in, gnn_hidden))
            gcn_in = gnn_hidden
        self.gcn_layers = nn.ModuleList(gcn_layers)

        # Projection layer: compress GCN output to match GRU input
        # If gnn_hidden != gru_hidden, add a linear projection
        if gnn_hidden != gru_hidden:
            self.gnn_to_gru = nn.Linear(gnn_hidden, gru_hidden)
        else:
            self.gnn_to_gru = None

        # Dropout applies between GRU layers, so only meaningful when gru_layers > 1
        self.gru = nn.GRU(
            input_size=gru_hidden,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=False,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        self.decoder = nn.Linear(gru_hidden, self.target_dim)

    def _encode_frame(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Apply stacked GCN layers to a single time frame.

        Args:
            x: (total_nodes, 1 + frame_feat_dim)
            edge_index: (2, E)

        Returns:
            h: (total_nodes, gnn_hidden)
        """
        h = self.input_norm(x) if self.input_norm is not None else x
        for gcn in self.gcn_layers:
            h = gcn(h, edge_index)
            h = F.relu(h)
            if self.dropout > 0 and self.training:
                h = F.dropout(h, p=self.dropout, training=True)
        return h

    def forward(self, data) -> torch.Tensor:
        """Predict next window for a PyG batched graph.

        Args:
            data: PyG Batch object from torch_geometric.loader.DataLoader.
                  Must have x (total_nodes, 1+T*F), edge_index (2, E),
                  and batch (total_nodes,).

        Returns:
            pred: (total_nodes, target_dim)
        """
        x = data.x             # (total_nodes, 1 + T*F)
        edge_index = data.edge_index

        channel_idx = x[:, :1]   # (total_nodes, 1)
        temporal = x[:, 1:]      # (total_nodes, T*F)

        T = self.history
        t_f = temporal.shape[1]
        if t_f % T != 0:
            raise ValueError(
                f"Temporal feature dim {t_f} is not divisible by history {T}. "
                f"Expected x of shape (N, 1 + {T}*F) for integer F."
            )
        F = t_f // T
        temporal = temporal.view(-1, T, F)  # (total_nodes, T, F)

        enc_slices = []
        for t in range(T):
            x_t = torch.cat([channel_idx, temporal[:, t, :]], dim=1)  # (total_nodes, 1+F)
            h_t = self._encode_frame(x_t, edge_index)                  # (total_nodes, gnn_hidden)
            enc_slices.append(h_t)

        enc_seq = torch.stack(enc_slices, dim=0)  # (T, total_nodes, gnn_hidden)
        
        # Project GCN output to GRU input dimension if needed
        if self.gnn_to_gru is not None:
            enc_seq = self.gnn_to_gru(enc_seq)  # (T, total_nodes, gru_hidden)
        
        out_seq, _ = self.gru(enc_seq)             # (T, total_nodes, gru_hidden)
        pred = self.decoder(out_seq[-1])            # (total_nodes, target_dim)
        return pred
