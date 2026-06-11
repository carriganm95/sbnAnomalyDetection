from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGraphConv(nn.Module):
    """Very small graph convolution implemented with dense adjacency.

    x' = W1 x + W2 (A x)
    """

    def __init__(self, in_feats: int, out_feats: int):
        super().__init__()
        self.lin_self = nn.Linear(in_feats, out_feats)
        self.lin_neigh = nn.Linear(in_feats, out_feats)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F) or (N, F) ; adj: (N, N)
        was_batched = x.dim() == 3
        if not was_batched:
            x = x.unsqueeze(0)
        # aggregate neighbor messages: (B, N, F)
        neigh = torch.matmul(adj.to(x.device), x)
        out = self.lin_self(x) + self.lin_neigh(neigh)
        out = F.relu(out)
        if not was_batched:
            out = out.squeeze(0)
        return out


class GNNForecaster(nn.Module):
    """GNN + temporal GRU forecaster for next-window prediction.

    Input: sequence of past windows shape (B, T, N, F)
    Output: predicted next window shape (B, N, F)
    """

    def __init__(
        self,
        node_feat_dim: int,
        gnn_hidden: int = 64,
        gnn_layers: int = 2,
        gru_hidden: int = 128,
        history: int = 4,
    ) -> None:
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.gnn_hidden = gnn_hidden
        self.gnn_layers = gnn_layers
        self.gru_hidden = gru_hidden
        self.history = history

        # GNN encoder per time slice
        convs: list[nn.Module] = []
        in_f = node_feat_dim
        for i in range(gnn_layers):
            convs.append(SimpleGraphConv(in_f, gnn_hidden))
            in_f = gnn_hidden
        self.convs = nn.ModuleList(convs)

        # Pool node features into per-node embeddings for temporal model
        # We'll keep node dimension and run GRU across time per-node after flattening features.
        # GRU expects (seq_len, batch, input_size). We'll treat batch as (B*N)
        self.gru = nn.GRU(input_size=gnn_hidden, hidden_size=gru_hidden, batch_first=False)

        # decoder: map GRU hidden -> node feature prediction
        self.dec = nn.Linear(gru_hidden, node_feat_dim)

    def encode_one(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Encode single window: x shape (B, N, F) -> (B, N, gnn_hidden)"""
        h = x
        for conv in self.convs:
            h = conv(h, adj)
        return h

    def forward(self, past: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Predict next window.

        past: (B, T, N, F)
        adj: (N, N) or (B, N, N)
        returns: (B, N, F_pred)
        """
        B, T, N, F = past.shape
        device = past.device
        # Encode each time slice
        enc_slices = []
        for t in range(T):
            x_t = past[:, t, :, :]
            h_t = self.encode_one(x_t, adj)
            enc_slices.append(h_t)
        # Stack -> (B, T, N, H)
        enc = torch.stack(enc_slices, dim=1)
        # Prepare for GRU: reshape to (T, B*N, H)
        enc_perm = enc.permute(1, 0, 2, 3).contiguous()
        enc_flat = enc_perm.view(T, B * N, -1)
        # Run GRU
        out_seq, h_n = self.gru(enc_flat)  # out_seq: (T, B*N, Hg)
        last = out_seq[-1]  # (B*N, Hg)
        pred_flat = self.dec(last)  # (B*N, F_pred)
        pred = pred_flat.view(B, N, -1)
        return pred
