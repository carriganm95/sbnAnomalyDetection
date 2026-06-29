"""Graph variational autoencoder for window-level TPC anomaly detection.

A reconstruction (not forecasting) model: it encodes a single window's
per-channel feature graph and reconstructs the node features. Trained on good
runs only, so anomalous windows reconstruct poorly -> high per-channel error.
Unlike the GRU forecaster it compares each window to the *absolute* learned
nominal, so persistently-bad states stay flagged (no adapting-away).

Design choices (see project discussion):
- ``SAGEConv`` encoder, which keeps a separate "self" transform, so a single
  anomalous channel is not smoothed away by its electronics neighbours the way
  a plain GCN would.
- Per-node variational bottleneck (mu/logvar) -> smooth latent + a KL anomaly
  term, consistent with the raw-waveform VAE branch.
- Optional masked/denoising reconstruction: a random fraction of nodes have
  their input features zeroed and must be rebuilt from neighbours, which forces
  the model to learn "what should this channel look like given its board-mates"
  and prevents a trivial identity mapping.

Expected PyG ``Data``/``Batch`` fields (from GraphReconDataset):
    x:          (N, 1 + F) — channel_idx column + F node features
    y:          (N, F)     — reconstruction target (clean features)
    edge_index: (2, E)
    batch:      (N,)       — graph id per node (added by the PyG DataLoader)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv, GCNConv
except Exception:  # pragma: no cover - import guarded for non-torch envs
    SAGEConv = GCNConv = None


class GraphVAE(nn.Module):
    """Per-node graph VAE that reconstructs window node features.

    Parameters
    ----------
    in_dim:
        Node feature dimension ``F`` (excluding the channel-index column).
    latent_dim:
        Per-node latent size.
    hidden:
        Hidden width of the message-passing layers.
    enc_layers:
        Number of encoder message-passing layers.
    dec_hidden:
        Hidden width of the (MLP) decoder.
    dropout:
        Dropout in encoder layers.
    mask_ratio:
        Fraction of nodes whose input features are masked during training
        (denoising). 0 disables masking.
    use_channel_idx:
        Concatenate the normalized channel-index column as a conditioning input
        (lets the model learn per-channel baselines).
    conv:
        ``"sage"`` (default, self-preserving) or ``"gcn"``.
    """

    def __init__(
        self,
        in_dim: int,
        latent_dim: int = 12,
        hidden: int = 64,
        enc_layers: int = 2,
        dec_hidden: int = 64,
        dropout: float = 0.1,
        mask_ratio: float = 0.15,
        use_channel_idx: bool = True,
        conv: str = "sage",
    ) -> None:
        super().__init__()
        if SAGEConv is None:
            raise ImportError("torch_geometric is required for GraphVAE")
        self.in_dim = int(in_dim)
        self.latent_dim = int(latent_dim)
        self.dropout = float(dropout)
        self.mask_ratio = float(mask_ratio)
        self.use_channel_idx = bool(use_channel_idx)

        Conv = SAGEConv if conv == "sage" else GCNConv
        enc_in = self.in_dim + (1 if use_channel_idx else 0)

        convs = []
        c_in = enc_in
        for _ in range(max(1, enc_layers)):
            convs.append(Conv(c_in, hidden))
            c_in = hidden
        self.convs = nn.ModuleList(convs)

        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)

        dec_in = latent_dim + (1 if use_channel_idx else 0)
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, dec_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dec_hidden, self.in_dim),
        )

    # ------------------------------------------------------------------
    def _split(self, x: torch.Tensor):
        if self.use_channel_idx:
            return x[:, :1], x[:, 1:]
        return None, x

    def encode(self, feats, channel_idx, edge_index):
        h = torch.cat([channel_idx, feats], dim=1) if self.use_channel_idx else feats
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            if self.dropout > 0 and self.training:
                h = F.dropout(h, p=self.dropout, training=True)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def decode(self, z, channel_idx):
        h = torch.cat([channel_idx, z], dim=1) if self.use_channel_idx else z
        return self.decoder(h)

    def forward(self, data):
        channel_idx, feats = self._split(data.x)
        edge_index = data.edge_index

        feats_in = feats
        if self.training and self.mask_ratio > 0:
            mask = torch.rand(feats.shape[0], device=feats.device) < self.mask_ratio
            feats_in = feats.clone()
            feats_in[mask] = 0.0

        mu, logvar = self.encode(feats_in, channel_idx, edge_index)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, channel_idx)
        return x_hat, mu, logvar, z

    # ------------------------------------------------------------------
    @torch.no_grad()
    def reconstruction_error(self, data) -> torch.Tensor:
        """Per-node mean squared reconstruction error ``(N,)``."""
        _, feats = self._split(data.x)
        x_hat, _, _, _ = self.forward(data)
        target = getattr(data, "y", feats)
        return ((x_hat - target) ** 2).mean(dim=-1)
