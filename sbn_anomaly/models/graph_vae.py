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
    graph_attr: (num_graphs, graph_dim) — optional per-window conditioning
                vector (e.g. event count), only required when the model is
                constructed with graph_dim > 0 (see SparseWindowDatasetPyG's
                graph_features).
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
    encoder_hidden_dims:
        Hidden dimensions for the graph encoder.
        Number of encoder graph layers = len(encoder_hidden_dims).
        Example: [128, 64, 32] builds enc_in -> 128 -> 64 -> 32.
    decoder_hidden_dims:
        Hidden dimensions for the decoder MLP.
        Number of decoder hidden layers = len(decoder_hidden_dims).
        Example: [64, 32] builds latent -> 64 -> 32 -> output.
        Use [] for a direct linear decoder from latent to output.
    dropout:
        Dropout in encoder layers.
    mask_ratio:
        Fraction of nodes whose input features are masked during training
        (denoising). 0 disables masking.
    use_channel_idx:
        Concatenate the normalized channel-index column as a conditioning input
        (lets the model learn per-channel baselines).
    graph_dim:
        Size of the per-graph (per-window) conditioning vector, e.g. event
        count / trigger rate for that window (see
        ``SparseWindowDatasetPyG(graph_features=...)``). 0 (default) disables
        it entirely -- existing configs/checkpoints are unaffected. When > 0,
        every ``Data`` the model sees must carry a ``graph_attr`` tensor of
        shape ``(1, graph_dim)`` (``(num_graphs, graph_dim)`` after PyG
        batching); it's broadcast to every node in its graph (via
        ``data.batch``) and concatenated into both the encoder and decoder
        inputs, the same way ``channel_idx`` already is -- letting the model
        learn "what a channel should look like *given this window's rate*"
        instead of forcing rate-driven variation to be explained away as
        per-channel anomaly. Conditioning both encode and decode (not just
        decode) means the latent code itself can factor out rate-driven
        variation rather than smuggling it into z.
    conv:
        ``"sage"`` (default, self-preserving) or ``"gcn"``.
    """

    def __init__(
        self,
        in_dim: int,
        latent_dim: int = 12,
        encoder_hidden_dims: list[int] | tuple[int, ...] = (64, 64),
        decoder_hidden_dims: list[int] | tuple[int, ...] = (64,),
        dropout: float = 0.1,
        mask_ratio: float = 0.15,
        use_channel_idx: bool = True,
        graph_dim: int = 0,
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
        self.graph_dim = int(graph_dim)
        if self.graph_dim < 0:
            raise ValueError(f"graph_dim must be >= 0, got {graph_dim}")

        if conv == "sage":
            Conv = SAGEConv
        elif conv == "gcn":
            Conv = GCNConv
        else:
            raise ValueError(f"conv must be 'sage' or 'gcn', got {conv!r}")

        def _positive_int_list(
            name: str,
            values: list[int] | tuple[int, ...],
            *,
            allow_empty: bool = False,
        ) -> list[int]:
            dims = [int(v) for v in values]
            if not allow_empty and len(dims) == 0:
                raise ValueError(f"{name} must be non-empty")
            if any(d <= 0 for d in dims):
                raise ValueError(f"{name} must contain positive integers, got {dims}")
            return dims

        self.encoder_hidden_dims = _positive_int_list(
            "encoder_hidden_dims",
            encoder_hidden_dims,
            allow_empty=False,
        )
        self.decoder_hidden_dims = _positive_int_list(
            "decoder_hidden_dims",
            decoder_hidden_dims,
            allow_empty=True,
        )

        enc_in = self.in_dim + (1 if self.use_channel_idx else 0) + self.graph_dim

        # Variable-width graph encoder.
        # Example: encoder_hidden_dims = [128, 64, 32]
        # Encoder: enc_in -> 128 -> 64 -> 32
        convs: list[nn.Module] = []
        c_in = enc_in
        for hidden_dim in self.encoder_hidden_dims:
            convs.append(Conv(c_in, hidden_dim))
            c_in = hidden_dim
        self.convs = nn.ModuleList(convs)

        encoder_out_dim = self.encoder_hidden_dims[-1]
        self.fc_mu = nn.Linear(encoder_out_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(encoder_out_dim, self.latent_dim)

        dec_in = self.latent_dim + (1 if self.use_channel_idx else 0) + self.graph_dim

        # Variable-width decoder MLP.
        # Example: decoder_hidden_dims = [64, 32]
        # Decoder: dec_in -> 64 -> 32 -> in_dim
        decoder_layers: list[nn.Module] = []
        c_in = dec_in
        for hidden_dim in self.decoder_hidden_dims:
            decoder_layers.append(nn.Linear(c_in, hidden_dim))
            decoder_layers.append(nn.ReLU(inplace=True))
            c_in = hidden_dim
        decoder_layers.append(nn.Linear(c_in, self.in_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    # ------------------------------------------------------------------
    def _split(self, x: torch.Tensor):
        if self.use_channel_idx:
            return x[:, :1], x[:, 1:]
        return None, x

    def _broadcast_graph_cond(self, data, num_nodes: int):
        """Return the per-graph conditioning vector broadcast to (num_nodes, graph_dim).

        ``data.graph_attr`` is (num_graphs, graph_dim) after PyG batching (or
        (1, graph_dim) for a single un-batched Data). ``data.batch`` maps each
        node to its graph id; a raw single Data has no ``batch`` attribute, in
        which case every node belongs to graph 0.
        """
        graph_attr = getattr(data, "graph_attr", None)
        if graph_attr is None:
            raise ValueError(
                f"GraphVAE was constructed with graph_dim={self.graph_dim} but this "
                "batch has no 'graph_attr' attribute. Build the dataset with "
                "data.graph_features set so every window carries the expected "
                "per-graph conditioning vector, or construct the model with "
                "graph_dim=0 to disable conditioning."
            )
        if graph_attr.shape[-1] != self.graph_dim:
            raise ValueError(
                f"graph_attr has width {graph_attr.shape[-1]} but the model expects "
                f"graph_dim={self.graph_dim}; data.graph_features must match the "
                "model config exactly (same list, same order)."
            )
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=graph_attr.device)
        return graph_attr[batch]

    def encode(self, feats, channel_idx, edge_index, graph_cond=None):
        parts = [channel_idx, feats] if self.use_channel_idx else [feats]
        if self.graph_dim > 0:
            parts.append(graph_cond)
        h = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
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

    def decode(self, z, channel_idx, graph_cond=None):
        parts = [channel_idx, z] if self.use_channel_idx else [z]
        if self.graph_dim > 0:
            parts.append(graph_cond)
        h = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        return self.decoder(h)

    def forward(self, data):
        channel_idx, feats = self._split(data.x)
        edge_index = data.edge_index

        graph_cond = self._broadcast_graph_cond(data, feats.shape[0]) if self.graph_dim > 0 else None

        feats_in = feats
        if self.training and self.mask_ratio > 0:
            mask = torch.rand(feats.shape[0], device=feats.device) < self.mask_ratio
            feats_in = feats.clone()
            feats_in[mask] = 0.0

        mu, logvar = self.encode(feats_in, channel_idx, edge_index, graph_cond)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, channel_idx, graph_cond)
        return x_hat, mu, logvar, z

    # ------------------------------------------------------------------
    @torch.no_grad()
    def reconstruction_error(self, data) -> torch.Tensor:
        """Per-node mean squared reconstruction error ``(N,)``."""
        _, feats = self._split(data.x)
        x_hat, _, _, _ = self.forward(data)
        target = getattr(data, "y", feats)
        return ((x_hat - target) ** 2).mean(dim=-1)
