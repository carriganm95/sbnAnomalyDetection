"""Autoencoder architectures for the SBN anomaly detection pipeline.

Models
------
TPCAutoencoder      – dense AE for per-event TPC feature vectors
PMTAutoencoder      – dense AE for per-event PMT feature vectors
FusionAutoencoder   – AE operating on concatenated TPC + PMT latents
WindowAutoencoder   – AE for fixed-length sliding windows (1-D conv)

All models expose a common interface:
    forward(x) -> (reconstruction, latent)
    reconstruction_loss(x) -> scalar tensor
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper: build an MLP encoder / decoder
# ---------------------------------------------------------------------------

def _make_mlp(
    in_dim: int,
    layer_dims: List[int],
    activation: nn.Module = None,
    add_final_activation: bool = True,
) -> nn.Sequential:
    """Build a sequential MLP from in_dim through layer_dims."""
    if activation is None:
        activation = nn.ReLU()
    layers: List[nn.Module] = []
    prev = in_dim
    for i, dim in enumerate(layer_dims):
        layers.append(nn.Linear(prev, dim))
        is_last = i == len(layer_dims) - 1
        if not is_last or add_final_activation:
            layers.append(type(activation)())
        prev = dim
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Dense autoencoder (shared base)
# ---------------------------------------------------------------------------

class _DenseAutoencoder(nn.Module):
    """Generic dense autoencoder used by TPC and PMT models."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
    ) -> None:
        super().__init__()
        enc_dims = hidden_dims + [latent_dim]
        dec_dims = list(reversed(hidden_dims)) + [input_dim]

        self.encoder = _make_mlp(input_dim, enc_dims, add_final_activation=False)
        self.decoder = _make_mlp(latent_dim, dec_dims, add_final_activation=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        recon, _ = self.forward(x)
        return F.mse_loss(recon, x)


# ---------------------------------------------------------------------------
# TPC Autoencoder
# ---------------------------------------------------------------------------

class TPCAutoencoder(_DenseAutoencoder):
    """Dense autoencoder for TPC feature vectors.

    Parameters
    ----------
    input_dim:
        Number of TPC features per event.
    hidden_dims:
        List of hidden layer widths for the encoder (mirrored for decoder).
    latent_dim:
        Bottleneck dimension.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        latent_dim: int = 32,
    ) -> None:
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        super().__init__(input_dim, hidden_dims, latent_dim)


# ---------------------------------------------------------------------------
# PMT Autoencoder
# ---------------------------------------------------------------------------

class PMTAutoencoder(_DenseAutoencoder):
    """Dense autoencoder for PMT feature vectors.

    Parameters
    ----------
    input_dim:
        Number of PMT features per event.
    hidden_dims:
        List of hidden layer widths for the encoder (mirrored for decoder).
    latent_dim:
        Bottleneck dimension.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        latent_dim: int = 32,
    ) -> None:
        if hidden_dims is None:
            hidden_dims = [128, 64]
        super().__init__(input_dim, hidden_dims, latent_dim)


# ---------------------------------------------------------------------------
# Fusion Autoencoder
# ---------------------------------------------------------------------------

class FusionAutoencoder(nn.Module):
    """Autoencoder that operates on the concatenation of TPC and PMT latents.

    The encoder takes ``[z_tpc || z_pmt]`` and compresses to a joint latent.
    The decoder reconstructs the concatenated latent space (not raw features).

    Parameters
    ----------
    tpc_latent_dim, pmt_latent_dim:
        Dimensions of the frozen upstream latent codes.
    hidden_dims:
        Encoder / decoder hidden widths.
    latent_dim:
        Fusion bottleneck dimension.
    """

    def __init__(
        self,
        tpc_latent_dim: int,
        pmt_latent_dim: int,
        hidden_dims: List[int] = None,
        latent_dim: int = 64,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        input_dim = tpc_latent_dim + pmt_latent_dim
        enc_dims = hidden_dims + [latent_dim]
        dec_dims = list(reversed(hidden_dims)) + [input_dim]
        self.encoder = _make_mlp(input_dim, enc_dims, add_final_activation=False)
        self.decoder = _make_mlp(latent_dim, dec_dims, add_final_activation=False)

    def encode(self, z_tpc: torch.Tensor, z_pmt: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_tpc, z_pmt], dim=-1)
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, z_tpc: torch.Tensor, z_pmt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(z_tpc, z_pmt)
        recon = self.decode(z)
        target = torch.cat([z_tpc, z_pmt], dim=-1)
        return recon, z, target

    def reconstruction_loss(
        self, z_tpc: torch.Tensor, z_pmt: torch.Tensor
    ) -> torch.Tensor:
        recon, _, target = self.forward(z_tpc, z_pmt)
        return F.mse_loss(recon, target)


# ---------------------------------------------------------------------------
# Window Autoencoder (1-D convolutional)
# ---------------------------------------------------------------------------

class WindowAutoencoder(nn.Module):
    """1-D convolutional autoencoder for sliding windows of events.

    Input shape: ``(batch, window_size, n_features)``
    The module transposes to ``(batch, n_features, window_size)`` before
    passing through the convolutional layers.

    Parameters
    ----------
    n_features:
        Number of features per time-step.
    window_size:
        Number of events in each window.
    channels:
        List of output channels for successive Conv1d layers in the encoder.
    latent_dim:
        Size of the flattened latent representation.
    kernel_size:
        Kernel width for all convolutional layers.
    """

    def __init__(
        self,
        n_features: int,
        window_size: int = 64,
        channels: List[int] = None,
        latent_dim: int = 128,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [32, 64]

        self.n_features = n_features
        self.window_size = window_size
        self.latent_dim = latent_dim
        self._channels = channels

        # --- Encoder ---
        enc_layers: List[nn.Module] = []
        in_ch = n_features
        for out_ch in channels:
            enc_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
            ]
            in_ch = out_ch
        self.conv_encoder = nn.Sequential(*enc_layers)

        # Compute flattened size after convolutions
        self._conv_out_size = in_ch * window_size
        self.fc_enc = nn.Linear(self._conv_out_size, latent_dim)

        # --- Decoder ---
        self.fc_dec = nn.Linear(latent_dim, self._conv_out_size)
        dec_layers: List[nn.Module] = []
        rev_channels = list(reversed(channels))
        in_ch = rev_channels[0]
        for out_ch in rev_channels[1:] + [n_features]:
            dec_layers += [
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
            ]
            in_ch = out_ch
        # Replace last ReLU with identity (unbounded output)
        dec_layers[-1] = nn.Identity()
        self.conv_decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) → (B, F, T)
        h = x.permute(0, 2, 1)
        h = self.conv_encoder(h)
        h = h.flatten(1)
        return self.fc_enc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z)
        rev_channels = list(reversed(self._channels))
        h = h.view(h.size(0), rev_channels[0], self.window_size)
        h = self.conv_decoder(h)
        # h: (B, F, T) → (B, T, F)
        return h.permute(0, 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        recon, _ = self.forward(x)
        return F.mse_loss(recon, x)
