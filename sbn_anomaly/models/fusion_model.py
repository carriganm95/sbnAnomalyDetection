"""Fusion autoencoder combining TPC and PMT latent representations.

The fusion model can be trained in two ways:
  1. **Jointly** – raw TPC and PMT feature vectors are concatenated and passed
     through a single autoencoder.
  2. **Late fusion** – frozen TPC and PMT encoders produce latent vectors that
     are concatenated and refined by a small fusion network.

This module implements approach (1) by default.  Late fusion is available via
``FusionAutoencoder(mode='late', tpc_encoder=..., pmt_encoder=...)``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from sbn_anomaly.models.tpc_model import TPCAutoencoder
from sbn_anomaly.models.pmt_model import PMTAutoencoder


class FusionAutoencoder(nn.Module):
    """Joint TPC + PMT autoencoder for multimodal anomaly detection.

    Parameters
    ----------
    tpc_input_dim:
        Dimensionality of raw TPC feature vectors.
    pmt_input_dim:
        Dimensionality of raw PMT feature vectors.
    latent_dim:
        Size of the shared bottleneck latent space.
    hidden_dims:
        Hidden layer sizes in the encoder half (decoder is mirrored).
    dropout:
        Dropout probability.
    mode:
        ``'joint'`` (default) – concatenate raw features before encoding.
        ``'late'``  – use frozen TPC/PMT encoders; only the fusion neck is
        trained.  Requires ``tpc_encoder`` and ``pmt_encoder`` to be supplied.
    tpc_encoder:
        Pre-trained :class:`TPCAutoencoder` (only used when ``mode='late'``).
    pmt_encoder:
        Pre-trained :class:`PMTAutoencoder` (only used when ``mode='late'``).
    """

    def __init__(
        self,
        tpc_input_dim: int = 256,
        pmt_input_dim: int = 128,
        latent_dim: int = 32,
        hidden_dims: tuple[int, ...] = (192, 96),
        dropout: float = 0.1,
        mode: str = "joint",
        tpc_encoder: Optional[TPCAutoencoder] = None,
        pmt_encoder: Optional[PMTAutoencoder] = None,
    ) -> None:
        super().__init__()
        if mode not in ("joint", "late"):
            raise ValueError(f"mode must be 'joint' or 'late', got '{mode}'.")
        self.mode = mode
        self.tpc_input_dim = tpc_input_dim
        self.pmt_input_dim = pmt_input_dim

        if mode == "late":
            if tpc_encoder is None or pmt_encoder is None:
                raise ValueError("mode='late' requires tpc_encoder and pmt_encoder.")
            self.tpc_encoder = tpc_encoder
            self.pmt_encoder = pmt_encoder
            for p in self.tpc_encoder.parameters():
                p.requires_grad = False
            for p in self.pmt_encoder.parameters():
                p.requires_grad = False
            combined_dim = tpc_encoder.latent_dim + pmt_encoder.latent_dim
        else:
            self.tpc_encoder = None  # type: ignore[assignment]
            self.pmt_encoder = None  # type: ignore[assignment]
            combined_dim = tpc_input_dim + pmt_input_dim

        # ---- Encoder ----
        enc_layers: list[nn.Module] = []
        prev = combined_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # ---- Decoder ----
        dec_layers: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        dec_layers.append(nn.Linear(prev, combined_dim))
        self.decoder = nn.Sequential(*dec_layers)

        self.combined_dim = combined_dim

    def forward(
        self,
        x_tpc: torch.Tensor,
        x_pmt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (reconstruction, latent, combined_input) tuple.

        The reconstruction is in the concatenated feature space; callers can
        split it back using ``tpc_input_dim`` and ``pmt_input_dim``.
        """
        if self.mode == "late":
            with torch.no_grad():
                _, z_tpc = self.tpc_encoder(x_tpc)
                _, z_pmt = self.pmt_encoder(x_pmt)
            combined = torch.cat([z_tpc, z_pmt], dim=-1)
        else:
            combined = torch.cat([x_tpc, x_pmt], dim=-1)

        z = self.encoder(combined)
        recon = self.decoder(z)
        return recon, z, combined

    def reconstruction_error(
        self, x_tpc: torch.Tensor, x_pmt: torch.Tensor
    ) -> torch.Tensor:
        """Per-sample MSE in the combined feature space (no gradient)."""
        with torch.no_grad():
            recon, _, combined = self.forward(x_tpc, x_pmt)
            return ((combined - recon) ** 2).mean(dim=-1)
