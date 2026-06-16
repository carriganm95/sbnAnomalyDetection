"""1-D convolutional variational autoencoder for raw TPC ADC waveforms.

Each channel's pre-processed waveform ``(n_ticks,)`` is encoded to a low
dimensional latent ``z`` (e.g. 16-32 dims).  The latent becomes the GNN node
feature; the reconstruction error and KL term provide unsupervised per-channel
anomaly scores on their own.

A *convolutional* (not fully-connected) encoder preserves temporal structure,
and the *variational* bottleneck gives a smooth, regularised latent space that
behaves better for downstream density-based anomaly scoring than a plain AE.

Shape contract
--------------
``forward(x)`` accepts ``x`` of shape ``(B, L)`` or ``(B, 1, L)`` and returns
``(x_hat, mu, logvar, z)`` with ``x_hat`` of shape ``(B, L)``.  ``L`` must be
divisible by ``2**depth``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TPCWaveformVAE(nn.Module):
    """Conv1d VAE for single-channel waveforms.

    Parameters
    ----------
    input_length:
        Number of ADC ticks per waveform.  Must be divisible by ``2**depth``.
    latent_dim:
        Bottleneck size (per-channel compressed representation / GNN node feat).
    base_channels:
        Channel width of the first conv layer; doubles each layer.
    depth:
        Number of stride-2 conv (down/up) layers.  Each halves/doubles length.
    kernel_size:
        Conv kernel size (odd).
    dropout:
        Dropout probability in conv blocks.
    """

    def __init__(
        self,
        input_length: int = 4096,
        latent_dim: int = 24,
        base_channels: int = 16,
        depth: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_length % (2 ** depth) != 0:
            raise ValueError(
                f"input_length ({input_length}) must be divisible by 2**depth "
                f"(2**{depth} = {2 ** depth})."
            )
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        self.input_length = int(input_length)
        self.latent_dim = int(latent_dim)
        self.depth = int(depth)
        self.base_channels = int(base_channels)
        pad = kernel_size // 2

        # ---- Encoder: depth x (Conv1d stride2 -> BN -> ReLU) ----
        enc: list[nn.Module] = []
        in_c = 1
        chans = []
        for d in range(depth):
            out_c = base_channels * (2 ** d)
            chans.append(out_c)
            enc.append(nn.Conv1d(in_c, out_c, kernel_size, stride=2, padding=pad))
            enc.append(nn.BatchNorm1d(out_c))
            enc.append(nn.ReLU(inplace=True))
            if dropout > 0:
                enc.append(nn.Dropout(dropout))
            in_c = out_c
        self.encoder = nn.Sequential(*enc)

        self.enc_out_channels = in_c
        self.enc_out_length = input_length // (2 ** depth)
        flat = self.enc_out_channels * self.enc_out_length

        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat)

        # ---- Decoder: depth x (ConvTranspose1d stride2 -> BN -> ReLU) ----
        dec: list[nn.Module] = []
        rev = list(reversed(chans))  # e.g. [128, 64, 32, 16]
        for d in range(depth):
            in_dc = rev[d]
            out_dc = rev[d + 1] if d + 1 < depth else base_channels
            dec.append(
                nn.ConvTranspose1d(
                    in_dc, out_dc, kernel_size, stride=2,
                    padding=pad, output_padding=1,
                )
            )
            dec.append(nn.BatchNorm1d(out_dc))
            dec.append(nn.ReLU(inplace=True))
            if dropout > 0:
                dec.append(nn.Dropout(dropout))
        self.decoder = nn.Sequential(*dec)
        self.head = nn.Conv1d(base_channels, 1, kernel_size, padding=pad)

    # ------------------------------------------------------------------
    @staticmethod
    def _as_bcl(x: torch.Tensor) -> torch.Tensor:
        """Coerce input to (B, 1, L)."""
        if x.dim() == 2:
            return x.unsqueeze(1)
        if x.dim() == 3:
            return x
        raise ValueError(f"expected (B, L) or (B, 1, L), got shape {tuple(x.shape)}")

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu, logvar)``."""
        h = self.encoder(self._as_bcl(x))
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z)
        h = h.view(-1, self.enc_out_channels, self.enc_out_length)
        h = self.decoder(h)
        x_hat = self.head(h)            # (B, 1, L)
        return x_hat.squeeze(1)         # (B, L)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z

    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_latents(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic latent (the mean ``mu``) for downstream node features."""
        mu, _ = self.encode(x)
        return mu

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample mean-squared reconstruction error ``(B,)``."""
        x2 = x if x.dim() == 2 else x.squeeze(1)
        x_hat, _, _, _ = self.forward(x2)
        return ((x2 - x_hat) ** 2).mean(dim=1)

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """Per-sample anomaly score = recon MSE + ``beta`` * per-sample KL ``(B,)``."""
        x2 = x if x.dim() == 2 else x.squeeze(1)
        x_hat, mu, logvar, _ = self.forward(x2)
        recon = ((x2 - x_hat) ** 2).mean(dim=1)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl = kl / mu.shape[1]  # per-latent-dim, comparable scale to recon
        return recon + beta * kl
