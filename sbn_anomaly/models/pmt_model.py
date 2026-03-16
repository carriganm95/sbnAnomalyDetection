"""PMT waveform autoencoder.

Encodes a fixed-length PMT feature vector (e.g. integrated charge, peak
amplitude, arrival-time features per channel) into a latent space and
reconstructs it.  Reconstruction error serves as an anomaly score.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PMTAutoencoder(nn.Module):
    """Fully-connected autoencoder for PMT waveform feature vectors.

    Parameters
    ----------
    input_dim:
        Dimensionality of the input feature vector.
    latent_dim:
        Size of the bottleneck / latent space.
    hidden_dims:
        Sizes of hidden layers in the encoder half.
    dropout:
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 128,
        latent_dim: int = 16,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # ---- Encoder ----
        enc_layers: list[nn.Module] = []
        prev = input_dim
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
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (reconstruction, latent) tuple."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample mean squared reconstruction error (no gradient)."""
        with torch.no_grad():
            x_hat, _ = self.forward(x)
            return ((x - x_hat) ** 2).mean(dim=-1)
