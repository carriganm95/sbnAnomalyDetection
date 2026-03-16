"""1-D convolutional autoencoder for sliding-window time-series anomaly detection.

``WindowAutoencoder`` is designed to operate on short temporal windows of raw
waveform samples (TPC or PMT).  The encoder uses strided convolutions; the
decoder uses transposed convolutions to reconstruct the original window.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class WindowAutoencoder(nn.Module):
    """1-D convolutional autoencoder for waveform window anomaly detection.

    Parameters
    ----------
    window_size:
        Number of time samples per input window.
    n_channels:
        Number of input channels (e.g. number of PMT channels).
    latent_dim:
        Flattened latent dimension after the convolutional encoder.
    base_filters:
        Number of filters in the first conv layer; doubled at each block.
    dropout:
        Dropout probability applied after each encoder block.
    """

    def __init__(
        self,
        window_size: int = 256,
        n_channels: int = 1,
        latent_dim: int = 64,
        base_filters: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.n_channels = n_channels
        self.latent_dim = latent_dim

        # ---- Encoder: three strided conv blocks ----
        f1, f2, f3 = base_filters, base_filters * 2, base_filters * 4
        self.enc_conv = nn.Sequential(
            nn.Conv1d(n_channels, f1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(f1, f2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(f2, f3, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        # Compute flattened size after convolutions.
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, window_size)
            conv_out = self.enc_conv(dummy)
            self._conv_out_shape = conv_out.shape[1:]  # (C, L)
            flat_dim = conv_out.numel()

        self.enc_fc = nn.Linear(flat_dim, latent_dim)

        # ---- Decoder ----
        self.dec_fc = nn.Linear(latent_dim, flat_dim)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose1d(f3, f2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(f2, f1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(f1, n_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode and decode a batch of windows.

        Parameters
        ----------
        x:
            Tensor of shape (B, n_channels, window_size).

        Returns
        -------
        x_hat:
            Reconstruction of shape (B, n_channels, window_size).
        z:
            Latent vector of shape (B, latent_dim).
        """
        # Encoder
        h = self.enc_conv(x)
        z = self.enc_fc(h.flatten(start_dim=1))

        # Decoder
        h_dec = self.dec_fc(z).view(-1, *self._conv_out_shape)
        x_hat = self.dec_conv(h_dec)
        # Trim/pad to match input length exactly.
        x_hat = x_hat[..., : self.window_size]
        return x_hat, z

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample mean MSE across channels and time (no gradient)."""
        with torch.no_grad():
            x_hat, _ = self.forward(x)
            return ((x - x_hat) ** 2).mean(dim=(-2, -1))
