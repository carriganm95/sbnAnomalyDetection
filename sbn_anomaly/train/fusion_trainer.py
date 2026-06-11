"""Fusion autoencoder trainer."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from sbn_anomaly.models.fusion_model import FusionAutoencoder
from sbn_anomaly.train.trainer import BaseTrainer


class FusionTrainer(BaseTrainer):
    """Trainer for :class:`~sbn_anomaly.models.FusionAutoencoder`.

    The fusion model is trained on matched (TPC, PMT) event pairs produced by
    :class:`~sbn_anomaly.data.EventJoiner`.

    Parameters
    ----------
    model:
        Fusion autoencoder instance.
    lr:
        Learning rate.
    weight_decay:
        L2 regularisation.
    device:
        Compute device.
    max_epochs:
        Training epochs.
    checkpoint_dir:
        Checkpoint save directory.
    log_interval:
        Logging frequency in batches.
    """

    def __init__(
        self,
        model: Optional[FusionAutoencoder] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "auto",
        max_epochs: int = 50,
        checkpoint_dir: Optional[str] = None,
        log_interval: int = 50,
        anomaly_threshold: Optional[float] = None,
        reconstruction_plot_max_values: int = 50000,
        save_best_only: bool = False,
    ) -> None:
        if model is None:
            model = FusionAutoencoder()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        super().__init__(
            model=model,
            optimizer=optimizer,
            device=device,
            max_epochs=max_epochs,
            checkpoint_dir=checkpoint_dir,
            log_interval=log_interval,
            anomaly_threshold=anomaly_threshold,
            reconstruction_plot_max_values=reconstruction_plot_max_values,
            save_best_only=save_best_only,
        )
        self.criterion = nn.MSELoss()

    def compute_loss(self, batch: tuple) -> torch.Tensor:
        """MSE reconstruction loss in the combined (TPC + PMT) feature space.

        Expects the DataLoader to yield ``(x_tpc, x_pmt)`` or
        ``(x_tpc, x_pmt, labels)`` tuples (from :class:`FusionDataset`).
        """
        x_tpc = batch[0].to(self.device)
        x_pmt = batch[1].to(self.device)
        recon, _, combined = self.model(x_tpc, x_pmt)
        return self.criterion(recon, combined)

    def compute_scores(self, batch: tuple) -> torch.Tensor:
        """Per-sample reconstruction MSE in the combined feature space."""
        x_tpc = batch[0].to(self.device)
        x_pmt = batch[1].to(self.device)
        return self.model.reconstruction_error(x_tpc, x_pmt)

    def compute_reconstruction_pair(
        self,
        batch: tuple,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return combined input and reconstruction tensors for plotting."""
        x_tpc = batch[0].to(self.device)
        x_pmt = batch[1].to(self.device)
        with torch.no_grad():
            recon, _, combined = self.model(x_tpc, x_pmt)
        return combined, recon
