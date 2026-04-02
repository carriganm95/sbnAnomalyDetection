"""TPC autoencoder trainer."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from sbn_anomaly.models.tpc_model import TPCAutoencoder
from sbn_anomaly.train.trainer import BaseTrainer


class TPCTrainer(BaseTrainer):
    """Trainer for :class:`~sbn_anomaly.models.TPCAutoencoder`.

    Parameters
    ----------
    model:
        TPC autoencoder instance.
    lr:
        Learning rate for the Adam optimiser.
    weight_decay:
        L2 regularisation strength.
    device:
        Compute device (``'auto'``, ``'cpu'``, or ``'cuda'``).
    max_epochs:
        Number of training epochs.
    checkpoint_dir:
        Directory to save per-epoch checkpoints.
    log_interval:
        Log loss every *N* batches.
    """

    def __init__(
        self,
        model: Optional[TPCAutoencoder] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "auto",
        max_epochs: int = 50,
        checkpoint_dir: Optional[str] = None,
        log_interval: int = 50,
        steps_per_epoch: Optional[int] = None,
    ) -> None:
        if model is None:
            model = TPCAutoencoder()
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
            steps_per_epoch=steps_per_epoch,
        )
        self.criterion = nn.MSELoss()

    def compute_loss(self, batch: tuple) -> torch.Tensor:
        """MSE reconstruction loss for a TPC batch.

        Expects the DataLoader to yield ``(features,)`` or
        ``(features, labels)`` tuples.
        """
        x = batch[0].to(self.device)
        x_hat, _ = self.model(x)
        return self.criterion(x_hat, x)
