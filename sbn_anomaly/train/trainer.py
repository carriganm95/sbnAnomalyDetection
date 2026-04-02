"""Base trainer class with common training loop logic."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Abstract base trainer providing a standard training loop.

    Subclasses must implement :meth:`compute_loss`.

    Parameters
    ----------
    model:
        PyTorch model to train.
    optimizer:
        Optimiser instance.
    device:
        ``'cuda'``, ``'cpu'``, or ``'auto'``.
    max_epochs:
        Number of training epochs.
    checkpoint_dir:
        Directory to save model checkpoints.  ``None`` disables saving.
    log_interval:
        Log training loss every N batches.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "auto",
        max_epochs: int = 50,
        checkpoint_dir: Optional[str] = None,
        log_interval: int = 50,
        steps_per_epoch: Optional[int] = None,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.log_interval = log_interval
        self.steps_per_epoch = steps_per_epoch

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def compute_loss(self, batch: tuple) -> torch.Tensor:
        """Compute scalar loss for one batch."""

    def train(self, loader: DataLoader) -> list[float]:
        """Run training for ``max_epochs`` epochs.

        Parameters
        ----------
        loader:
            DataLoader yielding batches consumed by :meth:`compute_loss`.

        Returns
        -------
        epoch_losses:
            Mean loss per epoch.
        """
        self.model.train()
        epoch_losses: list[float] = []

        for epoch in range(1, self.max_epochs + 1):
            running_loss = 0.0
            n_batches = 0
            for batch_idx, batch in enumerate(loader):
                if self.steps_per_epoch is not None and batch_idx >= self.steps_per_epoch:
                    break
                self.optimizer.zero_grad()
                loss = self.compute_loss(batch)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                n_batches += 1

                if (batch_idx + 1) % self.log_interval == 0:
                    logger.info(
                        "Epoch %d/%d  batch %d  loss=%.6f",
                        epoch,
                        self.max_epochs,
                        batch_idx + 1,
                        loss.item(),
                    )

            mean_loss = running_loss / max(n_batches, 1)
            epoch_losses.append(mean_loss)
            logger.info("Epoch %d/%d  mean_loss=%.6f", epoch, self.max_epochs, mean_loss)

            if self.checkpoint_dir:
                ckpt_path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": mean_loss,
                    },
                    ckpt_path,
                )
                logger.debug("Saved checkpoint: %s", ckpt_path)

        return epoch_losses

    def save(self, path: str) -> None:
        """Save model weights to *path*."""
        torch.save(self.model.state_dict(), path)
        logger.info("Model weights saved to %s", path)

    def load(self, path: str) -> None:
        """Load model weights from *path*."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        logger.info("Model weights loaded from %s", path)
