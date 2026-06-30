"""Trainer for the raw-waveform VAE (:class:`TPCWaveformVAE`).

Loss is the standard VAE evidence lower bound: a reconstruction term (MSE) plus
a ``beta``-weighted KL divergence to the standard-normal prior.  ``beta`` can be
linearly warmed up over the first ``beta_warmup_epochs`` epochs to avoid
posterior collapse early in training.

Reuses :class:`~sbn_anomaly.train.trainer.BaseTrainer` for the loop, logging,
checkpointing, history and reconstruction plots.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from sbn_anomaly.models.tpc_waveform_vae import TPCWaveformVAE
from sbn_anomaly.train.trainer import BaseTrainer


class VAETrainer(BaseTrainer):
    """Train a :class:`TPCWaveformVAE` with a beta-VAE objective.

    Parameters
    ----------
    model:
        VAE instance (defaults to a fresh ``TPCWaveformVAE``).
    lr, weight_decay:
        Adam optimiser settings.
    beta:
        Target KL weight.
    beta_warmup_epochs:
        Linearly ramp ``beta`` from 0 to ``beta`` over this many epochs (0 = off).
    score_beta:
        KL weight used when reporting per-sample anomaly scores (defaults to
        ``beta``).
    """

    def __init__(
        self,
        model: Optional[TPCWaveformVAE] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "auto",
        max_epochs: int = 50,
        checkpoint_dir: Optional[str] = None,
        log_interval: int = 50,
        steps_per_epoch: Optional[int] = None,
        anomaly_threshold: Optional[float] = None,
        reconstruction_plot_max_values: int = 50000,
        save_best_only: bool = False,
        use_amp: bool = False,
        beta: float = 1.0,
        beta_warmup_epochs: int = 0,
        score_beta: Optional[float] = None,
    ) -> None:
        if model is None:
            model = TPCWaveformVAE()
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
            anomaly_threshold=anomaly_threshold,
            reconstruction_plot_max_values=reconstruction_plot_max_values,
            save_best_only=save_best_only,
            use_amp=use_amp,
        )
        self.beta = float(beta)
        self.beta_warmup_epochs = int(beta_warmup_epochs)
        self.score_beta = float(score_beta) if score_beta is not None else float(beta)
        self._current_epoch = 0

    # BaseTrainer increments/exposes the epoch via train(); we also track it here
    # so the KL warmup schedule can read it. If BaseTrainer sets self.epoch we use
    # that; otherwise fall back to our own counter updated in compute_loss.
    def _effective_beta(self) -> float:
        if self.beta_warmup_epochs <= 0:
            return self.beta
        epoch = int(getattr(self, "epoch", self._current_epoch))
        frac = min(1.0, (epoch + 1) / float(self.beta_warmup_epochs))
        return self.beta * frac

    def compute_loss(self, batch: tuple) -> torch.Tensor:
        x = batch[0].to(self.device)
        x2 = x if x.dim() == 2 else x.squeeze(1)
        x_hat, mu, logvar, _ = self.model(x2)
        recon = ((x_hat - x2) ** 2).mean()  # mean over batch and ticks
        kl = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        ) / mu.shape[1]
        beta = self._effective_beta()
        return recon + beta * kl

    @torch.no_grad()
    def compute_scores(self, batch: tuple) -> torch.Tensor:
        x = batch[0].to(self.device)
        was_training = self.model.training
        self.model.eval()  # eval: BatchNorm uses running stats (no pollution), z=mu
        try:
            return self.model.anomaly_score(x, beta=self.score_beta)
        finally:
            if was_training:
                self.model.train()

    @torch.no_grad()
    def compute_reconstruction_pair(self, batch: tuple) -> tuple[torch.Tensor, torch.Tensor]:
        x = batch[0].to(self.device)
        x2 = x if x.dim() == 2 else x.squeeze(1)
        was_training = self.model.training
        self.model.eval()
        try:
            x_hat, _, _, _ = self.model(x2)
        finally:
            if was_training:
                self.model.train()
        return x2, x_hat
