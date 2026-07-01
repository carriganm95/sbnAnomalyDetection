"""Trainer for the graph VAE (window reconstruction)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from sbn_anomaly.train.trainer import BaseTrainer


class GraphVAETrainer(BaseTrainer):
    """Train :class:`~sbn_anomaly.models.graph_vae.GraphVAE` on good-run windows.

    Loss is the beta-VAE ELBO: standardized reconstruction MSE + ``beta`` * KL,
    with optional linear KL warmup over ``beta_warmup_epochs``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "auto",
        max_epochs: int = 50,
        checkpoint_dir: Optional[str] = None,
        log_interval: int = 50,
        anomaly_threshold: Optional[float] = None,
        save_best_only: bool = False,
        use_amp: bool = False,
        score_mode: str = "mean",
        beta: float = 1.0,
        beta_warmup_epochs: int = 0,
    ) -> None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        super().__init__(
            model=model,
            optimizer=optimizer,
            device=device,
            max_epochs=max_epochs,
            checkpoint_dir=checkpoint_dir,
            log_interval=log_interval,
            anomaly_threshold=anomaly_threshold,
            save_best_only=save_best_only,
            use_amp=use_amp,
            score_mode=score_mode,
        )
        self.beta = float(beta)
        self.beta_warmup_epochs = int(beta_warmup_epochs)
        # Running recon/KL accumulators for per-epoch logging.
        self._recon_sum = 0.0
        self._kl_sum = 0.0
        self._term_batches = 0

    def _infer_batch_size(self, batch) -> int:
        return int(getattr(batch, "num_graphs", 1))

    def _effective_beta(self) -> float:
        if self.beta_warmup_epochs <= 0:
            return self.beta
        epoch = int(getattr(self, "epoch", 1))
        return self.beta * min(1.0, epoch / float(self.beta_warmup_epochs))

    def compute_loss(self, batch) -> torch.Tensor:
        data = batch.to(self.device)
        x_hat, mu, logvar, _ = self.model(data)
        recon = ((x_hat - data.y.float()) ** 2).mean()
        kl = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        ) / max(1, mu.shape[1])
        # Accumulate the raw (unweighted) terms for per-epoch logging.
        self._recon_sum += float(recon.detach())
        self._kl_sum += float(kl.detach())
        self._term_batches += 1
        return recon + self._effective_beta() * kl

    def _epoch_extra_metrics(self) -> dict:
        n = max(1, self._term_batches)
        recon = self._recon_sum / n
        kl = self._kl_sum / n
        self._recon_sum = self._kl_sum = 0.0
        self._term_batches = 0
        import logging
        logging.getLogger(__name__).info(
            "  recon=%.5f  kl=%.5f  beta=%.3f", recon, kl, self._effective_beta()
        )
        return {"recon": recon, "kl": kl, "beta": self._effective_beta()}

    @torch.no_grad()
    def _eval_forward(self, data):
        """Forward with the model in eval mode (masking off, z = mu).

        Metrics and reconstruction plots must reflect the *clean* reconstruction,
        not the train-time masked/sampled one, or they look pessimistic.
        """
        was_training = self.model.training
        self.model.eval()
        try:
            return self.model(data)
        finally:
            if was_training:
                self.model.train()

    @torch.no_grad()
    def compute_scores(self, batch) -> Optional[torch.Tensor]:
        from torch_geometric.nn import global_max_pool, global_mean_pool

        data = batch.to(self.device)
        x_hat, _, _, _ = self._eval_forward(data)
        per_node = ((x_hat - data.y.float()) ** 2).mean(dim=-1)  # (N,)
        gid = data.batch
        wmean = global_mean_pool(per_node.unsqueeze(1), gid).squeeze(1)
        wmax = global_max_pool(per_node.unsqueeze(1), gid).squeeze(1)
        return torch.stack([wmean, wmax], dim=1)

    @torch.no_grad()
    def compute_reconstruction_pair(self, batch):
        data = batch.to(self.device)
        x_hat, _, _, _ = self._eval_forward(data)
        return data.y.float(), x_hat

    def collect_scores(self, loader) -> "tuple[np.ndarray, np.ndarray]":
        means, maxes = [], []
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for batch in loader:
                    s = self.compute_scores(batch)
                    if s is not None:
                        means.append(s[:, 0].cpu())
                        maxes.append(s[:, 1].cpu())
        finally:
            if was_training:
                self.model.train()
        if not means:
            return np.array([]), np.array([])
        return torch.cat(means).numpy(), torch.cat(maxes).numpy()

    def collect_channel_mse(self, loader, num_channels: int) -> torch.Tensor:
        mse_sum = torch.zeros(num_channels)
        mse_count = torch.zeros(num_channels)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for batch in loader:
                    data = batch.to(self.device)
                    x_hat, _, _, _ = self.model(data)
                    per_node = ((x_hat - data.y.float()) ** 2).mean(dim=-1).cpu()
                    active = data.active_mask.cpu()
                    mse_sum.scatter_add_(0, active, per_node)
                    mse_count.scatter_add_(0, active, torch.ones(per_node.shape[0]))
        finally:
            if was_training:
                self.model.train()
        out = mse_sum / mse_count
        out[mse_count == 0] = float("nan")
        return out
