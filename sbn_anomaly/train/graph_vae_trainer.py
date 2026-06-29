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
        return recon + self._effective_beta() * kl

    @torch.no_grad()
    def compute_scores(self, batch) -> Optional[torch.Tensor]:
        from torch_geometric.nn import global_max_pool, global_mean_pool

        data = batch.to(self.device)
        x_hat, _, _, _ = self.model(data)
        per_node = ((x_hat - data.y.float()) ** 2).mean(dim=-1)  # (N,)
        gid = data.batch
        wmean = global_mean_pool(per_node.unsqueeze(1), gid).squeeze(1)
        wmax = global_max_pool(per_node.unsqueeze(1), gid).squeeze(1)
        return torch.stack([wmean, wmax], dim=1)

    @torch.no_grad()
    def compute_reconstruction_pair(self, batch):
        data = batch.to(self.device)
        x_hat, _, _, _ = self.model(data)
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
