from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from sbn_anomaly.train.trainer import BaseTrainer


class GNNTrainer(BaseTrainer):
    """Trainer for the dense-adjacency GNNForecaster.

    Expects DataLoader to yield (past_windows, adj, target_window) where:
        past_windows: (B, T, N, F)
        adj:          (N, N) or (B, N, N)
        target_window:(B, N, F)
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "auto",
        max_epochs: int = 50,
        checkpoint_dir: Optional[str] = None,
        log_interval: int = 50,
        anomaly_threshold: Optional[float] = None,
        save_best_only: bool = False,
    ) -> None:
        if model is None:
            raise ValueError("GNNTrainer requires a model instance")
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
        )
        self.criterion = nn.MSELoss()

    def compute_loss(self, batch: tuple) -> torch.Tensor:
        past, adj, target = batch
        past = past.to(self.device).float()
        adj = adj.to(self.device).float()
        target = target.to(self.device).float()
        pred = self.model(past, adj)
        return self.criterion(pred, target)

    def compute_scores(self, batch: tuple) -> Optional[torch.Tensor]:
        past, adj, target = batch
        past = past.to(self.device).float()
        adj = adj.to(self.device).float()
        target = target.to(self.device).float()
        with torch.no_grad():
            pred = self.model(past, adj)
            per_node_mse = ((pred - target) ** 2).mean(dim=-1)  # (B, N)
            window_mean = per_node_mse.mean(dim=-1)
            window_max = per_node_mse.max(dim=-1).values
            return torch.stack([window_mean, window_max], dim=1)


class GNNTrainerPyG(BaseTrainer):
    """Trainer for GNNForecasterPyG using sparse PyG batches.

    Expects a torch_geometric.loader.DataLoader over a GraphWindowDatasetPyG,
    which yields PyG Batch objects with x, y, edge_index, and batch fields.
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "auto",
        max_epochs: int = 50,
        checkpoint_dir: Optional[str] = None,
        log_interval: int = 50,
        anomaly_threshold: Optional[float] = None,
        save_best_only: bool = False,
        use_amp: bool = False,
    ) -> None:
        if model is None:
            raise ValueError("GNNTrainerPyG requires a model instance")
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
        )
        self.criterion = nn.MSELoss()

    def _infer_batch_size(self, batch) -> int:
        return int(getattr(batch, "num_graphs", 1))

    def compute_loss(self, batch) -> torch.Tensor:
        data = batch.to(self.device)
        pred = self.model(data)
        return self.criterion(pred, data.y.float())

    def compute_scores(self, batch) -> Optional[torch.Tensor]:
        from torch_geometric.nn import global_max_pool, global_mean_pool

        data = batch.to(self.device)
        with torch.no_grad():
            pred = self.model(data)
            per_node_mse = ((pred - data.y.float()) ** 2).mean(dim=-1)  # (total_nodes,)
            graph_idx = data.batch  # (total_nodes,) — which graph each node belongs to
            window_mean = global_mean_pool(per_node_mse.unsqueeze(1), graph_idx).squeeze(1)
            window_max = global_max_pool(per_node_mse.unsqueeze(1), graph_idx).squeeze(1)
            return torch.stack([window_mean, window_max], dim=1)  # (B, 2)

    def compute_reconstruction_pair(self, batch) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Return (target, prediction) for the base trainer's reconstruction histograms."""
        data = batch.to(self.device)
        with torch.no_grad():
            pred = self.model(data)
        return data.y.float(), pred

    def collect_scores(self, loader) -> "tuple[np.ndarray, np.ndarray]":
        """Return per-window (mean_mse, max_mse) arrays over a full loader pass.

        Results are returned in the order the loader yields batches, so pass a
        non-shuffled loader to get scores in temporal (window index) order.

        Args:
            loader: torch_geometric DataLoader over GraphWindowDatasetPyG

        Returns:
            scores_mean: (N,) per-window mean node MSE
            scores_max:  (N,) per-window max node MSE
        """
        import numpy as np

        means: list[torch.Tensor] = []
        maxes: list[torch.Tensor] = []

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for batch in loader:
                    scores = self.compute_scores(batch)
                    if scores is not None:
                        means.append(scores[:, 0].cpu())
                        maxes.append(scores[:, 1].cpu())
        finally:
            if was_training:
                self.model.train()

        if not means:
            return np.array([]), np.array([])
        return torch.cat(means).numpy(), torch.cat(maxes).numpy()

    def collect_channel_mse(
        self,
        loader,
        num_channels: int,
    ) -> torch.Tensor:
        """Compute per-original-channel average MSE over a full DataLoader pass.

        Uses each Data object's ``active_mask`` (original channel indices) to scatter
        node-level MSE back to the global channel grid.

        Args:
            loader: torch_geometric DataLoader over GraphWindowDatasetPyG
            num_channels: total number of channels in the original (unpruned) graph

        Returns:
            channel_mse: (num_channels,) tensor, NaN for channels never active
        """
        mse_sum = torch.zeros(num_channels)
        mse_count = torch.zeros(num_channels)

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for batch in loader:
                    data = batch.to(self.device)
                    pred = self.model(data)
                    per_node_mse = ((pred - data.y.float()) ** 2).mean(dim=-1).cpu()
                    # active_mask concatenated by PyG: original channel indices per node
                    active_mask = data.active_mask.cpu()
                    mse_sum.scatter_add_(0, active_mask, per_node_mse)
                    mse_count.scatter_add_(0, active_mask, torch.ones(per_node_mse.shape[0]))
        finally:
            if was_training:
                self.model.train()

        channel_mse = torch.where(
            mse_count > 0,
            mse_sum / mse_count.clamp(min=1),
            torch.full_like(mse_sum, float("nan")),
        )
        return channel_mse
