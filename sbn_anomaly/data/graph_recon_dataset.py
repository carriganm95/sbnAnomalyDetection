"""Per-window reconstruction dataset for the graph VAE.

Each sample is ONE window's channel graph: node features are the per-channel
aggregates for that window, and the target is those same features (autoencoder).
This is the reconstruction counterpart of :class:`GraphWindowDatasetPyG` (which
builds history->next-frame forecasting samples).

Yields PyG ``Data`` with:
    x:                  (M, 1 + F) — channel_idx column + standardized features
    y:                  (M, F)     — standardized reconstruction target
    edge_index:         (2, E)     — electronics/sequential edges (pruned)
    active_mask:        (M,)       — original channel ids of kept nodes
    num_nodes_original: int

Features are z-scored per feature using good-run statistics so reconstruction
error means "deviation from nominal in sigma units"; the stats are exposed for
saving so inference uses the same normalization.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from sbn_anomaly.data.graph_window_dataset_pyg import _build_edge_index

logger = logging.getLogger(__name__)


class GraphReconDataset(Dataset):
    def __init__(
        self,
        windows: np.ndarray,
        radius: int = 4,
        prune_inactive: bool = True,
        node_feature_names: Optional[list] = None,
        channel_map: Optional[str] = None,
        edge_mode: str = "sequential",
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,
        standardize: bool = True,
    ) -> None:
        w = np.asarray(windows, dtype=np.float32)
        if w.ndim != 3:
            raise ValueError(f"windows must be 3-D (num_windows, num_nodes, F), got {w.ndim}-D")
        self.num_windows, self.num_nodes, self.node_feat_dim = w.shape
        self._windows = w
        self.radius = int(radius)
        self.prune_inactive = bool(prune_inactive)
        self.hit_branches = list(node_feature_names) if node_feature_names else None
        self.channel_map = channel_map
        self.edge_mode = str(edge_mode)
        self.standardize = bool(standardize)

        # Per-feature standardization from active (non-zero) rows of good data.
        if self.standardize:
            if feature_mean is None or feature_std is None:
                active = np.abs(w).sum(axis=2) > 1e-6      # (W, C)
                flat = w[active]                            # (n_active, F)
                if flat.shape[0] == 0:
                    flat = w.reshape(-1, self.node_feat_dim)
                feature_mean = flat.mean(axis=0)
                feature_std = flat.std(axis=0)
            self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
            self.feature_std = np.asarray(feature_std, dtype=np.float32)
            self.feature_std = np.where(self.feature_std < 1e-6, 1.0, self.feature_std).astype(np.float32)
        else:
            self.feature_mean = np.zeros(self.node_feat_dim, dtype=np.float32)
            self.feature_std = np.ones(self.node_feat_dim, dtype=np.float32)

        self.edge_index_full = _build_edge_index(
            self.num_nodes, radius=radius,
            channel_map=channel_map, edge_mode=self.edge_mode,
        )
        self._edge_src_np = self.edge_index_full[0].numpy().copy()
        self._edge_dst_np = self.edge_index_full[1].numpy().copy()
        self._channel_idx = (
            torch.arange(self.num_nodes, dtype=torch.float32) / max(1, self.num_nodes - 1)
        ).unsqueeze(1)

        logger.info(
            "GraphReconDataset: %d windows, %d channels, %d features/window, "
            "edge_mode=%s, standardize=%s",
            self.num_windows, self.num_nodes, self.node_feat_dim,
            self.edge_mode, self.standardize,
        )

    def __len__(self) -> int:
        return self.num_windows

    def standardization(self) -> dict:
        return {"feature_mean": self.feature_mean, "feature_std": self.feature_std}

    def __getitem__(self, idx: int) -> Data:
        raw = self._windows[idx]                                   # (C, F)
        feats = (raw - self.feature_mean) / self.feature_std       # standardized
        feats_t = torch.from_numpy(feats.astype(np.float32))
        x = torch.cat([self._channel_idx, feats_t], dim=1)         # (C, 1+F)
        y = feats_t                                                # (C, F)

        if not self.prune_inactive:
            return Data(x=x, y=y, edge_index=self.edge_index_full,
                        active_mask=torch.arange(self.num_nodes),
                        num_nodes_original=self.num_nodes)

        activity = np.abs(raw).sum(axis=1)                         # decide active on RAW
        active_idx = np.where(activity > 1e-6)[0]
        if active_idx.size == 0:
            active_idx = np.zeros(1, dtype=np.int64)
        active_t = torch.from_numpy(active_idx.astype(np.int64))

        x_p = x[active_t]
        y_p = y[active_t]
        m = active_idx.size
        x_p[:, 0] = torch.arange(m, dtype=torch.float32) / max(1, m - 1)

        remap = np.full(self.num_nodes, -1, dtype=np.int64)
        remap[active_idx] = np.arange(m, dtype=np.int64)
        new_src = remap[self._edge_src_np]
        new_dst = remap[self._edge_dst_np]
        keep = (new_src >= 0) & (new_dst >= 0)
        if keep.any():
            edge_index = torch.from_numpy(np.stack([new_src[keep], new_dst[keep]]).astype(np.int64))
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return Data(x=x_p, y=y_p, edge_index=edge_index,
                    active_mask=active_t, num_nodes_original=self.num_nodes)
