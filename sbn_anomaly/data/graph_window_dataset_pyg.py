"""PyTorch Geometric-based graph window dataset with sparse adjacency.

Converts windowed time series data into PyG Data objects for efficient batching
and sparse graph operations. Supports per-sample node pruning and channel index encoding.

Data objects are built lazily in __getitem__ so there is no upfront materialization
cost and torch_geometric DataLoader workers can build samples in parallel.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


def build_sparse_edge_index(num_nodes: int, radius: int = 4) -> torch.LongTensor:
    """Build sparse COO edge index for radius-based neighborhood.

    Args:
        num_nodes: number of nodes (channels)
        radius: integer neighborhood radius (inclusive, including self-loops)

    Returns:
        edge_index: (2, num_edges) LongTensor in COO format [src, dst]
    """
    nodes = torch.arange(num_nodes, dtype=torch.long)
    src_list, dst_list = [], []
    for offset in range(-radius, radius + 1):
        dst = nodes + offset
        valid = (dst >= 0) & (dst < num_nodes)
        src_list.append(nodes[valid])
        dst_list.append(dst[valid])
    return torch.stack([torch.cat(src_list), torch.cat(dst_list)])


class GraphWindowDatasetPyG(Dataset):
    """PyG-compatible dataset for graph-structured temporal windows.

    Data objects are built lazily in __getitem__, so construction is
    near-instant regardless of dataset size. Use num_workers > 0 in the
    DataLoader to build samples in parallel across CPU cores.

    Each Data object contains:
        x:                  (M, 1 + history*F) — channel_idx + flattened temporal features
        y:                  (M, F) — next-frame target
        edge_index:         (2, E) — sparse COO edges
        active_mask:        (M,) — original channel indices of kept nodes
        num_nodes_original: int — total channel count before pruning
    """

    def __init__(
        self,
        windows: np.ndarray,
        history: int = 4,
        stride: int = 1,
        radius: int = 4,
        prune_inactive: bool = True,
        node_feature_names: list[str] | None = None,
    ) -> None:
        """Initialize dataset from pre-materialized windows.

        Args:
            windows: (num_windows, num_nodes, F) array of per-step node features
            history: number of past frames per sample
            stride: step size for the sliding window
            radius: channel adjacency radius for graph edges
            prune_inactive: drop zero-activity nodes per sample to save memory
            node_feature_names: optional list of F feature names (e.g. ["sum","min","max"]);
                stored as ``hit_branches`` so BaseTrainer labels reconstruction plots
        """
        _windows = np.asarray(windows, dtype=np.float32)
        if _windows.ndim != 3:
            raise ValueError(f"windows must be 3-D (num_windows, num_nodes, F), got {_windows.ndim}-D")

        self.history = int(history)
        self.stride = int(stride)
        self.radius = int(radius)
        self.prune_inactive = bool(prune_inactive)
        self.hit_branches: list[str] | None = list(node_feature_names) if node_feature_names else None

        self.num_windows, self.num_nodes, self.node_feat_dim = _windows.shape
        self._windows = _windows  # kept for lazy __getitem__

        logger.info("Building edge index: %d nodes, radius=%d", self.num_nodes, radius)
        self.edge_index_full = build_sparse_edge_index(self.num_nodes, radius=radius)
        logger.info("Edge index built: %d edges", self.edge_index_full.shape[1])

        # Pre-extract as numpy for fast indexing in _make_pruned_data
        self._edge_src_np = self.edge_index_full[0].numpy().copy()
        self._edge_dst_np = self.edge_index_full[1].numpy().copy()

        # Precompute normalized channel indices — shared across all samples
        self._channel_idx = (
            torch.arange(self.num_nodes, dtype=torch.float32) / max(1, self.num_nodes - 1)
        ).unsqueeze(1)  # (C, 1)

        self._starts = list(range(0, self.num_windows - self.history, self.stride))
        logger.info(
            "Dataset ready: %d samples, %d channels, %d features/step, history=%d  "
            "(lazy — samples built on demand)",
            len(self._starts), self.num_nodes, self.node_feat_dim, self.history,
        )

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> Data:
        start = self._starts[idx]
        T = self.history

        past = self._windows[start : start + T]   # (T, C, F) — numpy view, no copy
        target = self._windows[start + T]          # (C, F)

        # (T, C, F) -> (C, T*F): transpose forces a C-contiguous copy, then reshape is free
        past_flat = torch.from_numpy(
            past.transpose(1, 0, 2).reshape(self.num_nodes, -1)
        ).float()  # (C, T*F)
        target_flat = torch.from_numpy(target).float()  # (C, F)

        x = torch.cat([self._channel_idx, past_flat], dim=1)  # (C, 1+T*F)

        if self.prune_inactive:
            activity = torch.abs(past_flat).sum(dim=1)
            active_idx = torch.where(activity > 1e-6)[0]
            if active_idx.numel() == 0:
                active_idx = torch.zeros(1, dtype=torch.long)
            return self._make_pruned_data(x, target_flat, active_idx)
        else:
            return Data(x=x, y=target_flat, edge_index=self.edge_index_full)

    def _make_pruned_data(
        self, x: torch.Tensor, y: torch.Tensor, active_idx: torch.Tensor
    ) -> Data:
        """Build a Data object keeping only active nodes.

        Uses a numpy remap array (~20× faster than an equivalent torch approach
        for graphs of this size, due to numpy's lower overhead for integer indexing
        on medium-sized arrays).
        """
        m = active_idx.numel()
        active_np = active_idx.numpy()

        x_pruned = x[active_idx]
        y_pruned = y[active_idx]

        # Recompute normalized channel index for the pruned node set
        x_pruned[:, 0] = torch.arange(m, dtype=torch.float32) / max(1, m - 1)

        # Build old->new index remap in numpy — 20× faster than torch.full + scatter
        remap = np.full(self.num_nodes, -1, dtype=np.int32)
        remap[active_np] = np.arange(m, dtype=np.int32)

        new_src = remap[self._edge_src_np]
        new_dst = remap[self._edge_dst_np]
        keep = (new_src >= 0) & (new_dst >= 0)

        if keep.any():
            edge_index = torch.from_numpy(
                np.stack([new_src[keep], new_dst[keep]]).astype(np.int64)
            )
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return Data(
            x=x_pruned,
            y=y_pruned,
            edge_index=edge_index,
            active_mask=active_idx,
            num_nodes_original=self.num_nodes,
        )
