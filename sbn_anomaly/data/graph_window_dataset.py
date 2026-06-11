from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def build_channel_adjacency(num_nodes: int, radius: int = 1) -> np.ndarray:
    """Build a dense adjacency matrix where channels within |i-j| <= radius are neighbors.

    Args:
        num_nodes: number of channel nodes
        radius: integer neighborhood radius (inclusive)

    Returns:
        adjacency matrix shape (num_nodes, num_nodes) dtype float32 with 1.0 for edges.
    """
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        lo = max(0, i - radius)
        hi = min(num_nodes - 1, i + radius)
        adj[i, lo : hi + 1] = 1.0
    # zero self if desired (keep self for now)
    return adj


class GraphWindowDataset(Dataset):
    """Dataset of graph snapshots (rolling windows) for forecasting.

    Expects input data to be either a numpy array of shape
    (N_windows, N_nodes, node_feature_dim) or a .npz archive with a
    'windows' array and optional 'channel_map' metadata.

    This dataset yields tuples ``(past_windows, adj, target_window)`` where
    - past_windows: Tensor shape (T, N_nodes, node_feat_dim)
    - adj: Tensor shape (N_nodes, N_nodes)
    - target_window: Tensor shape (N_nodes, node_feat_dim)

    The dataset uses a sliding history length `history` to form the input
    sequence that predicts the next window.
    """

    def __init__(
        self,
        windows: np.ndarray,
        history: int = 4,
        stride: int = 1,
        adjacency: Optional[np.ndarray] = None,
    ) -> None:
        if isinstance(windows, np.lib.npyio.NpzFile):
            if "windows" not in windows:
                raise ValueError(".npz archive must contain 'windows' array")
            windows = windows["windows"]
        self.windows = np.asarray(windows, dtype=np.float32)
        if self.windows.ndim != 3:
            raise ValueError("windows must be 3-D: (N, N_nodes, node_feat_dim)")
        self.history = int(history)
        self.stride = int(stride)
        self.adjacency = (
            np.asarray(adjacency, dtype=np.float32)
            if adjacency is not None
            else None
        )

        # compute starts for sequences where target exists
        self._starts = list(range(0, len(self.windows) - self.history, self.stride))

        # default adjacency: local neighbor radius configurable (default=4)
        if self.adjacency is None:
            num_nodes = int(self.windows.shape[1])
            self.adjacency = build_channel_adjacency(num_nodes, radius=4)

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int):
        start = self._starts[idx]
        past = self.windows[start : start + self.history]  # (T, N_nodes, feat)
        target = self.windows[start + self.history]  # (N_nodes, feat)
        adj = self.adjacency
        return (
            torch.from_numpy(past),
            torch.from_numpy(adj),
            torch.from_numpy(target),
        )
