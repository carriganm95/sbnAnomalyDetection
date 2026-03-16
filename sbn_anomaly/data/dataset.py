"""PyTorch Dataset wrappers for TPC, PMT and fused event representations.

Each dataset works with pre-loaded or lazily materialised NumPy arrays of
fixed-length feature vectors.  For production use, subclass and override
``_load()`` to stream from ``RootStreamer``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class TPCDataset(Dataset):
    """Dataset of TPC waveform feature vectors.

    Parameters
    ----------
    features:
        Float array of shape (N, tpc_feature_dim).
    labels:
        Optional int array of shape (N,).  ``None`` for unsupervised use.
    window_size:
        If > 1, features are treated as raw time series of length
        ``tpc_feature_dim`` and the dataset yields sliding windows.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None,
        window_size: int = 1,
    ) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        x = self.features[idx]
        if self.labels is not None:
            return x, self.labels[idx]
        return (x,)


class PMTDataset(Dataset):
    """Dataset of PMT waveform feature vectors.

    Parameters
    ----------
    features:
        Float array of shape (N, pmt_feature_dim).
    labels:
        Optional int array of shape (N,).
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        x = self.features[idx]
        if self.labels is not None:
            return x, self.labels[idx]
        return (x,)


class FusionDataset(Dataset):
    """Dataset that pairs matched TPC and PMT feature vectors.

    Both arrays must have the same length (pre-joined via ``EventJoiner``).

    Parameters
    ----------
    tpc_features:
        Float array of shape (N, tpc_feature_dim).
    pmt_features:
        Float array of shape (N, pmt_feature_dim).
    labels:
        Optional int array of shape (N,).
    """

    def __init__(
        self,
        tpc_features: np.ndarray,
        pmt_features: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> None:
        if len(tpc_features) != len(pmt_features):
            raise ValueError(
                f"TPC and PMT feature arrays must have the same length, "
                f"got {len(tpc_features)} vs {len(pmt_features)}."
            )
        self.tpc = torch.tensor(tpc_features, dtype=torch.float32)
        self.pmt = torch.tensor(pmt_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self) -> int:
        return len(self.tpc)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        x_tpc = self.tpc[idx]
        x_pmt = self.pmt[idx]
        if self.labels is not None:
            return x_tpc, x_pmt, self.labels[idx]
        return x_tpc, x_pmt


class WindowDataset(Dataset):
    """Dataset that yields overlapping temporal windows from a 1-D signal.

    Suitable for time-series anomaly detection where the input is a raw
    waveform and the model scores windows.

    Parameters
    ----------
    signal:
        1-D float array of length T (a single long waveform or concatenated
        waveforms).
    window_size:
        Number of samples per window.
    stride:
        Step size between consecutive windows.
    labels:
        Optional 1-D int array of length T (per-sample labels).
    """

    def __init__(
        self,
        signal: np.ndarray,
        window_size: int,
        stride: int = 1,
        labels: Optional[np.ndarray] = None,
    ) -> None:
        self.signal = torch.tensor(signal, dtype=torch.float32)
        self.window_size = window_size
        self.stride = stride
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None
        # Pre-compute start indices.
        self._starts = list(range(0, len(signal) - window_size + 1, stride))

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        start = self._starts[idx]
        window = self.signal[start : start + self.window_size]
        if self.labels is not None:
            label = self.labels[start : start + self.window_size]
            return window, label
        return (window,)
