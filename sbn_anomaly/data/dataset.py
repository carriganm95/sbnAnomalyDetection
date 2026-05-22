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
import logging

logger = logging.getLogger(__name__)


class TPCDataset(Dataset):
    """Dataset of TPC waveform feature vectors.

    Parameters
    ----------
    features:
        Float array of shape (N, tpc_feature_dim) or .npz archive.
    labels:
        Optional int array of shape (N,).  ``None`` for unsupervised use.
    input_dim:
        If provided, truncate/pad each feature vector to this length (like
        streaming datasets do). If ``None``, use features as-is.
    window_size:
        If > 1, features are treated as raw time series of length
        ``tpc_feature_dim`` and the dataset yields sliding windows.
    normalize:
        If True, apply per-sample z-score normalization.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None,
        input_dim: Optional[int] = None,
        window_size: int = 1,
        normalize: bool = False,
    ) -> None:
        # Handle both raw numpy arrays and .npz archives
        self.hit_branches = None
        self.tpc_branch_values = None
        self.input_filenames = None
        if isinstance(features, np.lib.npyio.NpzFile):
            # Extract metadata from .npz if available
            if "feature_branch_names" in features:
                branch_names = features["feature_branch_names"]
                # Convert from numpy array if needed
                if isinstance(branch_names, np.ndarray):
                    self.hit_branches = [str(name) for name in branch_names]
                else:
                    self.hit_branches = branch_names
                # Log the extracted branch names for debugging.
                try:
                    logger.info("Loaded TPC feature_branch_names: %s", self.hit_branches)
                except Exception:
                    pass
            # Load tpc_branch_values if available
            if "tpc_branch_values" in features:
                self.tpc_branch_values = features["tpc_branch_values"].astype(np.float32)
                try:
                    logger.info("Loaded TPC branch values with shape: %s", self.tpc_branch_values.shape)
                except Exception:
                    pass
            if "input_filenames" in features:
                self.input_filenames = np.asarray(features["input_filenames"], dtype=str)
                try:
                    logger.info("Loaded input_filenames with shape: %s", self.input_filenames.shape)
                except Exception:
                    pass
            features = features["features"]
        
        # Truncate/pad features to input_dim if specified
        if input_dim is not None and features.shape[1] != input_dim:
            features_adjusted = np.zeros((features.shape[0], input_dim), dtype=np.float32)
            copy_len = min(features.shape[1], input_dim)
            features_adjusted[:, :copy_len] = features[:, :copy_len]
            features = features_adjusted
        
        self.features = features.astype(np.float32)
        self.labels = labels
        self.window_size = window_size
        self.normalize = normalize
        
        # Pre-convert to tensors for efficiency (except we'll apply normalization per-item)
        if not normalize:
            self.features = torch.tensor(self.features, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self) -> int:
        if isinstance(self.features, np.ndarray):
            return len(self.features)
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        # Get feature (convert from numpy if needed)
        if isinstance(self.features, np.ndarray):
            x = self.features[idx].astype(np.float32)
            # Apply normalization if requested
            if self.normalize:
                mean = x.mean()
                std = x.std()
                if std > 0.0:
                    x = (x - mean) / std
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = self.features[idx]
        
        if self.labels is not None:
            if isinstance(self.labels, np.ndarray):
                label = torch.tensor(self.labels[idx], dtype=torch.long)
            else:
                label = self.labels[idx]
            return x, label
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
        # Handle both raw numpy arrays and .npz archives
        self.hit_branches = None
        self.tpc_branch_values = None
        self.input_filenames = None
        if isinstance(features, np.lib.npyio.NpzFile):
            if "feature_branch_names" in features:
                branch_names = features["feature_branch_names"]
                if isinstance(branch_names, np.ndarray):
                    self.hit_branches = [str(name) for name in branch_names]
                else:
                    self.hit_branches = branch_names
                try:
                    logger.info("Loaded PMT feature_branch_names: %s", self.hit_branches)
                except Exception:
                    pass
            # Load tpc_branch_values if available
            if "tpc_branch_values" in features:
                self.tpc_branch_values = features["tpc_branch_values"].astype(np.float32)
            if "input_filenames" in features:
                self.input_filenames = np.asarray(features["input_filenames"], dtype=str)
            features = features["features"]
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
        # Handle both raw numpy arrays and .npz archives
        # Preserve TPC feature metadata when passing .npz archives.
        self.hit_branches = None
        self.tpc_branch_values = None
        self.input_filenames = None
        if isinstance(tpc_features, np.lib.npyio.NpzFile):
            if "feature_branch_names" in tpc_features:
                branch_names = tpc_features["feature_branch_names"]
                if isinstance(branch_names, np.ndarray):
                    self.hit_branches = [str(name) for name in branch_names]
                else:
                    self.hit_branches = branch_names
                try:
                    logger.info("Loaded Fusion TPC feature_branch_names: %s", self.hit_branches)
                except Exception:
                    pass
            # Load tpc_branch_values if available
            if "tpc_branch_values" in tpc_features:
                self.tpc_branch_values = tpc_features["tpc_branch_values"].astype(np.float32)
            if "input_filenames" in tpc_features:
                self.input_filenames = np.asarray(tpc_features["input_filenames"], dtype=str)
            tpc_features = tpc_features["features"]
        if isinstance(pmt_features, np.lib.npyio.NpzFile):
            pmt_features = pmt_features["features"]
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
        # Handle both raw numpy arrays and .npz archives
        if isinstance(signal, np.lib.npyio.NpzFile):
            signal = signal["features"]
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
