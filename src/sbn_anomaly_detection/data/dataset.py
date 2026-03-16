"""PyTorch Dataset wrappers for SBN event data.

Supports both:
 - ``EventDataset``   – one sample = one event (for TPC / PMT / Fusion training)
 - ``WindowDataset``  – one sample = a sliding window of consecutive events
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from sbn_anomaly_detection.data.root_stream import stream_to_numpy

logger = logging.getLogger(__name__)


def _load_all(
    root_files,
    branches: List[str],
    tree_name: str = "events",
    step_size: int = 1000,
    max_events: Optional[int] = None,
) -> np.ndarray:
    """Materialize all streaming batches into a single float32 array."""
    chunks = list(
        stream_to_numpy(
            root_files,
            branches,
            tree_name=tree_name,
            step_size=step_size,
            max_events=max_events,
        )
    )
    if not chunks:
        raise RuntimeError("No data was loaded — check file paths and branch names.")
    arr = np.concatenate(chunks, axis=0)
    logger.info("Loaded %d events with %d features.", arr.shape[0], arr.shape[1])
    return arr


class EventDataset(Dataset):
    """Dataset where each item is a single normalised event feature vector.

    Parameters
    ----------
    data:
        Either a pre-loaded ``np.ndarray`` of shape ``(N, F)`` or a dict of
        ``{"root_files": ..., "branches": ..., ...}`` kwargs forwarded to
        :func:`~data.root_stream.stream_to_numpy`.
    transform:
        Optional callable applied to each sample tensor.
    """

    def __init__(
        self,
        data: Union[np.ndarray, dict],
        transform=None,
    ) -> None:
        if isinstance(data, np.ndarray):
            self._data = data.astype(np.float32)
        elif isinstance(data, dict):
            self._data = _load_all(**data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        self.transform = transform

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = torch.from_numpy(self._data[idx])
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    @property
    def n_features(self) -> int:
        return self._data.shape[1]


class WindowDataset(Dataset):
    """Dataset where each item is a fixed-length window of consecutive events.

    Each sample is a tensor of shape ``(window_size, n_features)``.

    Parameters
    ----------
    data:
        Pre-loaded array of shape ``(N, F)`` or a loader-kwargs dict.
    window_size:
        Number of consecutive events per window.
    stride:
        Step between successive window start positions. Defaults to 1.
    transform:
        Optional callable applied to each window tensor.
    """

    def __init__(
        self,
        data: Union[np.ndarray, dict],
        window_size: int = 64,
        stride: int = 1,
        transform=None,
    ) -> None:
        if isinstance(data, np.ndarray):
            self._data = data.astype(np.float32)
        elif isinstance(data, dict):
            self._data = _load_all(**data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        self.window_size = window_size
        self.stride = stride
        self.transform = transform

        n_events = len(self._data)
        if n_events < window_size:
            raise ValueError(
                f"Dataset has only {n_events} events but window_size={window_size}."
            )
        self._indices = list(range(0, n_events - window_size + 1, stride))
        logger.info(
            "WindowDataset: %d events → %d windows (size=%d, stride=%d)",
            n_events,
            len(self._indices),
            window_size,
            stride,
        )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = self._indices[idx]
        window = self._data[start : start + self.window_size]
        sample = torch.from_numpy(window)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    @property
    def n_features(self) -> int:
        return self._data.shape[1]
