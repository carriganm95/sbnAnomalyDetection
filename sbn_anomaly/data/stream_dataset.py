"""PyTorch IterableDataset that streams TPC waveform features from ROOT files.

Feature extraction strategy
----------------------------
Each event in ``tpc_waveform`` is a variable-length nested array (channels ×
ticks).  We flatten the entire event payload to a 1-D array and then
**pad** (with zeros) or **truncate** to a fixed ``input_dim`` so the model
receives constant-size tensors.

Usage::

    from sbn_anomaly.data.stream_dataset import TPCStreamDataset
    from torch.utils.data import DataLoader

    ds = TPCStreamDataset(
        file_paths=["/data/run.root"],
        tree_name="sbn_tree",
        waveform_branch="tpc_waveform",
        input_dim=256,
    )
    loader = DataLoader(ds, batch_size=256)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import torch
from torch.utils.data import IterableDataset

from sbn_anomaly.data.streaming import RootStreamer

logger = logging.getLogger(__name__)


def extract_tpc_features(
    raw: np.ndarray,
    input_dim: int = 256,
) -> np.ndarray:
    """Flatten *raw* waveform values and pad/truncate to *input_dim*.

    Parameters
    ----------
    raw:
        1-D float32 array containing concatenated waveform samples for one
        event (already flattened from a nested awkward array).
    input_dim:
        Target feature vector length.

    Returns
    -------
    numpy.ndarray of shape ``(input_dim,)`` and dtype ``float32``.
    """
    raw = raw.astype(np.float32)
    if len(raw) >= input_dim:
        return raw[:input_dim]
    padded = np.zeros(input_dim, dtype=np.float32)
    padded[: len(raw)] = raw
    return padded


class TPCStreamDataset(IterableDataset):
    """Stream per-event TPC feature vectors directly from ROOT files.

    Parameters
    ----------
    file_paths:
        One or more paths to ``.root`` files.  Glob strings should be
        expanded by the caller before passing here.
    tree_name:
        Name of the TTree inside each ROOT file.
    waveform_branch:
        Branch name that holds the per-event waveform data.  The branch
        contents are flattened per event and padded/truncated to
        ``input_dim``.
    branches:
        Full list of branches to load per event.  If ``None``, only
        ``waveform_branch`` is loaded.
    input_dim:
        Size of the output feature vector for each event (default 256).
    batch_size:
        Number of events per ``RootStreamer`` batch (controls memory use).
    normalize:
        When ``True``, apply per-event z-score standardisation (zero mean,
        unit variance).  Events with zero variance are left unchanged.
    max_events:
        Stop after yielding this many events (``None`` = unlimited).
    """

    def __init__(
        self,
        file_paths: Union[str, Path, Iterable[Union[str, Path]]],
        tree_name: str = "sbn_tree",
        waveform_branch: str = "tpc_waveform",
        branches: Optional[list[str]] = None,
        input_dim: int = 256,
        batch_size: int = 512,
        normalize: bool = False,
        max_events: Optional[int] = None,
    ) -> None:
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        self.file_paths = [Path(p) for p in file_paths]
        self.tree_name = tree_name
        self.waveform_branch = waveform_branch
        self.branches = branches or [waveform_branch]
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.normalize = normalize
        self.max_events = max_events

    # ------------------------------------------------------------------
    # IterableDataset interface
    # ------------------------------------------------------------------

    def __iter__(self):
        import awkward as ak

        streamer = RootStreamer(
            file_paths=self.file_paths,
            tree_name=self.tree_name,
            branches=self.branches,
            batch_size=self.batch_size,
        )

        n_yielded = 0
        for batch in streamer.stream():
            waveforms = batch[self.waveform_branch]
            for i in range(len(waveforms)):
                raw = ak.to_numpy(
                    ak.flatten(waveforms[i], axis=None)
                ).astype(np.float32)
                feat = extract_tpc_features(raw, self.input_dim)

                if self.normalize:
                    mean = feat.mean()
                    std = feat.std()
                    if std > 0.0:
                        feat = (feat - mean) / std

                yield (torch.from_numpy(feat),)

                n_yielded += 1
                if self.max_events is not None and n_yielded >= self.max_events:
                    return
