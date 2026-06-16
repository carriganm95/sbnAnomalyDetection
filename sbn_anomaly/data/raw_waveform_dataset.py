"""Datasets that feed the waveform VAE single-channel ADC waveforms.

Two flavours:

* :class:`RawWaveformStreamDataset` — an ``IterableDataset`` that streams events
  from the flat raw-ADC ntuple (via :class:`RawDigitReader`), pre-processes each
  event, and yields one pre-processed ``(input_length,)`` waveform per channel.
  Use this to train the VAE directly from ROOT.

* :class:`RawWaveformArrayDataset` — a map-style dataset over an in-memory or
  ``.npy`` array of shape ``(n_waveforms, input_length)``.  Handy for tests and
  for training from a pre-materialised waveform cache.

Both yield a 1-tuple ``(waveform,)`` so they drop straight into the existing
:class:`~sbn_anomaly.train.trainer.BaseTrainer` loop (which reads ``batch[0]``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from sbn_anomaly.data.raw_digit_reader import RawDigitReader
from sbn_anomaly.data.raw_preprocess import preprocess_event

logger = logging.getLogger(__name__)


class RawWaveformStreamDataset(IterableDataset):
    """Stream per-channel pre-processed waveforms from flat raw-ADC ntuples.

    Parameters
    ----------
    file_paths:
        Flat raw-ADC ntuple path(s) (output of ``dump_rawdigits.C``).
    input_length:
        Tick length each waveform is padded/truncated to (must match the VAE).
    tree_name, step_size:
        Forwarded to :class:`RawDigitReader`.
    max_events:
        Cap on events read from ROOT.
    channels_per_event:
        If set, randomly sample this many channels per event (keeps epochs
        balanced across events and bounds memory).  ``None`` uses all channels.
    preprocess_kwargs:
        Extra keyword args forwarded to
        :func:`~sbn_anomaly.data.raw_preprocess.preprocess_event`
        (e.g. ``coherent_group_size``, ``scale``, ``remove_coherent``).
    seed:
        RNG seed for channel sampling.
    """

    def __init__(
        self,
        file_paths: Union[str, Path, Sequence[Union[str, Path]]],
        input_length: int = 4096,
        tree_name: str = "rawdigits",
        step_size: int = 32,
        max_events: Optional[int] = None,
        channels_per_event: Optional[int] = None,
        preprocess_kwargs: Optional[dict] = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.file_paths = file_paths
        self.input_length = int(input_length)
        self.tree_name = tree_name
        self.step_size = int(step_size)
        self.max_events = max_events
        self.channels_per_event = channels_per_event
        self.preprocess_kwargs = dict(preprocess_kwargs or {})
        self.preprocess_kwargs.setdefault("n_ticks", self.input_length)
        self.seed = int(seed)

    def __iter__(self) -> Iterator[tuple]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        rng = np.random.default_rng(self.seed + worker_id)

        reader = RawDigitReader(
            self.file_paths,
            tree_name=self.tree_name,
            step_size=self.step_size,
            max_events=self.max_events,
        )
        for ev_idx, ev in enumerate(reader):
            # Shard events across workers so each waveform is emitted once.
            if num_workers > 1 and (ev_idx % num_workers) != worker_id:
                continue
            wf = preprocess_event(ev.adc, ev.pedestal, **self.preprocess_kwargs)
            n_chan = wf.shape[0]
            if self.channels_per_event is not None and self.channels_per_event < n_chan:
                sel = rng.choice(n_chan, size=self.channels_per_event, replace=False)
            else:
                sel = range(n_chan)
            for c in sel:
                yield (torch.from_numpy(wf[c]).float(),)


class RawWaveformArrayDataset(Dataset):
    """Map-style dataset over a ``(n_waveforms, input_length)`` array.

    Parameters
    ----------
    waveforms:
        One of: an in-memory ``(n_waveforms, input_length)`` array; a path to a
        single ``.npy``/``.npz`` (key ``waveforms``); a glob string matching
        several shard files (e.g. ``data/raw_waveforms_train_*.npz``); or a list
        of such paths. Multiple files are loaded and concatenated, so the sharded
        output of ``write_waveform_shards`` can be trained on directly.
    """

    def __init__(self, waveforms: Union[np.ndarray, str, Path, list]) -> None:
        import glob as _glob

        paths: list = []
        if isinstance(waveforms, (list, tuple)):
            for w in waveforms:
                paths.extend(sorted(_glob.glob(str(w))) or [str(w)])
        elif isinstance(waveforms, (str, Path)):
            matches = sorted(_glob.glob(str(waveforms)))
            paths = matches if matches else [str(waveforms)]

        if paths:
            arrays = []
            for pth in paths:
                arr = np.load(pth, allow_pickle=True)
                if isinstance(arr, np.lib.npyio.NpzFile):
                    arr = arr["waveforms"]
                arrays.append(np.asarray(arr, dtype=np.float32))
            wf = np.concatenate(arrays, axis=0) if len(arrays) > 1 else arrays[0]
        else:
            wf = np.asarray(waveforms, dtype=np.float32)

        if wf.ndim != 2:
            raise ValueError(
                f"waveforms must be 2-D (n_waveforms, input_length), got {wf.ndim}-D"
            )
        self._wf = wf

    def __len__(self) -> int:
        return self._wf.shape[0]

    def __getitem__(self, idx: int) -> tuple:
        return (torch.from_numpy(self._wf[idx]).float(),)
