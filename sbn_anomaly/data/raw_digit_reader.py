"""Reader for the flat raw-ADC ntuple produced by ``scripts/dump_rawdigits.C``.

The gallery dumper writes a TTree ``rawdigits`` whose per-event layout is::

    run, subrun, event : int
    nchan, nticks      : int
    channel[nchan]     : int
    pedestal[nchan]    : float
    adc[nchan*nticks]  : short   (channel-major, tick-minor)

This module reads that tree with uproot (no art dictionaries required) and
yields one :class:`RawEvent` per entry, with ``adc`` reshaped to a dense
``(nchan, nticks)`` matrix.

Why a flat ntuple rather than reading ``raw::RawDigit`` directly with uproot?
RawDigits may be Huffman-compressed inside the art file and carry custom
streamers; gallery/LArSoft decompresses them correctly, so dumping to a flat
tree is the robust, version-independent bridge to Python.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_TREE = "rawdigits"


@dataclass
class RawEvent:
    """One event's worth of raw waveforms."""

    run: int
    subrun: int
    event: int
    channel: np.ndarray   # (nchan,) int32
    pedestal: np.ndarray  # (nchan,) float32
    adc: np.ndarray       # (nchan, nticks) float32 (raw ADC, NOT pedestal-subtracted)

    @property
    def n_channels(self) -> int:
        return int(self.adc.shape[0])

    @property
    def n_ticks(self) -> int:
        return int(self.adc.shape[1])


def _reshape_adc(adc_flat: np.ndarray, nchan: int, nticks: int) -> np.ndarray:
    """Reshape the flat channel-major adc array to ``(nchan, nticks)``."""
    adc_flat = np.asarray(adc_flat)
    expected = int(nchan) * int(nticks)
    if adc_flat.size != expected:
        raise ValueError(
            f"adc length {adc_flat.size} != nchan*nticks = {nchan}*{nticks} = {expected}"
        )
    return adc_flat.reshape(int(nchan), int(nticks)).astype(np.float32)


class RawDigitReader:
    """Iterate events from one or more flat raw-ADC ntuples.

    Parameters
    ----------
    file_paths:
        Path(s) to ``.root`` files written by ``dump_rawdigits.C``.
    tree_name:
        Name of the TTree (default ``"rawdigits"``).
    step_size:
        uproot iteration step (entries read into memory per chunk).
    max_events:
        Stop after this many events (``None`` = all).
    """

    def __init__(
        self,
        file_paths: Union[str, Path, Sequence[Union[str, Path]]],
        tree_name: str = DEFAULT_TREE,
        step_size: int = 64,
        max_events: Optional[int] = None,
    ) -> None:
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        self.file_paths: List[str] = [str(p) for p in file_paths]
        self.tree_name = tree_name
        self.step_size = int(step_size)
        self.max_events = max_events

    def __iter__(self) -> Iterator[RawEvent]:
        import uproot  # imported lazily so numpy-only code paths don't need it

        branches = ["run", "subrun", "event", "nchan", "nticks",
                    "channel", "pedestal", "adc"]
        n_yielded = 0
        for path in self.file_paths:
            with uproot.open(path) as f:
                if self.tree_name not in f:
                    logger.warning("Tree '%s' not found in %s; skipping",
                                   self.tree_name, path)
                    continue
                tree = f[self.tree_name]
                for chunk in tree.iterate(
                    branches, step_size=self.step_size, library="np"
                ):
                    n_in_chunk = len(chunk["run"])
                    for i in range(n_in_chunk):
                        nchan = int(chunk["nchan"][i])
                        nticks = int(chunk["nticks"][i])
                        if nchan == 0 or nticks == 0:
                            continue
                        adc = _reshape_adc(chunk["adc"][i], nchan, nticks)
                        yield RawEvent(
                            run=int(chunk["run"][i]),
                            subrun=int(chunk["subrun"][i]),
                            event=int(chunk["event"][i]),
                            channel=np.asarray(chunk["channel"][i], dtype=np.int32),
                            pedestal=np.asarray(chunk["pedestal"][i], dtype=np.float32),
                            adc=adc,
                        )
                        n_yielded += 1
                        if self.max_events is not None and n_yielded >= self.max_events:
                            return

    def read_all(self) -> List[RawEvent]:
        """Eagerly read every event into a list (use only for small samples)."""
        return list(self)
