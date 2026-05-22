"""Streaming reader for SBN ROOT files using uproot.

Each ROOT file may contain multiple trees (TPC waveforms, PMT waveforms, etc.).
``RootStreamer`` iterates over batches of events without loading the entire file
into memory, making it suitable for very large production datasets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Iterable, Optional, Union

import awkward as ak
import numpy as np
import uproot

logger = logging.getLogger(__name__)

# Canonical branch names used in SBN ntuples.  Override via config if your
# ntuple uses different names.
DEFAULT_TPC_BRANCHES = [
    "run",
    "subrun",
    "event",
    "tpc_waveform",       # float array per channel
    "tpc_channel",        # channel index
    "tpc_tick",           # time tick index
]

DEFAULT_PMT_BRANCHES = [
    "run",
    "subrun",
    "event",
    "pmt_waveform",       # float array per channel
    "pmt_channel",
    "pmt_time_ns",        # absolute timestamp in nanoseconds
]


class RootStreamer:
    """Stream events from one or more ROOT files in fixed-size batches.

    Parameters
    ----------
    file_paths:
        Paths to ``.root`` files (glob strings accepted via ``Path``).
    tree_name:
        Name of the TTree inside each ROOT file.
    branches:
        List of branch names to load.  ``None`` loads all branches.
    batch_size:
        Number of events per yielded batch.
    step_size:
        ``uproot`` internal step size; controls memory usage within a batch.
    """

    def __init__(
        self,
        file_paths: Union[str, Path, Iterable[Union[str, Path]]],
        tree_name: str,
        branches: Optional[list[str]] = None,
        batch_size: int = 512,
        step_size: int = 512,
    ) -> None:
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        self.file_paths = [_normalize_root_input_path(p) for p in file_paths]
        self.tree_name = tree_name
        self.branches = branches
        self.batch_size = batch_size
        self.step_size = step_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stream(self) -> Generator[ak.Array, None, None]:
        """Yield :class:`awkward.Array` batches, one per ``batch_size`` events.

        Batches are assembled across file boundaries so downstream code sees a
        continuous stream regardless of how the input is split into files.
        """
        buffer: list[ak.Array] = []
        buffered_events = 0

        for path in self.file_paths:
            logger.debug("Opening ROOT file: %s", path)
            try:
                with uproot.open(str(path)) as root_file:
                    if self.tree_name not in root_file:
                        logger.warning(
                            "Tree '%s' not found in %s – skipping.", self.tree_name, path
                        )
                        continue
                    tree = root_file[self.tree_name]
                    for chunk in tree.iterate(
                        self.branches,
                        step_size=self.step_size,
                        library="ak",
                    ):
                        buffer.append(chunk)
                        buffered_events += len(chunk)

                        while buffered_events >= self.batch_size:
                            batch, buffer, buffered_events = self._pop_batch(
                                buffer, buffered_events
                            )
                            yield batch
            except FileNotFoundError:
                logger.error("ROOT file not found: %s", path)
                raise
            except OSError as exc:
                logger.warning("Skipping unreadable ROOT file %s: %s", path, exc)
                continue

        # Yield remaining events that did not fill a complete batch.
        if buffer:
            yield ak.concatenate(buffer)

    def stream_numpy(
        self,
        feature_branch: str,
    ) -> Generator[np.ndarray, None, None]:
        """Convenience wrapper that returns NumPy arrays for a single branch."""
        for batch in self.stream():
            yield ak.to_numpy(ak.flatten(batch[feature_branch], axis=None))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pop_batch(
        self,
        buffer: list[ak.Array],
        buffered_events: int,
    ) -> tuple[ak.Array, list[ak.Array], int]:
        """Extract exactly ``self.batch_size`` events from the front of the buffer."""
        combined = ak.concatenate(buffer)
        batch = combined[: self.batch_size]
        remainder = combined[self.batch_size :]
        new_buffer = [remainder] if len(remainder) > 0 else []
        return batch, new_buffer, buffered_events - self.batch_size


def _normalize_root_input_path(path: Union[str, Path]) -> Union[str, Path]:
    """Preserve ROOT URLs while still normalizing local paths as ``Path`` objects."""
    path_str = str(path)
    if "://" in path_str:
        return path_str
    return Path(path)
