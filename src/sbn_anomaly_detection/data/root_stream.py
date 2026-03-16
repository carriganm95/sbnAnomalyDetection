"""Lazy streaming reader for SBN ROOT files using uproot.iterate.

Yields batches of numpy/awkward arrays for the requested branches without
loading the entire file into memory.
"""

from __future__ import annotations

import glob
import logging
from typing import Iterator, List, Optional, Sequence, Union

import awkward as ak
import numpy as np
import uproot

logger = logging.getLogger(__name__)


def expand_paths(patterns: Union[str, Sequence[str]]) -> List[str]:
    """Expand shell globs and return a sorted list of resolved file paths."""
    if isinstance(patterns, str):
        patterns = [patterns]
    paths: List[str] = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern, recursive=True))
        if not matched:
            logger.warning("No files matched pattern: %s", pattern)
        paths.extend(matched)
    return paths


def stream_arrays(
    root_files: Union[str, Sequence[str]],
    branches: Sequence[str],
    tree_name: str = "events",
    step_size: int = 1000,
    max_events: Optional[int] = None,
    library: str = "np",
) -> Iterator[dict]:
    """Yield batches of arrays from one or more ROOT files.

    Parameters
    ----------
    root_files:
        Single glob pattern, list of file paths, or list of glob patterns.
    branches:
        Branch names to read from the TTree.
    tree_name:
        Name of the TTree inside each ROOT file.
    step_size:
        Number of entries per batch yielded by ``uproot.iterate``.
    max_events:
        If given, stop streaming after this many total events.
    library:
        Array library passed to uproot: ``"np"`` (numpy) or ``"ak"`` (awkward).

    Yields
    ------
    dict
        Mapping branch name → array for the current batch.
    """
    paths = expand_paths(root_files)
    if not paths:
        raise FileNotFoundError(f"No ROOT files found for patterns: {root_files}")

    # Build list of (path, tree) tuples for uproot
    file_tree_pairs = [f"{p}:{tree_name}" for p in paths]

    total_seen = 0
    for batch in uproot.iterate(
        file_tree_pairs,
        expressions=list(branches),
        step_size=step_size,
        library=library,
    ):
        if max_events is not None:
            remaining = max_events - total_seen
            if remaining <= 0:
                break
            # Trim the batch if it overshoots the limit
            if library == "np":
                batch = {k: v[:remaining] for k, v in batch.items()}
            else:
                batch = {k: v[:remaining] for k, v in batch.items()}

        n_events = len(next(iter(batch.values())))
        total_seen += n_events
        logger.debug("Yielding batch of %d events (total so far: %d)", n_events, total_seen)
        yield batch

        if max_events is not None and total_seen >= max_events:
            break


def stream_to_numpy(
    root_files: Union[str, Sequence[str]],
    branches: Sequence[str],
    tree_name: str = "events",
    step_size: int = 1000,
    max_events: Optional[int] = None,
) -> Iterator[np.ndarray]:
    """Convenience wrapper that yields 2-D numpy arrays (events × branches).

    Each yielded array has shape ``(batch_size, len(branches))``.
    Branches must be scalar-valued per event (e.g. summed charge, hit count).
    """
    for batch in stream_arrays(
        root_files,
        branches,
        tree_name=tree_name,
        step_size=step_size,
        max_events=max_events,
        library="np",
    ):
        cols = [batch[b].astype(np.float32) for b in branches]
        yield np.stack(cols, axis=1)
