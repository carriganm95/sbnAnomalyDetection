"""Aggregate per-channel reconstruction errors into per-window anomaly scores.

The graph VAE inference saves ``node_scores`` of shape ``(N_windows, N_channels)``
(per-channel reconstruction error, NaN for inactive channels). This module turns
that into a per-window score with several interchangeable aggregators, so you can
compare strategies on the SAME saved errors without re-running the model.

Aggregators
-----------
- ``mean``           : nanmean over channels. Sensitive to diffuse, many-channel
  shifts; dilutes a localized fault across all channels.
- ``max``            : nanmax. Sensitive to single channels but noisy.
- ``topk_mean``      : mean of the top-``k`` channel errors. Robust middle ground
  for localized faults (a board lights up a block of channels).
- ``group_max_mean`` : pool channel errors within electronics groups (ASIC/FEMB),
  take the worst group's mean. Best for coherent board/ASIC failures and tells
  you which group -- the recommended default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def mean_score(node_scores: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmean(node_scores, axis=1)


def max_score(node_scores: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmax(node_scores, axis=1)


def topk_mean_score(node_scores: np.ndarray, k: int = 64) -> np.ndarray:
    n = node_scores.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        row = node_scores[i]
        v = row[~np.isnan(row)]
        if v.size == 0:
            continue
        kk = min(int(k), v.size)
        out[i] = np.sort(v)[-kk:].mean()
    return out


def group_max_mean_score(node_scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Per window: mean error within each group, then the max over groups.

    ``groups`` is a ``(N_channels,)`` int array (group id per channel, ``-1`` to
    ignore). Returns ``(N_windows,)``.
    """
    groups = np.asarray(groups)
    gids = np.unique(groups[groups >= 0])
    W = node_scores.shape[0]
    if gids.size == 0:
        return mean_score(node_scores)
    gmeans = np.full((W, gids.size), np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        for j, g in enumerate(gids):
            cols = np.where(groups == g)[0]
            if cols.size:
                gmeans[:, j] = np.nanmean(node_scores[:, cols], axis=1)
        return np.nanmax(gmeans, axis=1)


def aggregate_windows(
    node_scores: np.ndarray,
    aggregator: str = "group_max_mean",
    *,
    groups: Optional[np.ndarray] = None,
    k: int = 64,
) -> np.ndarray:
    """Dispatch to the named aggregator."""
    node_scores = np.asarray(node_scores, dtype=np.float64)
    if aggregator == "mean":
        return mean_score(node_scores)
    if aggregator == "max":
        return max_score(node_scores)
    if aggregator == "topk_mean":
        return topk_mean_score(node_scores, k=k)
    if aggregator == "group_max_mean":
        if groups is None:
            raise ValueError("group_max_mean requires a `groups` array (use --channel-map)")
        return group_max_mean_score(node_scores, groups)
    raise ValueError(f"unknown aggregator {aggregator!r}")


def channel_to_group(csv, level: str = "femb", num_channels: Optional[int] = None) -> np.ndarray:
    """Channel-id-indexed group array from the channel map.

    ``level``: ``"femb"`` (FEMB/board) or ``"asic"``.
    """
    import pandas as pd

    df = csv if hasattr(csv, "columns") else pd.read_csv(csv)
    if level == "asic":
        key = df["FEMBSerialNum"].astype(str) + "_" + df["asic"].astype(str)
    elif level == "femb":
        key = df["FEMBSerialNum"].astype(str)
    else:
        raise ValueError("level must be 'femb' or 'asic'")
    gid = key.factorize()[0]
    n = int(num_channels) if num_channels else int(df["offlchan"].max()) + 1
    out = np.full(n, -1, dtype=np.int64)
    chans = df["offlchan"].to_numpy()
    valid = chans < n
    out[chans[valid]] = gid[valid]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Aggregate per-channel errors -> per-window scores")
    p.add_argument("--scores", required=True, help="inference npz with 'node_scores' (W, C)")
    p.add_argument("--aggregator", default="group_max_mean",
                   choices=["mean", "max", "topk_mean", "group_max_mean"])
    p.add_argument("--channel-map", default=None, help="channel map CSV (for group_max_mean)")
    p.add_argument("--group-level", default="femb", choices=["femb", "asic"])
    p.add_argument("--k", type=int, default=64, help="k for topk_mean")
    p.add_argument("--output", default=None, help="optional .npy to save the per-window scores")
    args = p.parse_args(argv)

    arch = np.load(args.scores, allow_pickle=False)
    node_scores = arch["node_scores"] if "node_scores" in arch else arch["scores"]

    groups = None
    if args.aggregator == "group_max_mean":
        if not args.channel_map:
            p.error("group_max_mean needs --channel-map")
        groups = channel_to_group(args.channel_map, level=args.group_level,
                                  num_channels=node_scores.shape[1])

    win = aggregate_windows(node_scores, args.aggregator, groups=groups, k=args.k)
    finite = win[np.isfinite(win)]
    print(f"# aggregator={args.aggregator} windows={win.size} "
          f"mean={np.nanmean(win):.4g} p95={np.nanpercentile(finite,95):.4g} "
          f"max={np.nanmax(win):.4g}")
    if args.output:
        np.save(args.output, win)
        print(f"# saved per-window scores to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
