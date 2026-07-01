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


def separation_auc(good: np.ndarray, bad: np.ndarray) -> float:
    """AUC = P(a random bad window scores higher than a random good one).

    Rank-based (Mann-Whitney U), no sklearn needed. 0.5 = no separation, 1.0 =
    perfect. NaNs are dropped.
    """
    g = np.asarray(good)[np.isfinite(good)]
    b = np.asarray(bad)[np.isfinite(bad)]
    if g.size == 0 or b.size == 0:
        return float("nan")
    allv = np.concatenate([g, b])
    order = allv.argsort()
    sorted_v = allv[order]
    # Average (mid) ranks so ties don't bias the statistic.
    ranks_sorted = np.empty(sorted_v.size, dtype=np.float64)
    i = 0
    n = sorted_v.size
    while i < n:
        j = i
        while j < n and sorted_v[j] == sorted_v[i]:
            j += 1
        ranks_sorted[i:j] = (i + 1 + j) / 2.0  # mean of 1-based ranks i+1..j
        i = j
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    rank_bad = ranks[g.size:].sum()
    u = rank_bad - b.size * (b.size + 1) / 2.0
    return float(u / (g.size * b.size))


def _aggregate_file(path: str, aggregator: str, group_args) -> np.ndarray:
    arch = np.load(path, allow_pickle=False)
    node_scores = arch["node_scores"] if "node_scores" in arch else arch["scores"]
    groups = None
    if aggregator == "group_max_mean":
        csv, level = group_args
        groups = channel_to_group(csv, level=level, num_channels=node_scores.shape[1])
    return aggregate_windows(node_scores, aggregator, groups=groups)


def _plot_overlay(score_lists, labels, path, aggregator, threshold=None):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    for s, lab in zip(score_lists, labels):
        v = np.asarray(s)[np.isfinite(s)]
        ax.hist(v, bins=60, alpha=0.5, density=True, label=f"{lab} (n={v.size})")
    if threshold is not None:
        ax.axvline(threshold, color="k", ls="--", lw=1, label=f"threshold={threshold:g}")
    ax.set_xlabel(f"window score ({aggregator})")
    ax.set_ylabel("density")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Aggregate per-channel errors -> per-window scores")
    p.add_argument("--scores", required=True, help="inference npz with 'node_scores' (W, C)")
    p.add_argument("--aggregator", default="group_max_mean",
                   choices=["mean", "max", "topk_mean", "group_max_mean"])
    p.add_argument("--channel-map", default=None, help="channel map CSV (for group_max_mean)")
    p.add_argument("--group-level", default="femb", choices=["femb", "asic"])
    p.add_argument("--k", type=int, default=64, help="k for topk_mean")
    p.add_argument("--output", default=None, help="optional .npy to save the per-window scores")
    p.add_argument("--plot", default=None, help="save a score-distribution PNG to this path")
    p.add_argument("--compare", default=None,
                   help="second scores npz (e.g. bad runs) to overlay + compute AUC vs --scores")
    p.add_argument("--labels", nargs=2, default=["good", "bad"], metavar=("A", "B"))
    p.add_argument("--threshold", type=float, default=None,
                   help="draw a threshold line and report TPR/FPR at it")
    args = p.parse_args(argv)

    arch = np.load(args.scores, allow_pickle=False)
    node_scores = arch["node_scores"] if "node_scores" in arch else arch["scores"]
    if args.aggregator == "group_max_mean" and not args.channel_map:
        p.error("group_max_mean needs --channel-map")
    group_args = (args.channel_map, args.group_level)

    groups = None
    if args.aggregator == "group_max_mean":
        groups = channel_to_group(args.channel_map, level=args.group_level,
                                  num_channels=node_scores.shape[1])
    win = aggregate_windows(node_scores, args.aggregator, groups=groups, k=args.k)
    finite = win[np.isfinite(win)]
    print(f"# {args.labels[0]}: aggregator={args.aggregator} windows={win.size} "
          f"mean={np.nanmean(win):.4g} p95={np.nanpercentile(finite,95):.4g} "
          f"p99={np.nanpercentile(finite,99):.4g} max={np.nanmax(win):.4g}")

    win_cmp = None
    if args.compare:
        win_cmp = _aggregate_file(args.compare, args.aggregator, group_args)
        fc = win_cmp[np.isfinite(win_cmp)]
        print(f"# {args.labels[1]}: windows={win_cmp.size} mean={np.nanmean(win_cmp):.4g} "
              f"p95={np.nanpercentile(fc,95):.4g} max={np.nanmax(win_cmp):.4g}")
        auc = separation_auc(win, win_cmp)
        print(f"# separation AUC ({args.labels[1]} vs {args.labels[0]}) = {auc:.4f}")
        # If a threshold from good-run p99 or --threshold, report FPR/TPR.
        thr = args.threshold if args.threshold is not None else float(np.nanpercentile(finite, 99))
        fpr = float(np.mean(finite > thr))
        tpr = float(np.mean(fc > thr))
        print(f"# at threshold={thr:.4g} (good p99 unless --threshold): "
              f"FPR={fpr:.3f}  TPR({args.labels[1]})={tpr:.3f}")

    if args.plot:
        if win_cmp is not None:
            _plot_overlay([win, win_cmp], args.labels, args.plot, args.aggregator, args.threshold)
        else:
            _plot_overlay([win], [args.labels[0]], args.plot, args.aggregator, args.threshold)
        print(f"# saved plot to {args.plot}")

    if args.output:
        np.save(args.output, win)
        print(f"# saved per-window scores to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
