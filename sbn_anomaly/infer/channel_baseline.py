"""Compute a per-channel (mu, sigma) baseline from training-set-only good scores.

Pools ``node_scores`` (shape ``(N_windows, N_channels)``) from one or more inference
npz files -- the ones from your TRAIN split of good runs, never the held-out good set
you evaluate separation on -- and computes, per channel: the mean, the std, and how
many non-NaN (active-channel) windows contributed.

Feed the saved baseline into ``window_score.py --aggregator zscore_mean --baseline
<this output>`` to z-score each channel against its own training-set baseline before
combining across channels. See ``window_score.py``'s module docstring for why that
aggregator exists (it targets a small, consistent shift spread across most channels --
the kind of signal raw-magnitude aggregators like ``mean``/``topk_mean``/
``group_max_mean`` can miss).

IMPORTANT: fit this only on train-set windows, and evaluate zscore_mean separation on a
disjoint held-out good set (+ bad). Fitting mu/sigma on the same windows you evaluate
on is optimistic -- the z-scores on that set will look tighter than they'd be on data
the baseline never saw.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np


def compute_channel_baseline(
    node_scores_list: Sequence[np.ndarray],
    min_count: int = 20,
    ddof: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool windows from several ``(W_i, C)`` arrays and return per-channel (mu, sigma, n).

    Channels with fewer than ``min_count`` non-NaN (active) windows, or a non-finite /
    non-positive std, get ``mu=sigma=NaN`` -- too few calibration samples to trust, and
    ``zscore_mean_score`` excludes them from the per-window mean rather than dividing
    by a near-zero or NaN sigma.
    """
    shapes = {a.shape[1] for a in node_scores_list}
    if len(shapes) != 1:
        raise ValueError(f"all inputs must have the same channel count, got {shapes}")
    combined = np.concatenate(
        [np.asarray(a, dtype=np.float64) for a in node_scores_list], axis=0
    )
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        n = np.sum(~np.isnan(combined), axis=0)
        mu = np.nanmean(combined, axis=0)
        sigma = np.nanstd(combined, axis=0, ddof=ddof)
    invalid = (n < min_count) | ~np.isfinite(sigma) | (sigma <= 0)
    mu = mu.copy()
    sigma = sigma.copy()
    mu[invalid] = np.nan
    sigma[invalid] = np.nan
    return mu, sigma, n


def _load_node_scores(path: str) -> np.ndarray:
    arch = np.load(path, allow_pickle=False)
    return arch["node_scores"] if "node_scores" in arch else arch["scores"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Compute a per-channel mu/sigma baseline from training-set-only "
                     "good scores, for use with window_score.py --aggregator zscore_mean"
    )
    p.add_argument("--scores", nargs="+", required=True,
                    help="one or more TRAIN-split good-run scores .npz files "
                         "(node_scores (W, C)); windows from all files are pooled")
    p.add_argument("--min-count", type=int, default=20,
                    help="channels with fewer than this many non-NaN windows get an "
                         "invalid (NaN) baseline and are excluded from zscore_mean")
    p.add_argument("--output", required=True, help="output .npz path (mu, sigma, n)")
    args = p.parse_args(argv)

    node_scores_list = []
    for path in args.scores:
        arr = _load_node_scores(path)
        node_scores_list.append(arr)
        print(f"# loaded {path}: windows={arr.shape[0]} channels={arr.shape[1]}")

    mu, sigma, n = compute_channel_baseline(node_scores_list, min_count=args.min_count)

    total_windows = sum(a.shape[0] for a in node_scores_list)
    num_channels = mu.shape[0]
    n_valid = int(np.sum(np.isfinite(mu)))
    print(f"# pooled {total_windows} windows across {len(args.scores)} file(s), "
          f"{num_channels} channels")
    print(f"# valid baseline: {n_valid}/{num_channels} channels "
          f"(min_count={args.min_count}); {num_channels - n_valid} excluded "
          "(too few active windows or zero variance)")
    finite_mu = mu[np.isfinite(mu)]
    finite_sigma = sigma[np.isfinite(sigma)]
    if finite_mu.size:
        print(f"# mu:    mean={finite_mu.mean():.4g}  p50={np.median(finite_mu):.4g}  "
              f"max={finite_mu.max():.4g}")
        print(f"# sigma: mean={finite_sigma.mean():.4g}  p50={np.median(finite_sigma):.4g}  "
              f"max={finite_sigma.max():.4g}")

    out_path = Path(args.output)
    if out_path.suffix.lower() != ".npz":
        out_path = out_path.with_suffix(".npz")
    np.savez(
        out_path,
        mu=mu.astype(np.float64),
        sigma=sigma.astype(np.float64),
        n=n.astype(np.int64),
        num_channels=np.array(num_channels, dtype=np.int64),
        min_count=np.array(args.min_count, dtype=np.int64),
        source_files=np.array(args.scores, dtype="U512"),
    )
    print(f"# saved baseline to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
