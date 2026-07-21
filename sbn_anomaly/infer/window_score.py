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
import warnings
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
    # All-NaN groups/windows are expected (inactive channels) -> NaN result;
    # silence the benign "Mean of empty slice" / "All-NaN slice" warnings.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
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


def _aggregate_file(path: str, aggregator: str, group_args):
    """Return (per-window scores, provenance-or-None) for a scores npz."""
    arch = np.load(path, allow_pickle=False)
    node_scores = arch["node_scores"] if "node_scores" in arch else arch["scores"]
    groups = None
    if aggregator == "group_max_mean":
        csv, level = group_args
        groups = channel_to_group(csv, level=level, num_channels=node_scores.shape[1])
    win = aggregate_windows(node_scores, aggregator, groups=groups)
    prov = arch["provenance"] if "provenance" in getattr(arch, "files", []) else None
    return win, prov


def per_run_scores(win, prov, thr, stat="frac_above", run_key="run"):
    """Roll per-window scores up to one score per run.

    ``stat``: ``frac_above`` (fraction of a run's windows above the window
    threshold ``thr`` -- best for intermittent faults), ``max``, ``p99``, ``mean``.
    ``run_key``: ``run`` groups by run number; ``runsubrun`` by (run, subrun).
    Returns (run_ids, run_scores) or (None, None) if no provenance.
    """
    if prov is None:
        return None, None
    prov = np.asarray(prov)
    if run_key == "runsubrun":
        keys = prov[:, 0].astype(np.int64) * 1_000_000 + prov[:, 1].astype(np.int64)
    else:
        keys = prov[:, 0].astype(np.int64)
    win = np.asarray(win, dtype=np.float64)
    run_ids, run_scores = [], []
    for k in np.unique(keys):
        w = win[keys == k]
        w = w[np.isfinite(w)]
        if w.size == 0:
            continue
        if stat == "frac_above":
            s = float(np.mean(w > thr))
        elif stat == "max":
            s = float(np.max(w))
        elif stat == "p99":
            s = float(np.percentile(w, 99))
        else:
            s = float(np.mean(w))
        run_ids.append(int(k))
        run_scores.append(s)
    return np.array(run_ids), np.array(run_scores)


def stream_detect(win, prov, thr, n=3, m=2, instant=None, run_key="run"):
    """Streaming M-of-N persistence detector, per run, in window order.

    Walks each run's windows in order; raises an alarm at the first window where
    either (a) the score exceeds ``instant`` (acute single-window fault), or
    (b) at least ``m`` of the last ``n`` windows exceed ``thr`` (sustained fault).
    This is a real-time rule: it fires within a few windows, without waiting for
    the whole run.

    Returns dict run_id -> (alarmed: bool, latency_windows: int or -1).
    """
    if prov is None:
        return None
    prov = np.asarray(prov)
    if run_key == "runsubrun":
        keys = prov[:, 0].astype(np.int64) * 1_000_000 + prov[:, 1].astype(np.int64)
    else:
        keys = prov[:, 0].astype(np.int64)
    win = np.asarray(win, dtype=np.float64)

    result = {}
    seen = []
    for k in keys:                       # preserve first-occurrence run order
        if k not in seen:
            seen.append(k)
    for k in seen:
        idx = np.where(keys == k)[0]      # windows of this run, in order
        w = win[idx]
        recent = []                       # last n booleans (score > thr)
        alarmed, latency = False, -1
        for j, s in enumerate(w):
            hot = bool(np.isfinite(s) and s > thr)
            recent.append(hot)
            if len(recent) > n:
                recent.pop(0)
            fire = (instant is not None and np.isfinite(s) and s > instant) \
                or (sum(recent) >= m)
            if fire:
                alarmed, latency = True, j + 1   # windows consumed until alarm
                break
        result[int(k)] = (alarmed, latency)
    return result


def _confusion(pos_scores, neg_scores, thr, pos_label, neg_label):
    """Print confusion matrix + precision/recall/F1 (positive = pos_scores)."""
    tp = int(np.sum(pos_scores > thr)); fn = int(pos_scores.size - tp)
    fp = int(np.sum(neg_scores > thr)); tn = int(neg_scores.size - fp)
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    print(f"#                 pred {neg_label:<6} pred {pos_label:<6}")
    print(f"#   actual {neg_label:<6}  {tn:>10d}  {fp:>10d}")
    print(f"#   actual {pos_label:<6}  {fn:>10d}  {tp:>10d}")
    print(f"# precision={prec:.3f}  recall(TPR)={rec:.3f}  F1={f1:.3f}  FPR={fpr:.3f}")


def _plot_overlay(score_lists, labels, path, aggregator, threshold=None, nbins=60):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    finite = [np.asarray(s)[np.isfinite(s)] for s in score_lists]
    nonempty = [f for f in finite if f.size]
    allv = np.concatenate(nonempty) if nonempty else np.array([0.0, 1.0])
    lo, hi = float(np.min(allv)), float(np.max(allv))
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, nbins + 1)  # shared bin edges for every series

    fig, ax = plt.subplots(figsize=(8, 4))
    for v, lab in zip(finite, labels):
        ax.hist(v, bins=edges, alpha=0.5, density=True, label=f"{lab} (n={v.size})")
    if threshold is not None:
        ax.axvline(threshold, color="k", ls="--", lw=1.3,
                   label=f"threshold = {threshold:.4g}")
        ax.text(threshold, ax.get_ylim()[1], f" thr={threshold:.4g}",
                rotation=90, va="top", ha="left", fontsize=8, color="k")
    ax.set_xlabel(f"window score ({aggregator})")
    ax.set_ylabel("density")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

"""
def _plot_overlay(
    score_lists,
    labels,
    path,
    aggregator,
    threshold=None,
    nbins=60,
):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    finite = [
        np.asarray(scores)[np.isfinite(scores)]
        for scores in score_lists
    ]

    nonempty = [values for values in finite if values.size]
    all_values = (
        np.concatenate(nonempty)
        if nonempty
        else np.array([0.0, 1.0])
    )

    lo = float(np.min(all_values))
    hi = float(np.max(all_values))

    if hi <= lo:
        hi = lo + 1.0

    # Use the same bin edges for all distributions.
    edges = np.linspace(lo, hi, nbins + 1)

    fig, ax = plt.subplots(figsize=(8, 4))

    for values, label in zip(finite, labels):
        ax.hist(
            values,
            bins=edges,
            alpha=0.5,
            density=True,
            label=f"{label} (n={values.size})",
        )

    if threshold is not None:
        ax.axvline(
            threshold,
            color="k",
            linestyle="--",
            linewidth=1.3,
            label=f"Anomaly threshold = {threshold:.4g}",
        )

        ax.text(
            threshold,
            ax.get_ylim()[1],
            f" threshold = {threshold:.4g}",
            rotation=90,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=8,
            color="k",
        )

    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Window density")
    ax.set_yscale("log")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
"""


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
                   help="explicit window operating threshold; overrides --percentile")
    p.add_argument("--percentile", type=float, default=99.0,
                   help="good-set percentile used as the window operating threshold when --threshold is not set")
    p.add_argument("--per-run", action="store_true",
                   help="also roll windows up to one score per run and report run-level metrics")
    p.add_argument("--run-stat", default="frac_above",
                   choices=["frac_above", "max", "p99", "mean"],
                   help="how to score a run from its windows (default: fraction above the window threshold)")
    p.add_argument("--run-key", default="run", choices=["run", "runsubrun"])
    p.add_argument("--run-threshold", type=float, default=None,
                   help="run-level decision threshold (else good-run p90)")
    p.add_argument("--stream", action="store_true",
                   help="real-time M-of-N streaming detector: report detection latency "
                        "(bad) and false-alarm rate (good), no full-run wait")
    p.add_argument("--persist-n", type=int, default=3, help="streaming window count N")
    p.add_argument("--persist-m", type=int, default=2, help="streaming M-of-N trigger")
    p.add_argument("--instant-threshold", type=float, default=None,
                   help="single-window score that fires immediately (default: none)")
    args = p.parse_args(argv)

    if not (0.0 <= float(args.percentile) <= 100.0):
        p.error("--percentile must be between 0 and 100")

    arch = np.load(args.scores, allow_pickle=False)
    node_scores = arch["node_scores"] if "node_scores" in arch else arch["scores"]
    prov_good = arch["provenance"] if "provenance" in getattr(arch, "files", []) else None
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

    # Operating threshold: explicit --threshold, else the requested good-set percentile.
    thr = (
        float(args.threshold)
        if args.threshold is not None
        else float(np.nanpercentile(finite, float(args.percentile)))
    )

    win_cmp = prov_bad = None
    if args.compare:
        win_cmp, prov_bad = _aggregate_file(args.compare, args.aggregator, group_args)
        fc = win_cmp[np.isfinite(win_cmp)]
        print(f"# {args.labels[1]}: windows={win_cmp.size} mean={np.nanmean(win_cmp):.4g} "
              f"p95={np.nanpercentile(fc,95):.4g} max={np.nanmax(win_cmp):.4g}")

        # ---- window-level ----
        auc = separation_auc(win, win_cmp)
        thr_src = "user" if args.threshold is not None else f"good p{float(args.percentile):g}"
        print(f"# [window-level] AUC({args.labels[1]} vs {args.labels[0]})={auc:.4f}  "
              f"threshold={thr:.4g} ({thr_src})")
        print(f"# confusion (rows=actual, cols=predicted; positive={args.labels[1]}):")
        _confusion(fc, finite, thr, args.labels[1], args.labels[0])

        # ---- run-level rollup ----
        if args.per_run:
            if prov_good is None or prov_bad is None:
                print("# [run-level] skipped: scores npz has no 'provenance' "
                      "(re-run inference so provenance is saved).")
            else:
                _, rs_good = per_run_scores(win, prov_good, thr, args.run_stat, args.run_key)
                _, rs_bad = per_run_scores(win_cmp, prov_bad, thr, args.run_stat, args.run_key)
                run_thr = (args.run_threshold if args.run_threshold is not None
                           else float(np.percentile(rs_good, 90)) if rs_good.size else 0.0)
                r_auc = separation_auc(rs_good, rs_bad)
                print(f"# [run-level] stat={args.run_stat} runs: "
                      f"{args.labels[0]}={rs_good.size} {args.labels[1]}={rs_bad.size}  "
                      f"AUC={r_auc:.4f}  run_threshold={run_thr:.4g}"
                      + ("" if args.run_threshold is not None else " (good-run p90)"))
                print(f"# run confusion (positive={args.labels[1]}):")
                _confusion(rs_bad, rs_good, run_thr, args.labels[1], args.labels[0])

    # ---- real-time streaming detector (detection latency, no full-run wait) ----
    if args.stream:
        if prov_good is None:
            print("# [stream] skipped: scores npz has no 'provenance'.")
        else:
            gd = stream_detect(win, prov_good, thr, args.persist_n, args.persist_m,
                               args.instant_threshold, args.run_key)
            n_good = len(gd)
            fa = [v for v in gd.values() if v[0]]
            print(f"# [stream] N={args.persist_n} M={args.persist_m} "
                  f"window_thr={thr:.4g}"
                  + (f" instant_thr={args.instant_threshold:.4g}" if args.instant_threshold else ""))
            print(f"#   {args.labels[0]} runs={n_good}  false-alarm rate="
                  f"{len(fa)/max(1,n_good):.3f}")
            if prov_bad is not None:
                bd = stream_detect(win_cmp, prov_bad, thr, args.persist_n, args.persist_m,
                                   args.instant_threshold, args.run_key)
                n_bad = len(bd)
                det = [v[1] for v in bd.values() if v[0]]
                det_rate = len(det) / max(1, n_bad)
                if det:
                    lat = np.array(det)
                    print(f"#   {args.labels[1]} runs={n_bad}  detection rate={det_rate:.3f}  "
                          f"latency(windows): median={np.median(lat):.0f} "
                          f"mean={lat.mean():.1f} p90={np.percentile(lat,90):.0f} "
                          f"max={lat.max():.0f}")
                    print("#   (latency x window_size = events until flagged)")
                else:
                    print(f"#   {args.labels[1]} runs={n_bad}  detection rate=0.000 "
                          "(no bad run alarmed — loosen M/N or lower threshold)")

    if args.plot:
        # Always draw the operating threshold line, labelled with its value.
        if win_cmp is not None:
            _plot_overlay([win, win_cmp], args.labels, args.plot, args.aggregator, thr)
        else:
            _plot_overlay([win], [args.labels[0]], args.plot, args.aggregator, thr)
        print(f"# saved plot to {args.plot}")

    if args.output:
        np.save(args.output, win)
        print(f"# saved per-window scores to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
