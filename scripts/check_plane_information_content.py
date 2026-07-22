#!/usr/bin/env python3
"""Check whether collection has less *information per window* than induction.

Hypothesis (not one of the original four): collection is usually the
best-calibrated, lowest-noise plane in SBND -- but that can come with fewer
hits per channel per window than induction. Per-plane standardization (see
check_standardization_per_plane.py) already corrects for a plane's absolute
*scale* (mean/std), but it does nothing about *why* that std is what it is.
If a channel only sees a handful of hits per window, the window-level
statistics computed from those hits (count, occupancy, mean-of-hit-integral)
are themselves noisy estimates purely from small sample size -- shot noise,
not detector physics. Standardization divides by that inflated std, so the
same absolute anomaly deviation buys fewer standard deviations on a plane
whose noise floor is wide for this reason. That would show up exactly as
"poor good/bad separation on collection" without any bug in the model,
features, or training loop.

This script tests the two halves of that claim directly from a good-run
events npz, with no model/torch/checkpoint required:

  1. Premise: does collection actually see fewer hits per window than
     induction? (median hits/window and occupancy per plane)

  2. Mechanism: is a channel's window-to-window count variance close to what
     pure Poisson sampling noise would predict at its hit rate, or is there
     real structure above that floor? For a channel with hits arriving
     ~independently at window-level rate lambda (mean hits/window), the
     count's coefficient of variation from Poisson statistics alone would be
     CV_theory = 1/sqrt(lambda). Comparing the *observed* CV(count) to this
     floor (ratio = CV_observed / CV_theory) tells you how much of a plane's
     window-level variance is irreducible counting noise (ratio near 1) vs.
     genuine structure (ratio >> 1).

How to read the result
-----------------------
- Collection has clearly lower median hits/window AND a ratio near 1 (mostly
  shot noise): this hypothesis is well supported. The fix is not more
  features/architecture -- it's giving collection more hits to average over
  per window (increase data.window_size, at the cost of time resolution)
  and/or a plane-specific anomaly threshold rather than comparing collection's
  raw score against a shared global one.
- Collection's hit rate is comparable to induction, or its ratio is well
  above 1 (variance dominated by real structure, not counting noise): this
  hypothesis isn't the explanation -- look elsewhere.

Usage
-----
    python scripts/check_plane_information_content.py \\
        --events data/good_events_val.npz \\
        --window-size 100 --stride 100 \\
        --channel-map configs/SBNDTPCChannelMap_v2_with_positions.csv

Or point --config at the training config to pick up window_size/stride/
channel_map from its `data` section automatically:
    python scripts/check_plane_information_content.py \\
        --config configs/graph_vae.yaml --events data/good_events_val.npz

Plane comes from `planes_flat` in the events npz if present, otherwise
--channel-map.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Pure-numpy helpers shared with the other diagnostic scripts -- no torch
# needed for any of this.
from compare_events_distributions import (  # noqa: E402
    _channel_plane_lookup,
    _channel_to_plane_lut,
    _group_indices,
    _windowed_channel_values,
)

logger = logging.getLogger("check_plane_information_content")


def _load_config_defaults(config_path: Optional[str]) -> dict:
    if not config_path:
        return {}
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("data", {})


def _per_channel_count_stats(channels: np.ndarray, counts: np.ndarray, n_windows: int) -> dict:
    """From (window,channel) count entries (zero-count windows absent), compute
    per-channel mean/std/occupancy *across all n_windows*, including the
    implicit zeros, via the sum/sum-of-squares trick (no need to materialize
    a dense windows x channels matrix).
    """
    if channels.size == 0:
        return {}
    unique_ch, order, start_idx = _group_indices(channels)
    counts_sorted = counts[order]
    out: dict = {}
    for i, c in enumerate(unique_ch):
        group = counts_sorted[start_idx[i]:start_idx[i + 1]]
        n_nonzero = group.size
        s, ss = float(group.sum()), float((group.astype(np.float64) ** 2).sum())
        mean = s / n_windows
        var = max(ss / n_windows - mean * mean, 0.0)
        std = var ** 0.5
        occupancy = n_nonzero / n_windows
        cv_obs = std / mean if mean > 1e-9 else float("nan")
        cv_theory = (1.0 / (mean ** 0.5)) if mean > 1e-9 else float("nan")
        ratio = (cv_obs / cv_theory) if (cv_theory and cv_theory > 1e-12) else float("nan")
        out[int(c)] = dict(
            mean_count=mean, std_count=std, occupancy=occupancy,
            cv_count_observed=cv_obs, cv_count_theory=cv_theory, ratio=ratio,
        )
    return out


def _per_channel_sum_stats(channels: np.ndarray, sums: np.ndarray, n_windows: int) -> dict:
    """Same sum/sum-of-squares trick, applied to per-window integral sum
    instead of hit count. No clean theoretical floor here (the compound
    Poisson-times-amplitude variance needs per-hit moments too), so this is
    reported for context only -- not used in the shot-noise verdict.
    """
    if channels.size == 0:
        return {}
    unique_ch, order, start_idx = _group_indices(channels)
    sums_sorted = sums[order]
    out: dict = {}
    for i, c in enumerate(unique_ch):
        group = sums_sorted[start_idx[i]:start_idx[i + 1]]
        s, ss = float(group.sum()), float((group.astype(np.float64) ** 2).sum())
        mean = s / n_windows
        var = max(ss / n_windows - mean * mean, 0.0)
        std = var ** 0.5
        cv = std / mean if abs(mean) > 1e-9 else float("nan")
        out[int(c)] = dict(mean_sum=mean, std_sum=std, cv_sum=cv)
    return out


def run(
    events_path: str,
    window_size: int,
    stride: int,
    channel_map: Optional[str],
    n_channels_override: Optional[int],
    max_windows: Optional[int],
    output_csv: Optional[str],
) -> None:
    logger.info("Loading %s ...", events_path)
    data = np.load(events_path, allow_pickle=False)
    if "channels_flat" not in data or "offsets" not in data:
        raise ValueError(f"{events_path}: not an events npz (missing channels_flat/offsets)")
    channels_flat = data["channels_flat"]
    integrals_flat = data["integrals_flat"]
    offsets = data["offsets"].astype(np.int64)
    n_channels = int(n_channels_override or data.get("n_channels", channels_flat.max() + 1))
    n_events = len(offsets) - 1

    if "planes_flat" in data:
        plane_lut = _channel_to_plane_lut(channels_flat, data["planes_flat"], n_channels)
    elif channel_map:
        plane_lut = _channel_plane_lookup(channel_map, n_channels)
    else:
        raise ValueError(
            "No plane info: events npz has no 'planes_flat' and no --channel-map given.")

    starts = list(range(0, n_events - window_size + 1, stride))
    n_windows = len(starts) if max_windows is None else min(len(starts), max_windows)
    if n_windows == 0:
        raise ValueError(f"No windows formed: n_events={n_events}, window_size={window_size}")
    logger.info("n_events=%d  window_size=%d  stride=%d  -> %d windows", n_events, window_size, stride, n_windows)

    logger.info("Forming windowed hit counts ...")
    w_ch_count, w_val_count = _windowed_channel_values(
        channels_flat, integrals_flat, offsets, window_size, stride, "count",
        max_windows=max_windows, label="count")
    logger.info("Forming windowed integral sums ...")
    w_ch_sum, w_val_sum = _windowed_channel_values(
        channels_flat, integrals_flat, offsets, window_size, stride, "sum",
        max_windows=max_windows, label="sum")

    count_stats = _per_channel_count_stats(w_ch_count, w_val_count, n_windows)
    sum_stats = _per_channel_sum_stats(w_ch_sum, w_val_sum, n_windows)

    rows = []
    for c, cs in count_stats.items():
        if not (0 <= c < n_channels):
            continue
        p = int(plane_lut[c])
        row = dict(channel=c, plane=p, **cs)
        ss = sum_stats.get(c)
        if ss:
            row.update(ss)
        rows.append(row)

    if not rows:
        raise ValueError("No channels had any hits across the analyzed windows -- nothing to report.")

    planes = sorted(set(r["plane"] for r in rows if r["plane"] >= 0))

    def _median(key, subset):
        vals = [r[key] for r in subset if key in r and not np.isnan(r[key])]
        return float(np.median(vals)) if vals else float("nan")

    print(f"\n{n_windows} windows analyzed (window_size={window_size}, stride={stride})\n")
    header = (f"{'plane':<7}{'n_ch':>7}{'med_hits/win':>15}{'med_occupancy':>15}"
              f"{'med_CV(count)':>16}{'med_CV_theory':>16}{'med_ratio':>12}{'med_CV(sum)':>14}")
    print(header)
    print("-" * len(header))
    plane_summary = {}
    for p in planes:
        subset = [r for r in rows if r["plane"] == p]
        med_hits = _median("mean_count", subset)
        med_occ = _median("occupancy", subset)
        med_cv_obs = _median("cv_count_observed", subset)
        med_cv_th = _median("cv_count_theory", subset)
        med_ratio = _median("ratio", subset)
        med_cv_sum = _median("cv_sum", subset)
        plane_summary[p] = dict(
            n_ch=len(subset), med_hits=med_hits, med_occ=med_occ,
            med_cv_obs=med_cv_obs, med_ratio=med_ratio,
        )
        print(f"{p:<7}{len(subset):>7}{med_hits:>15.3f}{med_occ:>15.3f}"
              f"{med_cv_obs:>16.3f}{med_cv_th:>16.3f}{med_ratio:>12.3f}{med_cv_sum:>14.3f}")

    print(
        "\nRead this as: 'med_hits/win' and 'med_occupancy' test the PREMISE (does this "
        "plane really see fewer hits per window?). 'med_ratio' (observed CV / Poisson-"
        "shot-noise CV) tests the MECHANISM: near 1.0 means a channel's window-to-window "
        "count variance is basically what you'd get from independent random hit "
        "occurrence at its own rate -- i.e. mostly irreducible small-N sampling noise, "
        "not genuine structure. Well above 1.0 means real structure dominates."
    )

    if len(planes) >= 2:
        lowest_hits_plane = min(plane_summary, key=lambda p: plane_summary[p]["med_hits"])
        lowest_ratio_plane = min(plane_summary, key=lambda p: plane_summary[p]["med_ratio"])
        print(
            f"\nLowest median hits/window: plane {lowest_hits_plane} "
            f"({plane_summary[lowest_hits_plane]['med_hits']:.3f}).  "
            f"Closest to the shot-noise floor (lowest ratio): plane {lowest_ratio_plane} "
            f"({plane_summary[lowest_ratio_plane]['med_ratio']:.3f})."
        )
        if lowest_hits_plane == lowest_ratio_plane and plane_summary[lowest_hits_plane]["med_ratio"] < 2.0:
            print(
                f"Plane {lowest_hits_plane} has both the fewest hits/window AND a "
                "near-shot-noise ratio -- consistent with the 'less information per "
                "window' hypothesis: its baseline variance looks mostly like counting "
                "noise from having few hits to average over, not genuine window-to-window "
                "structure. Standardizing by this std doesn't fix that -- it just gives "
                "you a noise floor to divide by. Consider a larger window_size (more hits "
                "to average, at the cost of time resolution) and/or per-plane anomaly "
                "thresholds instead of a single global one."
            )
        else:
            print(
                "No single plane stands out on both counts -- this hypothesis doesn't "
                "look like the dominant explanation here."
            )
    else:
        print("\nFewer than 2 planes had channels with hits -- no cross-plane comparison possible.")

    if output_csv:
        import pandas as pd
        pd.DataFrame(rows).sort_values(["plane", "channel"]).to_csv(output_csv, index=False)
        logger.info("Wrote per-channel table to %s", output_csv)


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", required=True, help="Good-run events .npz to analyze")
    ap.add_argument("--config", default=None,
                     help="Optional training config (e.g. configs/graph_vae.yaml) to pick up "
                          "window_size/stride/channel_map defaults from its data section")
    ap.add_argument("--window-size", type=int, default=None, help="Default: data.window_size in --config, else 100")
    ap.add_argument("--stride", type=int, default=None, help="Default: data.stride in --config, else same as --window-size")
    ap.add_argument("--channel-map", default=None,
                     help="CSV with offlchan/plane columns; default: data.channel_map in --config. "
                          "Not needed if the events npz has planes_flat.")
    ap.add_argument("--n-channels", type=int, default=None)
    ap.add_argument("--max-windows", type=int, default=None,
                     help="Cap the number of windows processed (fast first look on huge inputs)")
    ap.add_argument("--output-csv", default=None, help="Optional path to dump the per-channel table")
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data_cfg = _load_config_defaults(args.config)
    window_size = args.window_size if args.window_size is not None else int(data_cfg.get("window_size", 100))
    stride = args.stride if args.stride is not None else int(data_cfg.get("stride", window_size))
    channel_map = args.channel_map or data_cfg.get("channel_map")
    n_channels = args.n_channels if args.n_channels is not None else data_cfg.get("n_channels")
    run(
        events_path=args.events, window_size=window_size, stride=stride,
        channel_map=channel_map, n_channels_override=n_channels,
        max_windows=args.max_windows, output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
