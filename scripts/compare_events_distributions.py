#!/usr/bin/env python3
"""Compare variable distributions between two sparse "events" .npz files.

Takes two events npz files -- the compact sparse format written by
``SparseWindowDatasetPyG.save_events()`` / ``materialize_windows.py``
(keys: ``channels_flat``, ``integrals_flat``, ``offsets``, ``n_channels``,
and optionally ``times_flat``/``widths_flat``/``sumadcs_flat``/``mults_flat``/
``hassps_flat``/``planes_flat``) -- and writes ROOT histograms comparing them
at three granularities: per channel, per plane, and summed over the whole
detector. Output is a single ROOT file (written with uproot, no PyROOT
needed) rather than one PDF per channel, since a full detector has
thousands of channels.

This does NOT require re-running any model or torch/torch_geometric -- it
works directly off the materialized events npz with just numpy/pandas/uproot.

Two comparison modes, both can be run together:

  Raw per-hit (default, always on unless --no-raw-hit): histograms every
  individual hit's value. Answers "does this quantity differ hit-by-hit?"

  Windowed (--windowed): forms the same event windows the model trains on
  (--window-size consecutive events, matching data.window_size in your
  training config) and reduces each (window, channel) group to a summary
  statistic -- mean, median, stdev, sum, min, max, or count -- before
  histogramming. This is closer to "the actual numbers reaching the model",
  since node features are themselves per-window aggregates, not raw hits;
  it also lets you check statistics (like median) the model doesn't
  currently compute as a node_feature.

Neither mode requires the model or a training run -- if a variable/stat
shows no separation here, adding it as a node_feature is unlikely to help;
if it does show separation but the model still isn't picking it up, the
problem is more likely standardization/architecture/training than features.

Layout in the output ROOT file:
    all_detector/<variable>_<label>
    per_plane/plane<p>/<variable>_<label>
    per_channel/ch<channel>/<variable>_<label>
    windowed/<variable>/<stat>/all_detector_<label>
    windowed/<variable>/<stat>/per_plane/plane<p>/<label>
    windowed/<variable>/<stat>/per_channel/ch<channel>/<label>   (opt-in, see --windowed-per-channel)

Browse with ``root -l compare.root`` then ``TBrowser b``, or with uproot:
    import uproot; f = uproot.open("compare.root")
    f["per_channel/ch04200/width_good"].to_hist()
    f["windowed/width/median/per_channel/ch04200/good"].to_hist()

Examples
--------
Full detector, all available variables (integral/time/width/sumadc/mult/hasSP
-- whichever are present in either file), raw per-hit only:
    python scripts/compare_events_distributions.py \\
        --file-a data/events_good.npz --label-a good \\
        --file-b data/events_bad.npz  --label-b bad \\
        --output compare.root

Add windowed mean/median/stdev (window_size=100 to match configs/graph_vae.yaml),
at the all-detector and per-plane level only (cheap, always sensible defaults):
    python scripts/compare_events_distributions.py \\
        --file-a data/events_good.npz --file-b data/events_bad.npz \\
        --windowed --window-size 100 --output compare.root

Add the (larger, slower) per-channel windowed breakdown too, restricted to the
collection-plane region flagged in the DQM plot:
    python scripts/compare_events_distributions.py \\
        --file-a data/events_good.npz --file-b data/events_bad.npz \\
        --windowed --window-size 100 --windowed-per-channel \\
        --channel-range 4000 5500 --output compare_collection.root

Skip the raw per-hit section entirely and just look at windowed values:
    python scripts/compare_events_distributions.py \\
        --file-a data/events_good.npz --file-b data/events_bad.npz \\
        --no-raw-hit --windowed --output compare_windowed_only.root

Per-plane grouping uses ``planes_flat`` if the npz has it; otherwise pass
--channel-map to derive plane from channel id via the SBND channel map CSV.
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

logger = logging.getLogger("compare_events_distributions")

# Per-hit variables we know how to pull out of an events npz: display name -> npz key.
_VARIABLES = {
    "integral": "integrals_flat",
    "time": "times_flat",
    "width": "widths_flat",
    "sumadc": "sumadcs_flat",
    "mult": "mults_flat",
    "hasSP": "hassps_flat",
}

# Per-(window, channel) reduction stats available in --window-stats. "count"
# is handled separately (group size, no reduction function needed).
_WINDOW_STAT_FUNCS = {
    "mean": np.mean,
    "median": np.median,
    "stdev": np.std,
    "sum": np.sum,
    "min": np.min,
    "max": np.max,
}


def _load_events(path: str) -> dict:
    """Load the subset of an events npz this script needs.

    Returns a dict with 'channels_flat', 'offsets', 'n_channels', optionally
    'planes_flat', and one entry per variable in _VARIABLES that is present.
    """
    data = np.load(path, allow_pickle=False)
    if "channels_flat" not in data:
        raise ValueError(f"{path}: not an events npz (missing 'channels_flat')")
    out: dict = {
        "channels_flat": data["channels_flat"],
        "offsets": data["offsets"] if "offsets" in data else None,
        "n_channels": int(data["n_channels"]) if "n_channels" in data else int(data["channels_flat"].max()) + 1,
    }
    for name, key in _VARIABLES.items():
        if key in data:
            out[name] = data[key]
    if "planes_flat" in data:
        out["planes_flat"] = data["planes_flat"]
    return out


def _channel_plane_lookup(channel_map_csv: str, n_channels: int) -> np.ndarray:
    """Return plane[c] for c in [0, n_channels) from the channel-map CSV (-1 if unmapped)."""
    import pandas as pd

    df = pd.read_csv(channel_map_csv)
    lut = np.full(n_channels, -1, dtype=np.int32)
    offl = df["offlchan"].to_numpy()
    plane = df["plane"].to_numpy()
    valid = (offl >= 0) & (offl < n_channels)
    lut[offl[valid]] = plane[valid]
    return lut


def _hist_range(values_list, mode: str, lo_pct: float, hi_pct: float) -> tuple:
    """Shared (lo, hi) histogram range across one or more value arrays."""
    nonempty = [v for v in values_list if v is not None and len(v)]
    if not nonempty:
        return 0.0, 1.0
    all_vals = np.concatenate(nonempty)
    if mode == "minmax":
        lo, hi = float(all_vals.min()), float(all_vals.max())
    else:
        lo, hi = float(np.percentile(all_vals, lo_pct)), float(np.percentile(all_vals, hi_pct))
    if lo >= hi:
        lo, hi = lo - 0.5, hi + 0.5
    return lo, hi


def _group_indices(keys: np.ndarray):
    """One sort, reusable for every variable's value array.

    Returns (unique_keys, order, start_idx) such that for value array `v`
    aligned with `keys`, ``v[order][start_idx[i]:start_idx[i+1]]`` are all
    the entries with key == unique_keys[i].
    """
    order = np.argsort(keys, kind="stable")
    ks = keys[order]
    unique_keys, start_idx = np.unique(ks, return_index=True)
    start_idx = np.append(start_idx, len(ks))
    return unique_keys, order, start_idx


def _channel_to_plane_lut(channels_flat: np.ndarray, planes_flat: np.ndarray, n_channels: int) -> np.ndarray:
    """Collapse a per-hit (channel, plane) pair array into a per-channel-id LUT.

    Plane is constant per channel, so this just takes the plane seen on the
    first hit of each channel. Used to look up plane for windowed
    (channel, value) pairs, where we no longer have a per-hit planes_flat.
    """
    lut = np.full(n_channels, -1, dtype=np.int32)
    unique_ch, order, start_idx = _group_indices(channels_flat)
    planes_sorted = planes_flat[order]
    for i, c in enumerate(unique_ch):
        if 0 <= c < n_channels:
            lut[c] = planes_sorted[start_idx[i]]
    return lut


def _windowed_channel_values(
    channels_flat: np.ndarray,
    value_arr: np.ndarray,
    offsets: np.ndarray,
    window_size: int,
    stride: int,
    stat: str,
    max_windows: Optional[int] = None,
    progress_every: int = 200,
    label: str = "",
) -> tuple:
    """Reduce each (window, channel) group of hits to one number via `stat`.

    Mirrors the windowing SparseWindowDatasetPyG trains on (window_size
    consecutive events, sliding by `stride`), so these are the same
    (window, channel) groups the model's node features are computed from --
    just reduced with numpy directly (supports "median", which the model's
    node_features do not).

    Returns (channels, values) with one entry per (window, channel-with-hits)
    pair -- the same shape of output as the raw per-hit arrays, so it can be
    fed through the same _group_indices-based histogram writers.
    """
    if offsets is None:
        raise ValueError("Windowed mode requires 'offsets' in the events npz (both files).")
    n_events = len(offsets) - 1
    if n_events < window_size:
        logger.warning(
            "label=%s: only %d events, fewer than window_size=%d -- no windows formed.",
            label, n_events, window_size)
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    starts = list(range(0, n_events - window_size + 1, stride))
    if max_windows is not None:
        starts = starts[:max_windows]

    out_ch: list = []
    out_val: list = []
    stat_fn = None if stat == "count" else _WINDOW_STAT_FUNCS[stat]
    for wi, s in enumerate(starts):
        h_start = int(offsets[s])
        h_end = int(offsets[s + window_size])
        if h_end <= h_start:
            continue
        ch = channels_flat[h_start:h_end]
        val = value_arr[h_start:h_end]
        unique_c, order, start_idx = _group_indices(ch)
        val_sorted = val[order]
        for i, c in enumerate(unique_c):
            group = val_sorted[start_idx[i]:start_idx[i + 1]]
            out_ch.append(int(c))
            out_val.append(float(group.size) if stat == "count" else float(stat_fn(group)))
        if (wi + 1) % progress_every == 0:
            logger.info("  [%s] windowed: %d/%d windows processed", label, wi + 1, len(starts))

    return np.array(out_ch, dtype=np.int64), np.array(out_val, dtype=np.float32)


# ---------------------------------------------------------------------------
# Shared histogram writers -- operate on any (channels, values) pair, used for
# both the raw per-hit section and the windowed section.
# ---------------------------------------------------------------------------

def _write_all_detector(f, key: str, values: np.ndarray, bins: int, rng: tuple):
    if values.size == 0:
        return 0
    counts, edges = np.histogram(values, bins=bins, range=rng)
    f[key] = (counts.astype(np.float64), edges)
    return 1


def _write_per_plane(f, prefix: str, channels: np.ndarray, values: np.ndarray,
                      planes: np.ndarray, bins: int, rng: tuple, suffix: str) -> int:
    if planes is None or channels.size == 0:
        return 0
    unique_p, order, start_idx = _group_indices(planes)
    vals_sorted = values[order]
    n = 0
    for i, p in enumerate(unique_p):
        if p < 0:
            continue
        vals = vals_sorted[start_idx[i]:start_idx[i + 1]]
        if vals.size == 0:
            continue
        counts, edges = np.histogram(vals, bins=bins, range=rng)
        f[f"{prefix}/plane{int(p)}/{suffix}"] = (counts.astype(np.float64), edges)
        n += 1
    return n


def _write_per_channel(f, prefix: str, channels: np.ndarray, values: np.ndarray,
                        chan_range: tuple, bins: int, rng: tuple, suffix: str,
                        progress_every: int, label: str) -> int:
    lo, hi = chan_range
    mask = (channels >= lo) & (channels < hi)
    ch = channels[mask]
    vals_all = values[mask]
    if ch.size == 0:
        logger.warning("No entries in channel range [%d, %d) for label=%s (%s)", lo, hi, label, prefix)
        return 0
    unique_c, order, start_idx = _group_indices(ch)
    vals_sorted = vals_all[order]
    n = 0
    for i, c in enumerate(unique_c):
        vals = vals_sorted[start_idx[i]:start_idx[i + 1]]
        counts, edges = np.histogram(vals, bins=bins, range=rng)
        f[f"{prefix}/ch{int(c):05d}/{suffix}"] = (counts.astype(np.float64), edges)
        n += 1
        if (i + 1) % progress_every == 0:
            logger.info("  [%s] %s: %d/%d channels written", label, prefix, i + 1, len(unique_c))
    return n


def compare(
    file_a: str,
    file_b: str,
    output: str,
    label_a: str = "good",
    label_b: str = "bad",
    variables=None,
    bins: int = 60,
    range_mode: str = "percentile",
    range_percentiles=(0.1, 99.9),
    channel_map: Optional[str] = None,
    channel_range=None,
    do_raw_hit: bool = True,
    do_per_channel: bool = True,
    do_per_plane: bool = True,
    do_all_detector: bool = True,
    progress_every: int = 1000,
    windowed: bool = False,
    window_size: int = 100,
    window_stride: Optional[int] = None,
    window_stats=("mean", "median", "stdev"),
    window_variables=None,
    windowed_per_channel: bool = False,
    max_windows: Optional[int] = None,
    uproot_module=None,
) -> int:
    """Build the comparison ROOT file. Returns the number of histograms written.

    `uproot_module` is exposed for testing (inject a fake uproot without a
    real install); normal callers should leave it as None.
    """
    uproot = uproot_module
    if uproot is None:
        import uproot as uproot  # noqa: PLW0127 (real import for normal use)

    logger.info("Loading %s (label=%s) and %s (label=%s) ...", file_a, label_a, file_b, label_b)
    a = _load_events(file_a)
    b = _load_events(file_b)

    all_vars = sorted(set(k for k in _VARIABLES if k in a or k in b))
    variables = list(variables) if variables else all_vars
    unknown = [v for v in variables if v not in _VARIABLES]
    if unknown:
        raise ValueError(f"Unknown variables {unknown}; choose from {list(_VARIABLES)}")
    missing_a = [v for v in variables if v not in a]
    missing_b = [v for v in variables if v not in b]
    if missing_a:
        logger.warning("Not in %s (label=%s), skipped there: %s", file_a, label_a, missing_a)
    if missing_b:
        logger.warning("Not in %s (label=%s), skipped there: %s", file_b, label_b, missing_b)
    variables = [v for v in variables if v in a or v in b]
    if not variables:
        raise ValueError("No comparable variables found in either file.")
    logger.info("Comparing variables: %s", variables)

    if windowed:
        unknown_stats = [s for s in window_stats if s != "count" and s not in _WINDOW_STAT_FUNCS]
        if unknown_stats:
            raise ValueError(f"Unknown window_stats {unknown_stats}; choose from "
                              f"{list(_WINDOW_STAT_FUNCS) + ['count']}")

    n_channels = max(a["n_channels"], b["n_channels"])

    plane_lut = None
    if channel_map and ("planes_flat" not in a or "planes_flat" not in b):
        plane_lut = _channel_plane_lookup(channel_map, n_channels)

    def planes_for(d):
        if "planes_flat" in d:
            return d["planes_flat"]
        if plane_lut is not None:
            ch = d["channels_flat"]
            out = np.full(ch.shape, -1, dtype=np.int32)
            valid = ch < n_channels
            out[valid] = plane_lut[ch[valid]]
            return out
        return None

    # Per-channel-id plane LUT (for windowed mode, where values are indexed
    # by channel rather than by hit). Prefer planes_flat, else channel_map.
    def channel_plane_lut_for(d):
        if plane_lut is not None:
            return plane_lut
        if "planes_flat" in d:
            return _channel_to_plane_lut(d["channels_flat"], d["planes_flat"], n_channels)
        return None

    chan_lo, chan_hi = channel_range if channel_range else (0, n_channels)
    window_stride = window_stride or window_size
    window_variables = list(window_variables) if window_variables else variables

    n_written = 0
    with uproot.recreate(output) as f:

        # ---------------- raw per-hit ----------------
        if do_raw_hit:
            ranges = {var: _hist_range([a.get(var), b.get(var)], range_mode, *range_percentiles)
                      for var in variables}
            logger.info("Raw per-hit histogram ranges: %s",
                        {k: [round(v[0], 4), round(v[1], 4)] for k, v in ranges.items()})

            if do_all_detector:
                for var in variables:
                    for d, label in ((a, label_a), (b, label_b)):
                        if var not in d:
                            continue
                        n_written += _write_all_detector(
                            f, f"all_detector/{var}_{label}", d[var], bins, ranges[var])
                logger.info("all_detector (raw): done")

            if do_per_plane:
                for d, label in ((a, label_a), (b, label_b)):
                    planes = planes_for(d)
                    if planes is None:
                        logger.warning(
                            "No plane info for label=%s (no planes_flat in npz and no "
                            "--channel-map given) -- skipping per-plane (raw) for this file.", label)
                        continue
                    for var in variables:
                        if var not in d:
                            continue
                        n_written += _write_per_plane(
                            f, "per_plane", d["channels_flat"], d[var], planes, bins, ranges[var],
                            f"{var}_{label}")
                logger.info("per_plane (raw): done")

            if do_per_channel:
                for d, label in ((a, label_a), (b, label_b)):
                    for var in variables:
                        if var not in d:
                            continue
                        n_written += _write_per_channel(
                            f, "per_channel", d["channels_flat"], d[var],
                            (chan_lo, chan_hi), bins, ranges[var], f"{var}_{label}", progress_every, label)
                logger.info("per_channel (raw): done")

        # ---------------- windowed ----------------
        if windowed:
            logger.info(
                "Windowed mode: window_size=%d stride=%d stats=%s variables=%s"
                "%s", window_size, window_stride, list(window_stats), window_variables,
                f" max_windows={max_windows}" if max_windows else "")
            for var in window_variables:
                for stat in window_stats:
                    w_a_ch, w_a_val = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))
                    w_b_ch, w_b_val = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))
                    if var in a:
                        w_a_ch, w_a_val = _windowed_channel_values(
                            a["channels_flat"], a[var], a["offsets"], window_size, window_stride,
                            stat, max_windows, progress_every, f"{label_a}/{var}/{stat}")
                    if var in b:
                        w_b_ch, w_b_val = _windowed_channel_values(
                            b["channels_flat"], b[var], b["offsets"], window_size, window_stride,
                            stat, max_windows, progress_every, f"{label_b}/{var}/{stat}")

                    rng = _hist_range([w_a_val, w_b_val], range_mode, *range_percentiles)
                    base = f"windowed/{var}/{stat}"

                    if do_all_detector:
                        if w_a_val.size:
                            n_written += _write_all_detector(f, f"{base}/all_detector_{label_a}", w_a_val, bins, rng)
                        if w_b_val.size:
                            n_written += _write_all_detector(f, f"{base}/all_detector_{label_b}", w_b_val, bins, rng)

                    if do_per_plane:
                        a_plane_lut = channel_plane_lut_for(a) if w_a_val.size else None
                        b_plane_lut = channel_plane_lut_for(b) if w_b_val.size else None
                        if w_a_val.size and a_plane_lut is not None:
                            n_written += _write_per_plane(
                                f, f"{base}/per_plane", w_a_ch, w_a_val,
                                a_plane_lut[np.clip(w_a_ch, 0, n_channels - 1)], bins, rng, label_a)
                        if w_b_val.size and b_plane_lut is not None:
                            n_written += _write_per_plane(
                                f, f"{base}/per_plane", w_b_ch, w_b_val,
                                b_plane_lut[np.clip(w_b_ch, 0, n_channels - 1)], bins, rng, label_b)
                        if (w_a_val.size and a_plane_lut is None) or (w_b_val.size and b_plane_lut is None):
                            logger.warning(
                                "No plane info available for windowed/%s/%s -- pass --channel-map "
                                "or materialize with planes_flat.", var, stat)

                    if windowed_per_channel:
                        if w_a_val.size:
                            n_written += _write_per_channel(
                                f, f"{base}/per_channel", w_a_ch, w_a_val,
                                (chan_lo, chan_hi), bins, rng, label_a, progress_every, f"{label_a}/{var}/{stat}")
                        if w_b_val.size:
                            n_written += _write_per_channel(
                                f, f"{base}/per_channel", w_b_ch, w_b_val,
                                (chan_lo, chan_hi), bins, rng, label_b, progress_every, f"{label_b}/{var}/{stat}")
            logger.info("windowed: done")

    logger.info("Wrote %d histograms to %s", n_written, output)
    return n_written


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file-a", required=True, help="First events .npz (e.g. good runs)")
    ap.add_argument("--file-b", required=True, help="Second events .npz (e.g. bad runs)")
    ap.add_argument("--label-a", default="good")
    ap.add_argument("--label-b", default="bad")
    ap.add_argument("--output", required=True, help="Output ROOT file")
    ap.add_argument("--variables", nargs="+", default=None,
                     help=f"Subset of {list(_VARIABLES)} to compare (default: all present)")
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--range-mode", choices=["percentile", "minmax"], default="percentile")
    ap.add_argument("--range-percentiles", nargs=2, type=float, default=[0.1, 99.9])
    ap.add_argument("--channel-map", default=None,
                     help="CSV with offlchan/plane columns (e.g. "
                          "configs/SBNDTPCChannelMap_v2_with_positions.csv), used for "
                          "per-plane grouping if the npz files don't have planes_flat")
    ap.add_argument("--channel-range", nargs=2, type=int, default=None, metavar=("LO", "HI"),
                     help="Restrict per-channel histograms (raw and windowed) to channels in [LO, HI)")
    ap.add_argument("--no-raw-hit", action="store_true", help="Skip the raw per-hit section entirely")
    ap.add_argument("--no-per-channel", action="store_true", help="Skip the (large, slow) per-channel section")
    ap.add_argument("--no-per-plane", action="store_true")
    ap.add_argument("--no-all-detector", action="store_true")
    ap.add_argument("--progress-every", type=int, default=1000)

    ap.add_argument("--windowed", action="store_true",
                     help="Also form model-style windows and compare per-window "
                          "channel statistics (mean/median/stdev/etc), not just raw hits")
    ap.add_argument("--window-size", type=int, default=100,
                     help="Events per window -- match data.window_size in your training config")
    ap.add_argument("--window-stride", type=int, default=None,
                     help="Default: same as --window-size (non-overlapping windows)")
    ap.add_argument("--window-stats", nargs="+", default=["mean", "median", "stdev"],
                     choices=list(_WINDOW_STAT_FUNCS) + ["count"],
                     help="Per-(window, channel) reduction(s) to compare")
    ap.add_argument("--window-variables", nargs="+", default=None,
                     help="Subset of --variables to window-aggregate (default: same as --variables)")
    ap.add_argument("--windowed-per-channel", action="store_true",
                     help="Also write the per-channel windowed breakdown (large -- "
                          "combine with --channel-range). Off by default; "
                          "all_detector/per_plane windowed comparisons are always cheap.")
    ap.add_argument("--max-windows", type=int, default=None,
                     help="Cap the number of windows processed per file (fast first look on huge inputs)")
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    compare(
        file_a=args.file_a, file_b=args.file_b, output=args.output,
        label_a=args.label_a, label_b=args.label_b,
        variables=args.variables, bins=args.bins,
        range_mode=args.range_mode, range_percentiles=tuple(args.range_percentiles),
        channel_map=args.channel_map,
        channel_range=tuple(args.channel_range) if args.channel_range else None,
        do_raw_hit=not args.no_raw_hit,
        do_per_channel=not args.no_per_channel,
        do_per_plane=not args.no_per_plane,
        do_all_detector=not args.no_all_detector,
        progress_every=args.progress_every,
        windowed=args.windowed,
        window_size=args.window_size,
        window_stride=args.window_stride,
        window_stats=args.window_stats,
        window_variables=args.window_variables,
        windowed_per_channel=args.windowed_per_channel,
        max_windows=args.max_windows,
    )


if __name__ == "__main__":
    main()
