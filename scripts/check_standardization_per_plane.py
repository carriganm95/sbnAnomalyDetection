#!/usr/bin/env python3
"""Check whether graph_vae's pooled feature standardization hides a per-plane offset.

`SparseWindowDatasetPyG._fit_standardization()` (used when `data.standardize:
true`) fits ONE mean/std per feature column by pooling every active channel
from every plane together (see sparse_window_dataset.py). If a feature's true
scale differs by plane (plausible: collection is unipolar, induction is
bipolar), every "normal" window on the underrepresented plane gets z-scored to
a nonzero offset just from being that plane -- not from being anomalous. This
script re-fits the same statistic split by plane and reports how far each
plane's own mean/std sits from the pooled one, in units of the pooled std
(i.e. the systematic offset a channel on that plane would see purely from
using the shared, pooled normalization instead of its own plane's).

This directly reuses SparseWindowDatasetPyG's own `_compute_frame` /
`_fit_standardization` (same RNG seed/frame sampling), so it requires the
same environment as training (torch + torch_geometric), unlike
compare_events_distributions.py. Run it in the real venv, not a bare
numpy/pandas environment.

Usage
-----
Match your training config directly (recommended -- reads data.window_size,
data.n_temporal_bins, data.stride, data.node_features, data.channel_map from
the yaml, same as sbn-train would use):

    python scripts/check_standardization_per_plane.py \\
        --events data/good_events_train.npz --config configs/graph_vae.yaml

Override specific fields without editing the yaml, and dump a full CSV:

    python scripts/check_standardization_per_plane.py \\
        --events data/good_events_train.npz --config configs/graph_vae.yaml \\
        --max-frames 1000 --output-csv standardization_per_plane.csv

Plane comes from `planes_flat` in the events npz if present; otherwise pass
--channel-map to derive it from channel id.
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

# Pure-numpy helpers shared with the good/bad comparison script (no torch needed
# for these two functions specifically).
from compare_events_distributions import _channel_plane_lookup, _channel_to_plane_lut  # noqa: E402

logger = logging.getLogger("check_standardization_per_plane")


def _load_config_defaults(config_path: Optional[str]) -> dict:
    if not config_path:
        return {}
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg.get("data", {})
    return {
        "window_size": data_cfg.get("window_size"),
        "n_bins": data_cfg.get("n_temporal_bins"),
        "stride": data_cfg.get("stride"),
        "node_features": data_cfg.get("node_features"),
        "log_features": data_cfg.get("log_features"),
        "channel_map": data_cfg.get("channel_map"),
        "n_channels": data_cfg.get("n_channels"),
    }


def _sample_frame_rows(ds, max_frames: int, seed: int):
    """Replicates SparseWindowDatasetPyG._fit_standardization's own sampling
    exactly (same seed/rng.choice call), so results here are directly
    comparable to -- and cross-checked against -- the model's real fit.

    Returns a list of (feature_rows, active_channel_indices) per sampled frame.
    """
    starts = ds._starts
    if len(starts) > max_frames:
        rng = np.random.default_rng(seed)
        starts = [starts[i] for i in rng.choice(len(starts), max_frames, replace=False)]
    out = []
    for s in starts:
        frame = ds._compute_frame(s, s + ds.window_size).reshape(ds.num_nodes, -1)
        act = np.abs(frame).sum(axis=1) > 1e-6
        if act.any():
            out.append((frame[act], np.where(act)[0]))
    return out


def _column_labels(node_features, n_bins) -> list:
    return [f"bin{b}_{name}" for b in range(n_bins) for name in node_features]


def run(
    events_path: str,
    window_size: int,
    n_bins: int,
    stride: int,
    node_features,
    channel_map: Optional[str],
    n_channels: Optional[int],
    max_frames: int,
    seed: int,
    min_plane_samples: int,
    flag_threshold: float,
    output_csv: Optional[str],
) -> None:
    from sbn_anomaly.data.sparse_window_dataset import SparseWindowDatasetPyG

    logger.info("Loading %s ...", events_path)
    ds = SparseWindowDatasetPyG.from_npz(
        events_path,
        history=1,  # unused for reconstruction mode
        window_size=window_size,
        n_bins=n_bins,
        stride=stride,
        node_features=node_features,
        n_channels=n_channels,
        reconstruction=True,
        standardize=False,   # we fit it ourselves, pooled and per-plane, below
        prune_inactive=False,
        edge_mode="sequential",  # cheap; we don't need the real graph for this check
    )
    logger.info(
        "Dataset: %d channels, %d node_features x %d bins = %d columns, %d candidate windows",
        ds.num_nodes, ds.n_node_features, ds.n_bins, ds.node_feat_dim, len(ds._starts))

    # Plane lookup: prefer planes_flat already in the npz, else --channel-map.
    if ds._planes_flat is not None:
        channel_to_plane = _channel_to_plane_lut(ds._channels_flat, ds._planes_flat, ds.num_nodes)
    elif channel_map:
        channel_to_plane = _channel_plane_lookup(channel_map, ds.num_nodes)
    else:
        raise ValueError(
            "No plane info: events npz has no 'planes_flat' and no --channel-map given. "
            "Re-materialize with planes_flat, or pass --channel-map.")

    logger.info("Sampling up to %d frames (seed=%d) ...", max_frames, seed)
    sampled = _sample_frame_rows(ds, max_frames, seed)
    if not sampled:
        raise ValueError("No active frames sampled -- check window_size/stride against the events npz.")

    pooled_rows = [rows for rows, _ in sampled]
    pooled_flat = np.concatenate(pooled_rows, axis=0)
    pooled_mean = pooled_flat.mean(axis=0)
    pooled_std = pooled_flat.std(axis=0)

    # Cross-check against the model's own method -- should match exactly,
    # since we replicate its seed/sampling. If this warns, something about
    # ds._starts or the RNG call changed upstream; treat results with caution.
    ref_mean, ref_std = ds._fit_standardization(max_frames=max_frames)
    if not (np.allclose(pooled_mean, ref_mean, atol=1e-5) and np.allclose(pooled_std, ref_std, atol=1e-5)):
        logger.warning(
            "Replicated pooled mean/std does not exactly match "
            "SparseWindowDatasetPyG._fit_standardization()'s own output -- "
            "double-check this script's sampling still matches that method.")
    else:
        logger.info("Pooled mean/std cross-checked OK against ds._fit_standardization().")

    plane_rows: dict = {}
    for rows, chan_idx in sampled:
        planes_here = channel_to_plane[chan_idx]
        for p in np.unique(planes_here):
            if p < 0:
                continue
            mask = planes_here == p
            plane_rows.setdefault(int(p), []).append(rows[mask])

    planes = sorted(plane_rows)
    if not planes:
        raise ValueError("No valid (>=0) plane values found among active channels.")

    labels = _column_labels(ds.node_features, ds.n_bins)

    plane_stats = {}
    for p in planes:
        flat = np.concatenate(plane_rows[p], axis=0)
        n_samples = flat.shape[0]
        if n_samples < min_plane_samples:
            logger.warning(
                "plane %d: only %d active-channel samples (< --min-plane-samples=%d) -- "
                "estimate may be noisy.", p, n_samples, min_plane_samples)
        plane_stats[p] = {
            "n_samples": n_samples,
            "mean": flat.mean(axis=0),
            "std": flat.std(axis=0),
        }

    # ---- report ----
    safe_pooled_std = np.where(pooled_std < 1e-8, 1.0, pooled_std)
    rows_report = []
    print(f"\n{'column':<22}{'pooled_mean':>13}{'pooled_std':>12}", end="")
    for p in planes:
        print(f"  plane{p}_offset(sigma)  plane{p}_std_ratio", end="")
    print()

    flagged = []
    for i, label in enumerate(labels):
        line = f"{label:<22}{pooled_mean[i]:>13.4g}{pooled_std[i]:>12.4g}"
        for p in planes:
            offset = (plane_stats[p]["mean"][i] - pooled_mean[i]) / safe_pooled_std[i]
            std_ratio = plane_stats[p]["std"][i] / safe_pooled_std[i]
            line += f"  {offset:>20.3f}  {std_ratio:>17.3f}"
            rows_report.append({
                "column": label, "plane": p,
                "pooled_mean": pooled_mean[i], "pooled_std": pooled_std[i],
                "plane_mean": plane_stats[p]["mean"][i], "plane_std": plane_stats[p]["std"][i],
                "plane_n_samples": plane_stats[p]["n_samples"],
                "offset_in_pooled_sigma": offset, "std_ratio": std_ratio,
            })
            if abs(offset) >= flag_threshold:
                flagged.append((label, p, offset, std_ratio))
        print(line)

    plane_sample_summary = ", ".join(
        f"plane {p}: {plane_stats[p]['n_samples']} samples" for p in planes)
    print(f"\n{len(planes)} plane(s) found: {planes}  ({plane_sample_summary})")

    if flagged:
        print(f"\nColumns with |offset| >= {flag_threshold} pooled-sigma "
              f"(pooled standardization likely biasing this plane/feature):")
        for label, p, offset, std_ratio in sorted(flagged, key=lambda r: -abs(r[2])):
            print(f"  {label:<22} plane {p}: offset={offset:+.3f} sigma, std_ratio={std_ratio:.3f}")
    else:
        print(f"\nNo column exceeded |offset| >= {flag_threshold} pooled-sigma -- "
              f"pooled standardization doesn't look plane-biased for this data/config.")

    if output_csv:
        import pandas as pd
        pd.DataFrame(rows_report).to_csv(output_csv, index=False)
        logger.info("Wrote full per-column/per-plane table to %s", output_csv)


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", required=True, help="Events .npz to fit standardization from (e.g. good_events_train.npz)")
    ap.add_argument("--config", default=None,
                     help="YAML config (e.g. configs/graph_vae.yaml) to read "
                          "data.window_size/n_temporal_bins/stride/node_features/channel_map from")
    ap.add_argument("--window-size", type=int, default=None)
    ap.add_argument("--n-bins", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--node-features", nargs="+", default=None)
    ap.add_argument("--channel-map", default=None,
                     help="CSV with offlchan/plane columns, used if the events npz lacks planes_flat")
    ap.add_argument("--n-channels", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=500,
                     help="Must match training's default (500) to reproduce its exact fit")
    ap.add_argument("--seed", type=int, default=0,
                     help="Must be 0 to reproduce SparseWindowDatasetPyG._fit_standardization()'s sampling")
    ap.add_argument("--min-plane-samples", type=int, default=20,
                     help="Warn if a plane has fewer active-channel samples than this")
    ap.add_argument("--flag-threshold", type=float, default=0.3,
                     help="Flag columns whose per-plane offset exceeds this many pooled-std units")
    ap.add_argument("--output-csv", default=None, help="Optional path to dump the full per-column/per-plane table")
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    defaults = _load_config_defaults(args.config)
    window_size = args.window_size or defaults.get("window_size") or 100
    n_bins = args.n_bins or defaults.get("n_bins") or 4
    stride = args.stride or defaults.get("stride") or window_size
    node_features = args.node_features or defaults.get("node_features")
    channel_map = args.channel_map or defaults.get("channel_map")
    n_channels = args.n_channels or defaults.get("n_channels")

    if not node_features:
        raise SystemExit("No node_features given (via --node-features or --config data.node_features).")

    run(
        events_path=args.events,
        window_size=window_size, n_bins=n_bins, stride=stride,
        node_features=node_features, channel_map=channel_map, n_channels=n_channels,
        max_frames=args.max_frames, seed=args.seed,
        min_plane_samples=args.min_plane_samples, flag_threshold=args.flag_threshold,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
