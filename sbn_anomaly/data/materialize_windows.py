"""Create temporal-binned per-channel windows from ROOT hit branches.

This materializer aggregates hit integrals per channel across a rolling
window of events and splits the window into `n_bins` temporal bins. The
output is a NumPy array of shape (N_windows, N_channels, n_bins) and is
saved to an .npz archive along with provenance metadata.

Usage (simple):
    python -m sbn_anomaly.data.materialize_windows \
            --root-files /data/run1.root /data/run2.root \
            --output events.npz --window-size 20 --n-bins 4

Or provide a YAML config with `data.hit_branches`, `data.tree_name`,
`data.window_size`, `data.n_temporal_bins` and `data.adjacency_radius`.
"""

from __future__ import annotations

import argparse
import logging
from collections import deque
from pathlib import Path
from typing import Any, Iterable, List, Optional

import awkward as ak
import numpy as np

from sbn_anomaly.data.streaming import RootStreamer

logger = logging.getLogger(__name__)


def _group_hit_prefixes(hit_branches: Iterable[str]) -> List[str]:
    """Return unique hit-prefixes like 'hits0.h' from branch names.

    E.g. 'hits0.h.integral' and 'hits0.h.channel' -> prefix 'hits0.h'.
    """
    prefixes = set()
    for b in hit_branches:
        parts = str(b).rsplit('.', 1)
        if len(parts) == 2:
            prefixes.add(parts[0])
    return sorted(prefixes)


def _extract_tpc_branch_value(batch: Any, i: int, branch: str) -> Optional[float]:
    """Extract the first numeric value for a TPC metadata branch.

    This mirrors the working fallback logic in build_tpc_features_from_root.py:
    index the per-event element first, then try awkward flattening, then fall
    back to NumPy conversion for scalar-like values.
    """
    try:
        col = batch[branch]
    except Exception:
        return None

    try:
        element = col[i]
    except Exception:
        return None

    try:
        vals = ak.to_numpy(ak.flatten(element, axis=None))
    except Exception:
        try:
            vals = np.asarray(element)
        except Exception:
            return None

    vals = np.asarray(vals).ravel()
    if vals.size == 0:
        return None
    try:
        return float(vals[0])
    except Exception:
        return None


def materialize_windows_from_root(
    file_paths: Iterable[str],
    tree_name: str,
    hit_branches: List[str],
    window_size: int = 20,
    n_bins: int = 4,
    stride: int = 1,
    n_channels: Optional[int] = None,
    tpc_branches: Optional[List[str]] = None,
    node_features: Optional[List[str]] = None,
    max_events: Optional[int] = None,
    output_path: Optional[str] = None,
):
    """Materialize windows and return (windows, metadata).

    Parameters
    ----------
    node_features : list of str, optional
        Which per-channel statistics to compute per temporal bin.
        Default: ["sum", "min", "max", "stdev", "count"]
        Allowed: "sum", "min", "max", "stdev", "mean", "count"
    max_events : int, optional
        Stop after collecting this many events. Useful for quick testing.
        If None, process all events in the input files.

    Returns
    -------
    windows: np.ndarray  shape (N_windows, N_channels, n_bins, n_features)
    metadata: dict with keys: n_temporal_bins, window_size, stride, n_channels, 
              channel_map, input_filenames, run, subrun, evt, node_features
    """
    # Set default node features if not provided
    if node_features is None:
        node_features = ["sum", "min", "max", "stdev", "count"]
    node_features = [str(f) for f in node_features]
    allowed_features = {"sum", "min", "max", "stdev", "mean", "count"}
    invalid = set(node_features) - allowed_features
    if invalid:
        raise ValueError(f"Invalid node_features: {invalid}. Allowed: {allowed_features}")
    n_node_features = len(node_features)

    prefixes = _group_hit_prefixes(hit_branches)
    # Identify required branch names for each prefix
    integral_branches = [p + ".integral" for p in prefixes]
    channel_branches = [p + ".channel" for p in prefixes]

    # Build list of all branches to stream
    all_branches = list(set(integral_branches + channel_branches))
    if tpc_branches:
        all_branches.extend(tpc_branches)

    logger.info("Streaming from %d file(s), tree='%s', window_size=%d, n_bins=%d, stride=%d", 
                len(list(file_paths)) if isinstance(file_paths, list) else 1,
                tree_name, window_size, n_bins, stride)
    logger.info("Hit prefixes: %s, TPC branches: %s, node_features: %s", prefixes, tpc_branches or [], node_features)

    streamer = RootStreamer(
        file_paths=file_paths, tree_name=tree_name, branches=all_branches, batch_size=512
    )

    # First pass: collect all events with their metadata
    all_events: List[dict] = []  # Each event: {ch_values, metadata, filename}
    discovered_max_channel = -1
    total_events = 0
    batch_count = 0

    logger.info("Collecting all events from ROOT files...")
    for batch in streamer.stream():
        batch_count += 1
        n = len(batch)
        total_events += n
        logger.debug("Processing batch %d with %d events (total so far: %d)", batch_count, n, total_events)
        
        for i in range(n):
            # Check if we've reached max_events limit
            if max_events is not None and len(all_events) >= max_events:
                logger.info("Reached max_events limit of %d", max_events)
                break
            
            # Extract TPC metadata (run, subrun, event)
            event_meta = {}
            if tpc_branches:
                for b in tpc_branches:
                    val = _extract_tpc_branch_value(batch, i, b)
                    if val is not None:
                        event_meta[b] = val
            
            # build per-event hit arrays — keep as raw numpy to avoid Python loops
            ch_parts: List[np.ndarray] = []
            val_parts: List[np.ndarray] = []
            for p, integ_b, ch_b in zip(prefixes, integral_branches, channel_branches):
                try:
                    integrals = ak.to_numpy(ak.flatten(batch[integ_b][i], axis=None)).astype(np.float32)
                    channels = ak.to_numpy(ak.flatten(batch[ch_b][i], axis=None)).astype(np.int64)
                except Exception:
                    continue
                m = min(len(integrals), len(channels))
                if m == 0:
                    continue
                integrals = integrals[:m]
                channels = channels[:m]
                valid = channels >= 0
                ch_parts.append(channels[valid])
                val_parts.append(integrals[valid])

            if ch_parts:
                channels_all = np.concatenate(ch_parts)
                integrals_all = np.concatenate(val_parts)
                if channels_all.size:
                    discovered_max_channel = max(discovered_max_channel, int(channels_all.max()))
            else:
                channels_all = np.empty(0, dtype=np.int64)
                integrals_all = np.empty(0, dtype=np.float32)

            filename = str(streamer.file_paths[0]) if hasattr(streamer, 'file_paths') else "unknown"
            all_events.append({
                'channels': channels_all,
                'integrals': integrals_all,
                'metadata': event_meta,
                'filename': filename,
            })

        # Break outer loop if max_events reached
        if max_events is not None and len(all_events) >= max_events:
            logger.info("Reached max_events limit of %d", max_events)
            break

    logger.info("Collected %d total events. Sorting by (run, subrun, event)...", len(all_events))

    # Sort events by run, subrun, event
    def _event_sort_key(evt):
        meta = evt['metadata']
        # Find run, subrun, event values in the metadata
        run = subrun = event_num = None
        for b in tpc_branches or []:
            if b in meta:
                val = int(meta[b]) if isinstance(meta[b], float) else meta[b]
                if "run" in b.lower() and "subrun" not in b.lower():
                    run = val
                elif "subrun" in b.lower():
                    subrun = val
                elif "evt" in b.lower():
                    event_num = val
        # Use large defaults so unsorted events go to the end
        return (run or 999999999, subrun or 999999999, event_num or 999999999)

    all_events.sort(key=_event_sort_key)
    logger.info("Events sorted. Creating windows...")

    # Pre-compute output shape so we can allocate once — either on disk (memmap)
    # or in RAM — instead of accumulating a Python list and np.stack-ing at the end.
    N_channels_final = int(discovered_max_channel + 1) if n_channels is None else int(n_channels)
    if N_channels_final <= 0:
        N_channels_final = 0
    n_events_collected = len(all_events)
    n_windows_expected = max(0, (n_events_collected - window_size) // stride + 1) if n_events_collected >= window_size else 0
    windows_shape = (n_windows_expected, N_channels_final, n_bins, n_node_features)

    if output_path is not None:
        npy_path = str(output_path) if str(output_path).endswith(".npy") else str(output_path) + ".npy"
        logger.info(
            "Pre-allocating output memmap %s shape=%s (~%.1f GB)",
            npy_path, windows_shape,
            n_windows_expected * N_channels_final * n_bins * n_node_features * 4 / 1e9,
        )
        windows_arr = np.memmap(npy_path, dtype=np.float32, mode="w+", shape=windows_shape)
    else:
        logger.info(
            "Pre-allocating in-RAM windows array shape=%s (~%.1f GB)",
            windows_shape,
            n_windows_expected * N_channels_final * n_bins * n_node_features * 4 / 1e9,
        )
        windows_arr = np.zeros(windows_shape, dtype=np.float32)

    window_idx = 0

    # Second pass: form windows from sorted events
    # Use deques so popleft() is O(1) instead of list.pop(0) which is O(n)
    event_buffer: deque = deque()
    event_metadata_buffer: deque = deque()
    event_filename_buffer: deque = deque()
    window_metadata: List[dict] = []
    window_filenames: List[str] = []
    input_filenames_list = []
    window_runs: List[int] = []
    window_subruns: List[int] = []
    window_evts: List[int] = []

    # Precompute bin splits — same for every window
    bin_splits = np.array_split(np.arange(window_size), n_bins)

    for evt in all_events:
        event_buffer.append({'channels': evt['channels'], 'integrals': evt['integrals']})
        event_metadata_buffer.append(evt['metadata'])
        event_filename_buffer.append(evt['filename'])

        if evt['filename'] not in input_filenames_list:
            input_filenames_list.append(evt['filename'])

        if len(event_buffer) >= window_size:
            if n_channels is not None:
                N_channels = int(n_channels)
            else:
                N_channels = int(discovered_max_channel + 1) if discovered_max_channel >= 0 else 0

            window_arr = np.zeros((N_channels, n_bins, n_node_features), dtype=np.float32)

            if N_channels > 0:
                buf = list(event_buffer)  # snapshot for indexing
                for b_idx, split in enumerate(bin_splits):
                    # Concatenate all hits from events in this bin — one numpy op
                    ch_parts = [buf[idx]['channels'] for idx in split if buf[idx]['channels'].size]
                    val_parts = [buf[idx]['integrals'] for idx in split if buf[idx]['integrals'].size]
                    if not ch_parts:
                        continue
                    ch_arr = np.concatenate(ch_parts)
                    val_arr = np.concatenate(val_parts)

                    # Filter to valid channel range
                    mask = ch_arr < N_channels
                    ch_arr = ch_arr[mask]
                    val_arr = val_arr[mask]
                    if ch_arr.size == 0:
                        continue

                    # Sort by channel so reduceat groups contiguous runs
                    order = np.argsort(ch_arr, kind='stable')
                    ch_sorted = ch_arr[order]
                    val_sorted = val_arr[order]

                    unique_chs, starts, counts = np.unique(ch_sorted, return_index=True, return_counts=True)

                    for feat_idx, feat_name in enumerate(node_features):
                        if feat_name == "count":
                            window_arr[unique_chs, b_idx, feat_idx] = counts
                        elif feat_name == "sum":
                            window_arr[unique_chs, b_idx, feat_idx] = np.add.reduceat(val_sorted, starts)
                        elif feat_name == "min":
                            window_arr[unique_chs, b_idx, feat_idx] = np.minimum.reduceat(val_sorted, starts)
                        elif feat_name == "max":
                            window_arr[unique_chs, b_idx, feat_idx] = np.maximum.reduceat(val_sorted, starts)
                        elif feat_name == "mean":
                            sums = np.add.reduceat(val_sorted, starts)
                            window_arr[unique_chs, b_idx, feat_idx] = sums / counts
                        elif feat_name == "stdev":
                            sums = np.add.reduceat(val_sorted, starts)
                            means_exp = np.repeat(sums / counts, counts)
                            sq = np.add.reduceat((val_sorted - means_exp) ** 2, starts)
                            window_arr[unique_chs, b_idx, feat_idx] = np.sqrt(sq / counts)

            # Write directly into pre-allocated array — no list accumulation
            if window_idx < n_windows_expected:
                windows_arr[window_idx] = window_arr
            window_idx += 1

            # Collect metadata for this window (from the first event in the window)
            if event_metadata_buffer:
                window_meta = event_metadata_buffer[0].copy()
                window_metadata.append(window_meta)
                run_val = subrun_val = evt_val = None
                for b in tpc_branches or []:
                    if b in window_meta:
                        if "run" in b.lower() and "subrun" not in b.lower():
                            run_val = int(window_meta[b])
                        elif "subrun" in b.lower():
                            subrun_val = int(window_meta[b])
                        elif "evt" in b.lower():
                            evt_val = int(window_meta[b])
                if run_val is not None:
                    window_runs.append(run_val)
                if subrun_val is not None:
                    window_subruns.append(subrun_val)
                if evt_val is not None:
                    window_evts.append(evt_val)

            if event_filename_buffer:
                window_filenames.append(event_filename_buffer[0])

            if window_idx % 100 == 0:
                logger.info("Created %d windows (total events processed: %d)", window_idx, len(all_events))

            # Advance by stride — deque.popleft() is O(1) vs list.pop(0) O(n)
            for _ in range(stride):
                if event_buffer:
                    event_buffer.popleft()
                if event_metadata_buffer:
                    event_metadata_buffer.popleft()
                if event_filename_buffer:
                    event_filename_buffer.popleft()

    logger.info("Finished windowing: %d total events, %d complete windows created", len(all_events), window_idx)

    # Trim to actual count in case our pre-computed estimate was off
    windows_arr = windows_arr[:window_idx]

    # Build channel_map: physical_channel -> index
    channel_map = {i: i for i in range(N_channels_final)}

    # Build metadata dictionary
    # Windows shape is now (N_windows, N_channels, n_temporal_bins, n_node_features)
    meta = {
        "n_temporal_bins": int(n_bins),
        "window_size": int(window_size),
        "stride": int(stride),
        "n_channels": int(N_channels_final),
        "channel_map": np.array([channel_map.get(i, i) for i in range(N_channels_final)], dtype=np.int64),
        "hit_branches": np.array(hit_branches, dtype=str),
        "input_filenames": np.array(window_filenames, dtype=str) if window_filenames else np.array([], dtype=str),
        "node_features": np.array(node_features, dtype=str),
    }

    # If TPC metadata branches were requested, save their names and values per-window.
    if tpc_branches:
        tpc_names = list(tpc_branches)
        # build array shape (N_windows, n_tpc_branches) filled with NaN for missing values
        n_w = windows_arr.shape[0]
        n_tb = len(tpc_names)
        tpc_vals = np.full((n_w, n_tb), np.nan, dtype=np.float64)
        for wi, wmeta in enumerate(window_metadata):
            for j, b in enumerate(tpc_names):
                if b in wmeta:
                    try:
                        tpc_vals[wi, j] = float(wmeta[b])
                    except Exception:
                        tpc_vals[wi, j] = np.nan

        meta["tpc_branches"] = np.array(tpc_names, dtype=str)
        meta["tpc_branch_values"] = tpc_vals

        # Also populate convenience arrays `run`, `subrun`, `evt` when branch names exist
        # Find indices for common names
        def _find_idx(key):
            for idx, name in enumerate(tpc_names):
                if key in name.lower():
                    return idx
            return None

        run_idx = _find_idx("run")
        subrun_idx = _find_idx("subrun")
        evt_idx = _find_idx("evt")

        # Safely convert branch float values (with NaN for missing) to int arrays.
        # Converting NaN directly to int can produce sentinel values like
        # -9223372036854775808; instead fill missing entries with -1.
        if run_idx is not None:
            run_f = tpc_vals[:, run_idx]
            run_int = np.full(run_f.shape, -1, dtype=np.int64)
            mask = np.isfinite(run_f)
            if mask.any():
                run_int[mask] = run_f[mask].astype(np.int64)
            meta["run"] = run_int
        if subrun_idx is not None:
            subrun_f = tpc_vals[:, subrun_idx]
            subrun_int = np.full(subrun_f.shape, -1, dtype=np.int64)
            mask = np.isfinite(subrun_f)
            if mask.any():
                subrun_int[mask] = subrun_f[mask].astype(np.int64)
            meta["subrun"] = subrun_int
        if evt_idx is not None:
            evt_f = tpc_vals[:, evt_idx]
            evt_int = np.full(evt_f.shape, -1, dtype=np.int64)
            mask = np.isfinite(evt_f)
            if mask.any():
                evt_int[mask] = evt_f[mask].astype(np.int64)
            meta["evt"] = evt_int

    return windows_arr, meta


def _main_cli(argv=None):
    parser = argparse.ArgumentParser(description="Materialize temporal-binned windows from ROOT hit branches")
    parser.add_argument("--config", type=str, default=None, help="YAML config file with data.hit_branches, data.window_size, data.n_temporal_bins, data.stride, data.tree_name, data.node_features")
    parser.add_argument("--root-files", nargs="+", default=None)
    parser.add_argument("--root-file-list", nargs="+", default=None, metavar="FILE", help="One or more text files containing ROOT paths, one per line")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tree-name", default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--n-bins", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--n-channels", type=int, default=None)
    parser.add_argument("--hit-branches", nargs="+", default=None, help="List of hit branch names; overrides config")
    parser.add_argument("--node-features", nargs="+", default=None, help="Node features to compute: sum, min, max, stdev, mean, count (default: all)")
    parser.add_argument("--max-events", type=int, default=None, help="Stop after collecting this many events (useful for testing)")
    parser.add_argument(
        "--windows",
        action="store_true",
        default=False,
        help=(
            "Save dense windows as .npy + _meta.npz instead of the default sparse events .npz. "
            "This is mainly for legacy workflows that expect prebuilt window arrays."
        ),
    )
    args = parser.parse_args(argv)

    # Load defaults from YAML config if provided
    tree_name = "sbn_tree"
    window_size = 20
    n_bins = 4
    stride = 1
    hit_branches = None
    tpc_branches = None
    n_channels = None
    node_features = None
    max_events = None

    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        data_cfg = cfg.get("data", {})
        tree_name = data_cfg.get("tree_name", tree_name)
        window_size = data_cfg.get("window_size", window_size)
        n_bins = data_cfg.get("n_temporal_bins", n_bins)
        stride = data_cfg.get("stride", stride)
        hit_branches = data_cfg.get("hit_branches")
        tpc_branches = data_cfg.get("tpc_branches")
        n_channels = data_cfg.get("n_channels")
        node_features = data_cfg.get("node_features")
        max_events = data_cfg.get("max_events")
        logger.info("Loaded config from %s", args.config)

    # CLI arguments override config
    if args.tree_name is not None:
        tree_name = args.tree_name
    if args.window_size is not None:
        window_size = args.window_size
    if args.n_bins is not None:
        n_bins = args.n_bins
    if args.stride is not None:
        stride = args.stride
    if args.hit_branches is not None:
        hit_branches = args.hit_branches
    if args.n_channels is not None:
        n_channels = args.n_channels
    if args.node_features is not None:
        node_features = args.node_features
    if args.max_events is not None:
        max_events = args.max_events

    if not hit_branches:
        parser.error("--hit-branches required or must be in config under data.hit_branches")

    # Resolve root files from --root-files and/or --root-file-list
    root_files = []
    if args.root_files:
        root_files.extend(args.root_files)
    if args.root_file_list:
        from sbn_anomaly.data.root_files import resolve_root_files
        root_files.extend(resolve_root_files(args.root_file_list))
    
    if not root_files:
        parser.error("Provide --root-files and/or --root-file-list")

    logging.basicConfig(level=logging.INFO)

    out_path = Path(args.output)

    if not args.windows:
        # --- Compact sparse events path (default) ---
        # Stores raw (channels, integrals) per event — small enough to reuse
        # across training runs. Windows are formed on the fly by
        # SparseWindowDatasetPyG.
        from sbn_anomaly.data.sparse_window_dataset import SparseWindowDatasetPyG
        dataset = SparseWindowDatasetPyG.from_root(
            root_files=root_files,
            tree_name=tree_name,
            hit_branches=list(hit_branches),
            n_channels=n_channels,
            sort_events=True,
            tpc_branches=tpc_branches,
            max_events=max_events,
            # dataset_kwargs not needed for saving — use defaults
            history=1,        # dummy, not used when just saving
            window_size=window_size,
            n_bins=n_bins,
            node_features=node_features,
        )
        events_path = out_path if str(out_path).endswith(".npz") else out_path.with_suffix(".npz")
        dataset.save_events(str(events_path))
        logger.info(
            "Saved compact events: %d events, %d channels, ~%.1f MB  ->  %s",
            len(dataset._offsets) - 1, dataset.num_nodes,
            (dataset._channels_flat.nbytes + dataset._integrals_flat.nbytes) / 1e6,
            events_path,
        )
        return

    # Derive output paths for dense windows: windows go to a .npy file
    # (memory-mapped, no RAM spike), metadata goes to a companion _meta.npz.
    npy_path = out_path.with_suffix(".npy") if out_path.suffix != ".npy" else out_path
    meta_path = npy_path.with_name(npy_path.stem + "_meta.npz")

    windows, meta = materialize_windows_from_root(
        file_paths=root_files,
        tree_name=tree_name,
        hit_branches=list(hit_branches),
        window_size=window_size,
        n_bins=n_bins,
        stride=stride,
        n_channels=n_channels,
        tpc_branches=tpc_branches,
        node_features=node_features,
        max_events=max_events,
        output_path=str(npy_path),
    )

    # windows is already a memmap on disk at npy_path; flush and save metadata separately.
    if isinstance(windows, np.memmap):
        windows.flush()
    else:
        np.save(npy_path, windows)

    np.savez_compressed(meta_path, **meta)
    logger.info("Wrote %d windows (shape=%s) to %s", windows.shape[0], windows.shape, npy_path)
    logger.info("Wrote metadata to %s", meta_path)
    logger.info("  n_temporal_bins: %d, window_size: %d, stride: %d, n_channels: %d",
                meta['n_temporal_bins'], meta['window_size'], meta['stride'], meta['n_channels'])


if __name__ == "__main__":
    _main_cli()
