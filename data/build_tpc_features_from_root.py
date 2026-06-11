#!/usr/bin/env python3
"""Build a precomputed TPC feature matrix (.npz) from ROOT files.

This script mirrors the feature extraction behavior used by streaming training
so users can materialize features once and train from a map-style dataset.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import yaml
import uproot

from sbn_anomaly.data.root_files import resolve_root_files
from sbn_anomaly.data.stream_dataset import extract_hit_features, extract_tpc_features
from sbn_anomaly.data.streaming import RootStreamer
from sbn_anomaly.utils.logging import setup_logging


def _build_branches(
    data_cfg: dict[str, Any],
    waveform_branch: str | None,
    hit_branches: list[str],
) -> list[str]:
    branches = data_cfg.get("tpc_branches")
    if branches is None:
        return [waveform_branch] if waveform_branch is not None else list(hit_branches)

    merged: list[str] = [str(branch) for branch in branches]
    seen = {branch for branch in merged}
    for branch in hit_branches:
        if branch not in seen:
            merged.append(branch)
            seen.add(branch)
    return merged


def _filter_readable_root_files(
    root_files: list[str],
    tree_name: str,
) -> list[str]:
    """Return only ROOT files that can be opened and contain *tree_name*.

    This skips files that are unreadable through uproot/xrootd so a single bad
    file does not stop batch materialization.
    """

    readable: list[str] = []
    skipped = 0
    for path in root_files:
        try:
            with uproot.open(str(path)) as root_file:
                if tree_name not in root_file:
                    logger = logging.getLogger(__name__)
                    logger.warning("Skipping %s: tree '%s' not found", path, tree_name)
                    skipped += 1
                    continue
        except (OSError, FileNotFoundError) as exc:
            logger = logging.getLogger(__name__)
            logger.warning("Skipping unreadable ROOT file %s: %s", path, exc)
            skipped += 1
            continue
        readable.append(path)

    if skipped > 0:
        logging.getLogger(__name__).info(
            "Skipped %d unreadable/missing ROOT file(s)", skipped
        )
    return readable


def _extract_one_feature(
    batch: Any,
    i: int,
    waveform_branch: str | None,
    hit_branches: list[str],
    input_dim: int,
    max_hits: int | None = None,
) -> np.ndarray:
    if waveform_branch is not None:
        raw = ak.to_numpy(ak.flatten(batch[waveform_branch][i], axis=None)).astype(np.float32)
        return raw.astype(np.float32)

    event_data = {
        branch: ak.to_numpy(ak.flatten(batch[branch][i], axis=None)).astype(np.float32)
        for branch in hit_branches
    }

    # Optional whole-hit mode: keep exactly N hits worth of branch values,
    # preserving hit boundaries (no mid-hit truncation).
    if max_hits is not None:
        n_branches = len(hit_branches)
        target_dim = max_hits * n_branches
        feat = np.zeros(target_dim, dtype=np.float32)

        n_available_hits = 0
        for branch in hit_branches:
            n_available_hits = max(n_available_hits, int(event_data[branch].shape[0]))
        n_use_hits = min(max_hits, n_available_hits)

        for hit_idx in range(n_use_hits):
            base = hit_idx * n_branches
            for branch_idx, branch in enumerate(hit_branches):
                values = event_data[branch]
                if hit_idx < values.shape[0]:
                    feat[base + branch_idx] = float(values[hit_idx])
        return feat
    # Default: return the full interleaved hit vector (no truncation here).
    raw_parts: list[float] = []
    n_hits = 0
    for b in hit_branches:
        if b in event_data:
            n_hits = max(n_hits, int(event_data[b].shape[0]))
    for hit_idx in range(n_hits):
        for branch in hit_branches:
            if branch in event_data:
                values = event_data[branch]
                if hit_idx < values.shape[0]:
                    raw_parts.append(float(values[hit_idx]))
                else:
                    raw_parts.append(0.0)
    return np.asarray(raw_parts, dtype=np.float32)


def _extract_tpc_branch_values(
    batch: Any,
    i: int,
    tpc_branches: list[str],
) -> np.ndarray:
    """Extract raw values from tpc_branches for a single event.
    
    Returns a 1D array concatenating all branch values (in order of branches).
    """
    if not tpc_branches:
        return np.array([], dtype=np.float32)
    
    raw_parts: list[float] = []
    for branch in tpc_branches:
        # Robustly handle several possible upstream shapes:
        # - batch[branch] may be a 1-D array of scalars
        # - batch[branch] may be a nested awkward array (per-hit lists)
        # - batch may contain a record field; accessing the dotted name
        #   may yield a record/element that needs special handling
        try:
            col = batch[branch]
        except Exception:
            # Branch not present in this chunk
            continue

        try:
            element = col[i]
        except Exception:
            # Unable to index this column for event i
            continue

        # Try to flatten awkward structures; fall back to treating as scalar
        try:
            vals = ak.to_numpy(ak.flatten(element, axis=None))
        except Exception:
            try:
                # element may be a simple scalar or numpy-like
                vals = np.asarray(element)
            except Exception:
                # Give up on this branch for this event
                continue

        vals = np.asarray(vals).ravel()
        if vals.size:
            raw_parts.extend([float(x) for x in vals])
    
    return np.asarray(raw_parts, dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize TPC features from ROOT files to a .npz archive "
            "using the extraction logic in sbn_anomaly.data.stream_dataset."
        )
    )
    parser.add_argument("--config", required=True, help="Path to TPC YAML config.")
    parser.add_argument(
        "--root-files",
        nargs="+",
        default=None,
        metavar="PATH",
        help="ROOT files or glob patterns.",
    )
    parser.add_argument(
        "--root-file-list",
        nargs="+",
        default=None,
        metavar="FILE",
        help="Text file(s) containing ROOT paths, one per line.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output .npz path. Defaults to data.features_path from config "
            "(converted to .npz if needed)."
        ),
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional cap on number of events to materialize.",
    )
    parser.add_argument(
        "--batch-size-stream",
        type=int,
        default=None,
        help="Override data.batch_size_stream for ROOT streaming.",
    )
    parser.add_argument(
        "--max-hits",
        type=int,
        default=None,
        help=(
            "Hit-mode only. Preserve whole-hit blocks by extracting up to this many "
            "hits per event. Output feature length becomes max_hits * n_hit_branches."
        ),
    )
    parser.add_argument(
        "--start-file-index",
        type=int,
        default=0,
        help="Start processing from this file index (0-based). Use for batch processing.",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=None,
        help="Process up to this many files starting from --start-file-index. Default: all.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    root_inputs: list[str] = []
    if args.root_files:
        root_inputs.extend(args.root_files)
    if args.root_file_list:
        root_inputs.extend(args.root_file_list)
    if not root_inputs:
        raise SystemExit("Provide --root-files or --root-file-list.")

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"Config file not found: {cfg_path}")

    with cfg_path.open() as fh:
        cfg = yaml.safe_load(fh)

    if str(cfg.get("model_type", "")).lower() != "tpc":
        raise SystemExit("This script currently supports only model_type='tpc'.")

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    input_dim = int(model_cfg.get("input_dim", 256))
    tree_name = data_cfg.get("tree_name", "sbn_tree")
    waveform_branch = data_cfg.get("waveform_branch", "tpc_waveform")
    hit_branches = [str(branch) for branch in (data_cfg.get("hit_branches") or [])]
    if waveform_branch is None and not hit_branches:
        raise SystemExit("Hit mode requires data.hit_branches in the config.")
    if args.max_hits is not None and args.max_hits <= 0:
        raise SystemExit("--max-hits must be a positive integer.")
    if args.max_hits is not None and waveform_branch is not None:
        raise SystemExit("--max-hits is only supported in hit mode (set waveform_branch to null).")

    max_events = args.max_events
    if max_events is None:
        max_events = train_cfg.get("max_events")

    # We do not enforce model.input_dim or apply normalization when materializing
    # features. Instead we preserve whole-hit vectors (or pad to --max-hits when
    # requested) and save raw features. Normalization and truncation are applied
    # by dataset loaders during training/inference.

    output_path = Path(args.output) if args.output else Path(str(data_cfg.get("features_path", "")))
    if not output_path:
        raise SystemExit("No output path provided and data.features_path missing in config.")
    if output_path.suffix.lower() != ".npz":
        output_path = output_path.with_suffix(".npz")
    if output_path.exists() and not args.allow_overwrite:
        raise SystemExit(f"Output already exists: {output_path} (use --allow-overwrite)")

    root_files = resolve_root_files(root_inputs)
    if not root_files:
        raise SystemExit("No ROOT files were resolved from the given inputs.")

    # Handle batch file processing via start index and num files
    start_idx = int(args.start_file_index)
    if start_idx >= len(root_files):
        raise SystemExit(
            f"--start-file-index {start_idx} is >= total files ({len(root_files)})"
        )
    end_idx = len(root_files)
    if args.num_files is not None:
        end_idx = min(start_idx + int(args.num_files), len(root_files))
    root_files = root_files[start_idx:end_idx]
    if not root_files:
        raise SystemExit(f"No files to process after applying file index range.")
    logger.info("Processing files [%d:%d]", start_idx, end_idx)

    root_files = _filter_readable_root_files(root_files, tree_name)
    if not root_files:
        raise SystemExit("No readable ROOT files remain after filtering.")

    branches = _build_branches(data_cfg, waveform_branch, hit_branches)
    tpc_branch_names = [str(b) for b in (data_cfg.get("tpc_branches") or [])]
    # Ensure tpc_branches are included in streaming
    all_branches = list(branches)
    for b in tpc_branch_names:
        if b not in all_branches:
            all_branches.append(b)
    
    stream_batch_size = (
        int(args.batch_size_stream)
        if args.batch_size_stream is not None
        else int(data_cfg.get("batch_size_stream", 512))
    )

    logger.info("Resolved %d ROOT file(s)", len(root_files))
    logger.info("Streaming branches: %s", all_branches)

    features: list[np.ndarray] = []
    tpc_branch_values: list[np.ndarray] = []
    input_filenames: list[str] = []
    n_events = 0
    n_non_finite = 0

    for root_file in root_files:
        streamer = RootStreamer(
            file_paths=[root_file],
            tree_name=tree_name,
            branches=all_branches,
            batch_size=stream_batch_size,
        )

        for batch in streamer.stream():
            for i in range(len(batch)):
                feat = _extract_one_feature(
                    batch,
                    i,
                    waveform_branch,
                    hit_branches,
                    input_dim,
                    max_hits=args.max_hits,
                )

                if not np.isfinite(feat).all():
                    n_non_finite += 1
                    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

                features.append(feat.astype(np.float32, copy=False))

                # Extract tpc_branch values for this event
                if tpc_branch_names:
                    tpc_vals = _extract_tpc_branch_values(batch, i, tpc_branch_names)
                    tpc_branch_values.append(tpc_vals)

                input_filenames.append(str(root_file))

                n_events += 1

                if n_events % 10 == 0:
                    logger.debug("Processed %d events so far", n_events)

                if max_events is not None and n_events >= int(max_events):
                    break
            if max_events is not None and n_events >= int(max_events):
                break

        if max_events is not None and n_events >= int(max_events):
            break

    if not features:
        raise SystemExit("No events were extracted; output would be empty.")

    # Pad all event vectors to the maximum observed length so we can save a
    # regular 2-D array. This preserves all extracted information without
    # truncating to model.input_dim.
    max_len = max(int(f.shape[0]) for f in features)
    padded = np.zeros((len(features), max_len), dtype=np.float32)
    for idx, f in enumerate(features):
        padded[idx, : f.shape[0]] = f
    x = padded
    
    # Pad tpc_branch_values the same way
    tpc_vals_array = None
    if tpc_branch_values:
        max_tpc_len = max(int(v.shape[0]) for v in tpc_branch_values) if tpc_branch_values else 0
        if max_tpc_len > 0:
            tpc_vals_array = np.zeros((len(tpc_branch_values), max_tpc_len), dtype=np.float32)
            for idx, v in enumerate(tpc_branch_values):
                tpc_vals_array[idx, : v.shape[0]] = v
    
    feature_branch_names = ([str(waveform_branch)] if waveform_branch is not None else list(hit_branches))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    savez_kwargs = {
        "features": x,
        "feature_branch_names": np.asarray(feature_branch_names, dtype=str),
        "streamer_branches": np.asarray(branches, dtype=str),
        "tpc_branch_names": np.asarray(tpc_branch_names, dtype=str),
        "input_filenames": np.asarray(input_filenames, dtype=str),
        "feature_length": np.asarray([x.shape[1]], dtype=np.int64),
        "max_hits": np.asarray([args.max_hits if args.max_hits is not None else -1], dtype=np.int64),
    }
    
    # Only save tpc_branch_values if they exist
    if tpc_vals_array is not None:
        savez_kwargs["tpc_branch_values"] = tpc_vals_array
    
    np.savez(output_path, **savez_kwargs)

    logger.info("Saved features to %s", output_path)
    logger.info("Shape=%s dtype=%s", x.shape, x.dtype)
    if n_non_finite > 0:
        logger.warning("Replaced non-finite values in %d event(s)", n_non_finite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
