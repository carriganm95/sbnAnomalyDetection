"""Inference CLI entry-point.

Usage::

    sbn-infer --config configs/tpc.yaml --input /data/run.root --output scores.npy
    sbn-infer --config configs/tpc.yaml --root-files /data/run1.root /data/run2.root
    python -m sbn_anomaly.infer.cli --config configs/fusion.yaml \\
        --tpc-input tpc_features.npy --pmt-input pmt_features.npy
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

from sbn_anomaly.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run inference."""
    parser = argparse.ArgumentParser(description="Run SBN anomaly detection inference.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to .npy feature file (tpc/pmt/window modes).",
    )
    parser.add_argument(
        "--root-files",
        nargs="+",
        default=None,
        metavar="PATH",
        help=(
            "One or more ROOT file paths (glob patterns accepted). "
            "Supported for model_type=tpc: streams directly from ROOT and "
            "scores events without requiring an intermediate .npy file."
        ),
    )
    parser.add_argument(
        "--root-file-list",
        nargs="+",
        default=None,
        metavar="FILE",
        help=(
            "One or more text files containing ROOT paths, one per line. "
            "Lines may be comments starting with # and may also contain glob "
            "patterns."
        ),
    )
    parser.add_argument(
        "--tpc-input",
        type=str,
        default=None,
        help="Path to TPC .npy feature file (fusion mode).",
    )
    parser.add_argument(
        "--pmt-input",
        type=str,
        default=None,
        help="Path to PMT .npy feature file (fusion mode).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scores.npy",
        help="Path to save anomaly scores (.npy).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        return 1

    with config_path.open() as fh:
        cfg = yaml.safe_load(fh)

    model_type = cfg.get("model_type", "").lower()
    infer_cfg = cfg.get("inference", {})
    checkpoint = infer_cfg.get("checkpoint_path")
    if not checkpoint:
        logger.error("inference.checkpoint_path not set in config.")
        return 1

    # GNN uses a separate PyG-based inference path with per-node output
    if model_type == "gnn":
        _infer_gnn(cfg, checkpoint, args.output)
        return 0

    scorer = _build_scorer(cfg, model_type, checkpoint)

    root_files: list[str] | None = None
    if args.root_files or args.root_file_list:
        from sbn_anomaly.data.root_files import resolve_root_files

        root_inputs = []
        if args.root_files:
            root_inputs.extend(args.root_files)
        if args.root_file_list:
            root_inputs.extend(args.root_file_list)
        root_files = resolve_root_files(root_inputs)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    if model_type == "fusion":
        tpc_path = args.tpc_input or infer_cfg.get("tpc_input_path")
        pmt_path = args.pmt_input or infer_cfg.get("pmt_input_path")
        if not tpc_path or not pmt_path:
            logger.error("Fusion mode requires --tpc-input and --pmt-input.")
            return 1
        tpc_feat = np.load(tpc_path, allow_pickle=True)
        pmt_feat = np.load(pmt_path, allow_pickle=True)
        # Handle .npz archives
        if isinstance(tpc_feat, np.lib.npyio.NpzFile):
            tpc_feat = tpc_feat["features"]
        if isinstance(pmt_feat, np.lib.npyio.NpzFile):
            pmt_feat = pmt_feat["features"]
        # Truncate/pad to model input dims
        tpc_input_dim = model_cfg.get("tpc_input_dim", 256)
        pmt_input_dim = model_cfg.get("pmt_input_dim", 16)
        tpc_feat = _truncate_or_pad(tpc_feat, tpc_input_dim)
        pmt_feat = _truncate_or_pad(pmt_feat, pmt_input_dim)
        scores = scorer.score(tpc_feat, pmt_feat)
    else:
        if root_files:
            if model_type != "tpc":
                logger.error("--root-files is currently supported only for model_type='tpc'.")
                return 1
            scores = _score_tpc_from_root(cfg, scorer, root_files)
        else:
            input_path = args.input or infer_cfg.get("input_path")
            if not input_path:
                logger.error("Provide --input or set inference.input_path in config.")
                return 1
            input_archive = np.load(input_path, allow_pickle=True)
            # Preserve archive metadata when present
            if isinstance(input_archive, np.lib.npyio.NpzFile):
                features = input_archive["features"]
            else:
                features = input_archive
            # Truncate/pad to model input dim
            input_dim = model_cfg.get("input_dim", 256)
            features = _truncate_or_pad(features, input_dim)
            scores = scorer.score(features)

    # Prepare metadata for saving alongside scores so branches can be matched.
    meta: dict[str, object] = {}
    meta["event_index"] = np.arange(len(scores), dtype=np.int64)
    # Map-style input archives may contain branch metadata.
    if not root_files:
        # fusion handled separately above; for single-input mode we may have
        # preserved the loaded archive in `input_archive`.
        try:
            if isinstance(input_archive, np.lib.npyio.NpzFile):
                for key in ("feature_branch_names", "tpc_branches", "tpc_branch_values", "input_filenames", "streamer_branches", "feature_length", "max_hits"):
                    if key in input_archive:
                        meta[key] = input_archive[key]
        except Exception:
            pass
        # Always include configured hit_branches if present
        if data_cfg.get("hit_branches"):
            meta["hit_branches"] = np.asarray(data_cfg.get("hit_branches"), dtype=str)
    else:
        # For ROOT-streamed inputs include configured branches
        if data_cfg.get("tpc_branches"):
            meta["tpc_branches"] = np.asarray(data_cfg.get("tpc_branches"), dtype=str)
        if data_cfg.get("hit_branches"):
            meta["hit_branches"] = np.asarray(data_cfg.get("hit_branches"), dtype=str)

    out_path = Path(args.output)
    # If user requested an .npz file, save scores + metadata into it. Otherwise
    # save the numeric scores as .npy and write a companion metadata .npz.
    if out_path.suffix.lower() == ".npz":
        np.savez_compressed(out_path, scores=scores, **meta)
        logger.info("Saved %d anomaly scores and metadata to %s", len(scores), out_path)
    else:
        np.save(out_path, scores)
        meta_path = Path(str(out_path) + ".meta.npz")
        np.savez_compressed(meta_path, scores=scores, **meta)
        logger.info("Saved %d anomaly scores to %s and metadata to %s", len(scores), out_path, meta_path)
    return 0


def _truncate_or_pad(features: np.ndarray, target_dim: int) -> np.ndarray:
    """Truncate or pad feature array to target dimension."""
    if features.shape[1] == target_dim:
        return features
    adjusted = np.zeros((features.shape[0], target_dim), dtype=np.float32)
    copy_len = min(features.shape[1], target_dim)
    adjusted[:, :copy_len] = features[:, :copy_len]
    return adjusted


def _score_tpc_from_root(
    cfg: dict,
    scorer,
    root_files: list[str],
) -> np.ndarray:
    """Stream TPC events from ROOT, extract fixed-size features, score in batches."""
    import awkward as ak

    from sbn_anomaly.data.stream_dataset import extract_hit_features, extract_tpc_features
    from sbn_anomaly.data.streaming import RootStreamer

    logger = logging.getLogger(__name__)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    infer_cfg = cfg.get("inference", {})
    train_cfg = cfg.get("training", {})

    input_dim = int(model_cfg.get("input_dim", 256))
    waveform_branch = data_cfg.get("waveform_branch", "tpc_waveform")
    hit_branches = data_cfg.get("hit_branches") or []
    if waveform_branch is None and not hit_branches:
        raise ValueError(
            "For ROOT inference in hit mode (waveform_branch=null), set data.hit_branches."
        )

    branches = data_cfg.get("tpc_branches")
    if branches is None:
        if waveform_branch is not None:
            branches = [waveform_branch]
        else:
            branches = list(hit_branches)

    batch_size_stream = int(data_cfg.get("batch_size_stream", 512))
    max_events = infer_cfg.get("max_events", train_cfg.get("max_events"))
    max_events = None if max_events is None else int(max_events)

    logger.info("Scoring TPC stream from %d ROOT file(s)", len(root_files))
    streamer = RootStreamer(
        file_paths=root_files,
        tree_name=data_cfg.get("tree_name", "sbn_tree"),
        branches=branches,
        batch_size=batch_size_stream,
    )

    score_chunks: list[np.ndarray] = []
    n_scored = 0

    for batch in streamer.stream():
        feats: list[np.ndarray] = []
        n_in_batch = len(batch)
        for i in range(n_in_batch):
            if max_events is not None and n_scored >= max_events:
                break

            if waveform_branch is not None:
                raw = ak.to_numpy(ak.flatten(batch[waveform_branch][i], axis=None)).astype(
                    np.float32
                )
                feat = extract_tpc_features(raw, input_dim)
            else:
                event_data = {
                    b: ak.to_numpy(ak.flatten(batch[b][i], axis=None)).astype(np.float32)
                    for b in hit_branches
                }
                feat = extract_hit_features(event_data, hit_branches, input_dim)

            feats.append(feat)
            n_scored += 1

        if feats:
            feat_arr = np.stack(feats, axis=0).astype(np.float32)
            score_chunks.append(scorer.score(feat_arr))

        if max_events is not None and n_scored >= max_events:
            break

    if not score_chunks:
        logger.warning("No events scored from ROOT input. Returning empty score array.")
        return np.empty((0,), dtype=np.float32)

    return np.concatenate(score_chunks)
def _infer_gnn(cfg: dict, checkpoint: str, output: str) -> None:
    """Run per-node GNN inference and save results as a compressed npz archive.
            input_dim = model_cfg.get("input_dim", 256)
            features = _truncate_or_pad(features, input_dim)
            scores = scorer.score(features)
    node_scores   : (N_windows, N_channels) float32 — per-channel MSE, NaN = inactive
    scores        : (N_windows,) float32 — mean active-node MSE per window
    scores_max    : (N_windows,) float32 — max active-node MSE per window
    event_index   : (N_windows,) int64
    is_anomaly    : (N_windows,) bool — only present when threshold is configured
    """
    from torch_geometric.loader import DataLoader as PyGDataLoader

    from sbn_anomaly.infer.inferrer import GNNScorer
    from sbn_anomaly.models.gnn_forecaster_pyg import GNNForecasterPyG

    logger = logging.getLogger(__name__)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    infer_cfg = cfg.get("inference", {})

    input_path = (
        infer_cfg.get("input_path")
        or data_cfg.get("events_path")
        or data_cfg.get("windows_path")
    )
    if not input_path:
        raise ValueError(
            "Set inference.input_path (or data.events_path) in the config for GNN inference."
        )

    history = int(model_cfg.get("history", 4))
    stride = int(data_cfg.get("stride", 1))
    radius = int(data_cfg.get("adjacency_radius", 4))
    node_features = data_cfg.get("node_features") or None
    window_size = int(data_cfg.get("window_size", 20))
    n_bins = int(data_cfg.get("n_temporal_bins", 4))

    # Detect sparse events NPZ (has 'channels_flat') vs legacy dense windows NPZ.
    archive = np.load(input_path, allow_pickle=False)
    is_sparse = isinstance(archive, np.lib.npyio.NpzFile) and "channels_flat" in archive

    if is_sparse:
        from sbn_anomaly.data.sparse_window_dataset import SparseWindowDatasetPyG

        logger.info("Loading sparse events from %s ...", input_path)
        dataset = SparseWindowDatasetPyG.from_npz(
            input_path,
            history=history,
            window_size=window_size,
            n_bins=n_bins,
            stride=stride,
            radius=radius,
            node_features=node_features,
            prune_inactive=True,
        )
    else:
        from sbn_anomaly.data.graph_window_dataset_pyg import GraphWindowDatasetPyG

        logger.info("Loading dense windows from %s ...", input_path)
        if isinstance(archive, np.lib.npyio.NpzFile):
            windows = archive["windows"] if "windows" in archive else archive["features"]
        else:
            windows = np.asarray(archive)
        if windows.ndim == 4:
            N, C, T, F = windows.shape
            windows = windows.reshape(N, C, T * F)
            logger.info("Reshaped windows (%d,%d,%d,%d) -> (%d,%d,%d)", N, C, T, F, N, C, T * F)
        dataset = GraphWindowDatasetPyG(
            windows,
            history=history,
            stride=stride,
            radius=radius,
            prune_inactive=True,
            node_feature_names=node_features,
        )

    num_channels = dataset.num_nodes
    frame_feat_dim = dataset.node_feat_dim

    # Keep a reference to the base dataset before any Subset so we can call
    # window_metadata() with the correct window indices afterward.
    base_dataset = dataset
    scored_indices = range(len(dataset))

    max_windows = infer_cfg.get("max_windows")
    if max_windows is not None and int(max_windows) < len(dataset):
        from torch.utils.data import Subset
        total = len(dataset)
        scored_indices = range(int(max_windows))
        dataset = Subset(base_dataset, scored_indices)
        logger.info("Limiting inference to first %d windows (of %d total)", int(max_windows), total)

    logger.info(
        "GNN inference dataset: %d windows, %d channels, frame_feat_dim=%d",
        len(dataset), num_channels, frame_feat_dim,
    )

    loader = PyGDataLoader(
        dataset,
        batch_size=int(infer_cfg.get("batch_size", 4)),
        shuffle=False,  # preserve window order
        num_workers=int(infer_cfg.get("num_workers", 4)),
    )

    model = GNNForecasterPyG(
        frame_feat_dim=frame_feat_dim,
        target_dim=frame_feat_dim,
        gnn_hidden=int(model_cfg.get("gnn_hidden", 64)),
        gnn_layers=int(model_cfg.get("gnn_layers", 2)),
        gru_hidden=int(model_cfg.get("gru_hidden", 128)),
        gru_layers=int(model_cfg.get("gru_layers", 1)),
        history=history,
        dropout=float(model_cfg.get("dropout", 0.1)),
    )

    threshold = infer_cfg.get("threshold")
    scorer = GNNScorer.from_checkpoint(
        checkpoint_path=checkpoint,
        model=model,
        num_channels=num_channels,
        threshold=threshold,
    )

    node_scores, scores_mean, scores_max = scorer.score_loader(loader)
    logger.info(
        "Scored %d windows — node_scores shape %s, mean score %.4g",
        len(scores_mean), node_scores.shape, float(scores_mean.mean()) if scores_mean.size else float("nan"),
    )

    save_dict: dict[str, np.ndarray] = {
        "node_scores": node_scores,                                    # (N, C)
        "scores": scores_mean,                                         # (N,)
        "scores_max": scores_max,                                      # (N,)
        "window_index": np.arange(len(scores_mean), dtype=np.int64),  # sequential window position
        "num_channels": np.array(num_channels, dtype=np.int64),
    }
    if threshold is not None:
        save_dict["is_anomaly"] = scores_mean > float(threshold)
        save_dict["threshold"] = np.array(threshold, dtype=np.float32)
    node_feat_names = data_cfg.get("node_features")
    if node_feat_names:
        save_dict["node_feature_names"] = np.asarray(node_feat_names, dtype=str)

    # Attach per-window provenance (run/subrun/event/filename) when available
    if hasattr(base_dataset, "window_metadata"):
        provenance = base_dataset.window_metadata(indices=scored_indices)
        if provenance:
            for key in ("first_run", "first_subrun", "first_event_num", "last_event_num",
                        "first_file_idx", "last_file_idx"):
                if key in provenance:
                    save_dict[key] = provenance[key]
            if provenance.get("filenames"):
                save_dict["filenames"] = np.array(provenance["filenames"], dtype="U512")
            logger.info("Attached provenance metadata for %d windows", len(scores_mean))
        else:
            logger.info(
                "No provenance metadata available — re-stream from ROOT with save_events_path "
                "set to generate an events NPZ that includes run/subrun/event/filename."
            )

    out_path = Path(output)
    if out_path.suffix.lower() != ".npz":
        out_path = out_path.with_suffix(".npz")
    np.savez_compressed(out_path, **save_dict)
    logger.info("Saved GNN scores to %s", out_path)


def _build_scorer(cfg: dict, model_type: str, checkpoint: str):
    from sbn_anomaly.infer.inferrer import AnomalyScorer

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    infer_cfg = cfg.get("inference", {})
    threshold = infer_cfg.get("threshold")
    normalize = bool(data_cfg.get("normalize", False))

    if model_type == "tpc":
        from sbn_anomaly.models.tpc_model import TPCAutoencoder

        model = TPCAutoencoder(
            input_dim=model_cfg.get("input_dim", 256),
            latent_dim=model_cfg.get("latent_dim", 32),
        )
    elif model_type == "pmt":
        from sbn_anomaly.models.pmt_model import PMTAutoencoder

        model = PMTAutoencoder(
            input_dim=model_cfg.get("input_dim", 128),
            latent_dim=model_cfg.get("latent_dim", 16),
        )
    elif model_type == "fusion":
        from sbn_anomaly.models.fusion_model import FusionAutoencoder

        model = FusionAutoencoder(
            tpc_input_dim=model_cfg.get("tpc_input_dim", 256),
            pmt_input_dim=model_cfg.get("pmt_input_dim", 128),
            latent_dim=model_cfg.get("latent_dim", 32),
        )
    elif model_type == "window":
        from sbn_anomaly.models.window_model import WindowAutoencoder

        model = WindowAutoencoder(
            window_size=model_cfg.get("window_size", 256),
            n_channels=model_cfg.get("n_channels", 1),
            latent_dim=model_cfg.get("latent_dim", 64),
        )
    elif model_type == "gnn":
        from sbn_anomaly.models.gnn_forecaster_pyg import GNNForecasterPyG

        frame_feat_dim = int(model_cfg.get("frame_feat_dim") or model_cfg.get("node_feat_dim") or 5)
        model = GNNForecasterPyG(
            frame_feat_dim=frame_feat_dim,
            target_dim=frame_feat_dim,
            gnn_hidden=int(model_cfg.get("gnn_hidden", 64)),
            gnn_layers=int(model_cfg.get("gnn_layers", 2)),
            gru_hidden=int(model_cfg.get("gru_hidden", 128)),
            gru_layers=int(model_cfg.get("gru_layers", 1)),
            history=int(model_cfg.get("history", 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'.")

    return AnomalyScorer.from_checkpoint(
        checkpoint_path=checkpoint,
        model=model,
        model_type=model_type,
        threshold=threshold,
        normalize=normalize,
    )


if __name__ == "__main__":
    sys.exit(main())
