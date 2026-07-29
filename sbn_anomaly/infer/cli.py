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


def _score_threshold_from_config(
    scores: np.ndarray,
    infer_cfg: dict,
    *,
    logger: logging.Logger | None = None,
) -> float | None:
    """Resolve an inference threshold from either an explicit value or a percentile.

    ``inference.threshold`` has priority. If it is absent and
    ``inference.threshold_percentile`` is set, the threshold is computed from the
    finite scores in the current inference output.
    """
    threshold = infer_cfg.get("threshold")
    if threshold is not None:
        return float(threshold)

    percentile = infer_cfg.get("threshold_percentile")
    if percentile is None:
        return None

    percentile = float(percentile)
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"inference.threshold_percentile must be between 0 and 100, got {percentile}")

    finite = np.asarray(scores, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        if logger is not None:
            logger.warning("Cannot compute percentile threshold: no finite scores.")
        return None

    resolved = float(np.nanpercentile(finite, percentile))
    if logger is not None:
        logger.info("Resolved inference threshold from p%s of current scores: %.6g", f"{percentile:g}", resolved)
    return resolved


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run inference."""
    parser = argparse.ArgumentParser(description="Run SBN anomaly detection inference.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input to score. Overrides inference.input_path/data.events_path for "
             "gnn/raw_gnn/graph_vae (events or windows npz); .npy features for tpc/pmt/window.",
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
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional inference threshold override. If set, saved outputs include is_anomaly.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=None,
        help="Optional inference threshold percentile override. Used only when --threshold is not set.",
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
    infer_cfg = cfg.setdefault("inference", {})
    if args.threshold is not None:
        infer_cfg["threshold"] = float(args.threshold)
    if args.percentile is not None:
        if not 0.0 <= float(args.percentile) <= 100.0:
            logger.error("--percentile must be between 0 and 100.")
            return 1
        infer_cfg["threshold_percentile"] = float(args.percentile)
    checkpoint = infer_cfg.get("checkpoint_path")
    if not checkpoint:
        logger.error("inference.checkpoint_path not set in config.")
        return 1

    # GNN uses a separate PyG-based inference path with per-node output.
    # raw_gnn reuses the same path: it scores a pre-computed per-channel latent
    # windows npz (from sbn_anomaly.data.build_raw_latents).
    if model_type in ("gnn", "raw_gnn"):
        _infer_gnn(cfg, checkpoint, args.output, input_override=args.input)
        return 0

    if model_type == "graph_vae":
        _infer_graph_vae(cfg, checkpoint, args.output, input_override=args.input)
        return 0

    # raw_vae streams raw-ADC ntuples and scores each channel waveform directly.
    if model_type == "raw_vae":
        from sbn_anomaly.data.root_files import resolve_root_files

        raw_inputs: list[str] = []
        if args.root_files:
            raw_inputs.extend(args.root_files)
        if args.root_file_list:
            raw_inputs.extend(args.root_file_list)
        if args.input:
            raw_inputs.append(args.input)
        if not raw_inputs:
            raw_inputs.append(cfg.get("inference", {}).get("input_path"))
        raw_inputs = [r for r in raw_inputs if r]
        if not raw_inputs:
            logger.error("raw_vae inference needs --root-files / --input or inference.input_path.")
            return 1
        raw_files = resolve_root_files(raw_inputs)
        _infer_raw_vae(cfg, checkpoint, raw_files, args.output)
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

    threshold = _score_threshold_from_config(scores, infer_cfg, logger=logger)

    # Prepare metadata for saving alongside scores so branches can be matched.
    meta: dict[str, object] = {}
    meta["event_index"] = np.arange(len(scores), dtype=np.int64)
    if threshold is not None:
        meta["threshold"] = np.array(threshold, dtype=np.float32)
        meta["is_anomaly"] = np.asarray(scores > threshold, dtype=bool)
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
def _infer_raw_vae(
    cfg: dict,
    checkpoint: str,
    root_files: list[str],
    output: str,
) -> None:
    """Score per-channel raw waveforms with the trained VAE.

    Streams events from flat raw-ADC ntuples, pre-processes each, and computes a
    per-channel anomaly score (reconstruction MSE + beta*KL). Saves a compressed
    npz with:

        node_scores : (N_events, n_channels) float32 — per-channel score, NaN=missing
        scores      : (N_events,) float32 — mean over present channels
        scores_max  : (N_events,) float32 — max over present channels
        provenance  : (N_events, 3) int32 — (run, subrun, event)
    """
    import torch

    from sbn_anomaly.data.build_raw_latents import _load_vae_from_checkpoint
    from sbn_anomaly.data.raw_digit_reader import RawDigitReader
    from sbn_anomaly.data.raw_preprocess import preprocess_event

    logger = logging.getLogger(__name__)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    infer_cfg = cfg.get("inference", {})

    n_channels = int(data_cfg.get("n_channels", 11264))
    input_length = int(model_cfg.get("input_length", 4096))
    pp = dict(data_cfg.get("preprocess", {}) or {})
    pp.setdefault("n_ticks", input_length)
    score_beta = float(infer_cfg.get("score_beta", cfg.get("training", {}).get("beta", 1.0)))
    encode_batch = int(infer_cfg.get("batch_size", 512))
    max_events = infer_cfg.get("max_windows") or infer_cfg.get("max_events")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = _load_vae_from_checkpoint(checkpoint, model_cfg).to(device).eval()

    reader = RawDigitReader(
        root_files,
        tree_name=data_cfg.get("raw_tree_name", "rawdigits"),
        max_events=int(max_events) if max_events else None,
    )

    node_scores: list[np.ndarray] = []
    prov: list[tuple] = []
    for ev in reader:
        wf = preprocess_event(ev.adc, ev.pedestal, **pp)
        row = np.full(n_channels, np.nan, dtype=np.float32)
        chans = ev.channel.astype(np.int64)
        valid = (chans >= 0) & (chans < n_channels)
        wf, chans = wf[valid], chans[valid]
        with torch.no_grad():
            for s in range(0, wf.shape[0], encode_batch):
                x = torch.from_numpy(wf[s:s + encode_batch]).float().to(device)
                sc = vae.anomaly_score(x, beta=score_beta).cpu().numpy()
                row[chans[s:s + encode_batch]] = sc
        node_scores.append(row)
        prov.append((ev.run, ev.subrun, ev.event))

    if not node_scores:
        logger.warning("No events scored.")
        return

    node_scores_arr = np.stack(node_scores, axis=0)
    with np.errstate(invalid="ignore"):
        mean_scores = np.nanmean(node_scores_arr, axis=1).astype(np.float32)
        max_scores = np.nanmax(node_scores_arr, axis=1).astype(np.float32)
    prov_arr = np.asarray(prov, dtype=np.int32)

    out = {
        "node_scores": node_scores_arr,
        "scores": mean_scores,
        "scores_max": max_scores,
        "provenance": prov_arr,
    }
    threshold = _score_threshold_from_config(mean_scores, infer_cfg, logger=logger)
    if threshold is not None:
        out["is_anomaly"] = (mean_scores > threshold)
        out["threshold"] = np.array(threshold, dtype=np.float32)

    out_path = Path(output)
    if out_path.suffix != ".npz":
        out_path = out_path.with_suffix(".npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    logger.info(
        "Saved raw_vae scores for %d events to %s", node_scores_arr.shape[0], out_path
    )


def _infer_graph_vae(cfg: dict, checkpoint: str, output: str, input_override: str | None = None) -> None:
    """Score windows with the graph VAE.

    Saves per-window per-channel reconstruction error (`node_scores`, NaN for
    inactive channels) plus an aggregated per-window `scores` (default
    group_max_mean). Re-aggregate later with sbn_anomaly.infer.window_score.
    """
    import torch
    from torch_geometric.loader import DataLoader as PyGDataLoader

    from sbn_anomaly.data.graph_recon_dataset import GraphReconDataset
    from sbn_anomaly.models.graph_vae import GraphVAE
    from sbn_anomaly.infer.window_score import aggregate_windows, channel_to_group

    logger = logging.getLogger(__name__)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    infer_cfg = cfg.get("inference", {})

    input_path = (input_override or infer_cfg.get("input_path")
                  or data_cfg.get("events_path") or data_cfg.get("windows_path"))
    if not input_path:
        raise ValueError("Set --input, inference.input_path, or data.events_path for graph_vae.")
    logger.info("graph_vae inference input: %s", input_path)

    # Load training standardization (saved next to the checkpoint).
    std_path = data_cfg.get("standardization_path") or str(Path(checkpoint).parent / "standardization.npz")
    feat_mean = feat_std = None
    graph_feat_mean = graph_feat_std = None
    if Path(std_path).exists():
        s = np.load(std_path)
        feat_mean, feat_std = s["feature_mean"], s["feature_std"]
        if "graph_feature_mean" in s:
            graph_feat_mean, graph_feat_std = s["graph_feature_mean"], s["graph_feature_std"]
        logger.info("Loaded standardization from %s", std_path)
    else:
        logger.warning("No standardization.npz at %s; standardizing from input.", std_path)

    if not str(input_path).endswith((".npz", ".npy")):
        raise ValueError(
            f"--input must be a materialized events/windows .npz, got '{input_path}'. "
            "A ROOT-file list is not scored directly — first materialize it:\n"
            "  python -m sbn_anomaly.data.materialize_windows "
            f"--config <config> --root-file-list {input_path} --output <events.npz>\n"
            "then pass that .npz to --input."
        )
    try:
        arch = np.load(input_path, allow_pickle=True)
    except Exception as exc:
        raise ValueError(
            f"Could not read '{input_path}' as an .npz/.npy. If this is a ROOT-file "
            "list, materialize it first with sbn_anomaly.data.materialize_windows."
        ) from exc
    is_sparse = isinstance(arch, np.lib.npyio.NpzFile) and "channels_flat" in arch
    provenance = None

    if is_sparse:
        from sbn_anomaly.data.sparse_window_dataset import SparseWindowDatasetPyG
        logger.info("Loading sparse events for graph_vae from %s", input_path)
        dataset = SparseWindowDatasetPyG.from_npz(
            input_path,
            n_channels=data_cfg.get("n_channels"),
            window_size=int(data_cfg.get("window_size", 20)),
            n_bins=int(data_cfg.get("n_temporal_bins", 4)),
            stride=int(data_cfg.get("stride", 1)),
            window_mode=str(data_cfg.get("window_mode", "event")),
            window_duration=data_cfg.get("window_duration"),
            stride_duration=data_cfg.get("stride_duration"),
            radius=int(data_cfg.get("adjacency_radius", 4)),
            node_features=data_cfg.get("node_features") or None,
            prune_inactive=bool(data_cfg.get("prune_inactive", True)),
            channel_map=data_cfg.get("channel_map"),
            edge_mode=str(data_cfg.get("edge_mode", "sequential")),
            reconstruction=True,
            standardize=bool(data_cfg.get("standardize", True)),
            feature_mean=feat_mean, feature_std=feat_std,
            log_features=data_cfg.get("log_features") or None,
            graph_features=data_cfg.get("graph_features") or None,
            graph_feature_mean=graph_feat_mean, graph_feature_std=graph_feat_std,
        )
    else:
        windows = arch["windows"] if (isinstance(arch, np.lib.npyio.NpzFile) and "windows" in arch) \
            else (arch["features"] if isinstance(arch, np.lib.npyio.NpzFile) else np.asarray(arch))
        windows = np.asarray(windows)
        if windows.ndim == 4:
            N, C, T, Fd = windows.shape
            windows = windows.reshape(N, C, T * Fd)
        if isinstance(arch, np.lib.npyio.NpzFile) and "provenance" in arch:
            provenance = arch["provenance"]
        dataset = GraphReconDataset(
            windows,
            radius=int(data_cfg.get("adjacency_radius", 4)),
            prune_inactive=bool(data_cfg.get("prune_inactive", True)),
            node_feature_names=data_cfg.get("node_features") or None,
            channel_map=data_cfg.get("channel_map"),
            edge_mode=str(data_cfg.get("edge_mode", "sequential")),
            feature_mean=feat_mean, feature_std=feat_std,
            standardize=bool(data_cfg.get("standardize", True)),
            log_features=data_cfg.get("log_features") or None,
        )
    num_channels = dataset.num_nodes

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = GraphVAE(
        in_dim=dataset.node_feat_dim,
        latent_dim=int(model_cfg.get("latent_dim", 12)),
        encoder_hidden_dims=model_cfg.get("encoder_hidden_dims", [128, 64]),
        decoder_hidden_dims=model_cfg.get("decoder_hidden_dims", [32, 64]),
        dropout=float(model_cfg.get("dropout", 0.1)),
        mask_ratio=0.0,  # no masking at inference
        use_channel_idx=bool(model_cfg.get("use_channel_idx", True)),
        graph_dim=getattr(dataset, "graph_feat_dim", 0),
        conv=str(model_cfg.get("conv", "sage")),
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(
        "GraphVAE inference model params=%d  in_dim=%d  latent_dim=%d  "
        "encoder_hidden_dims=%s  decoder_hidden_dims=%s  "
        "dropout=%.3f  mask_ratio=%.3f  use_channel_idx=%s  graph_dim=%d  conv=%s",
        n_params,
        dataset.node_feat_dim,
        int(model_cfg.get("latent_dim", 12)),
        model.encoder_hidden_dims,
        model.decoder_hidden_dims,
        float(model_cfg.get("dropout", 0.1)),
        0.0,
        bool(model_cfg.get("use_channel_idx", True)),
        model.graph_dim,
        str(model_cfg.get("conv", "sage")),
    )

    logger.info("Inference model architecture:\n%s", model)

    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model = model.to(device).eval()

    loader = PyGDataLoader(dataset, batch_size=int(infer_cfg.get("batch_size", 8)), shuffle=False)
    rows: list = []
    with torch.no_grad():
        for batch in loader:
            data = batch.to(device)
            x_hat, _, _, _ = model(data)
            per_node = ((x_hat - data.y.float()) ** 2).mean(dim=-1).cpu().numpy()
            bidx = data.batch.cpu().numpy()
            amask = data.active_mask.cpu().numpy()
            ng = int(getattr(data, "num_graphs", 1))
            for g in range(ng):
                sel = bidx == g
                row = np.full(num_channels, np.nan, dtype=np.float32)
                row[amask[sel]] = per_node[sel]
                rows.append(row)

    node_scores = np.stack(rows, axis=0) if rows else np.zeros((0, num_channels), np.float32)

    # Per-window provenance from the sparse events' metadata, when available.
    # event_count needs no provenance data (just window bounds), so it's always
    # present; the run/subrun/event triple requires provenance and is only
    # added when present, hence the explicit key check rather than `if meta:`.
    event_count = None
    if is_sparse:
        meta = dataset.window_metadata()
        event_count = meta.get("event_count")
        if provenance is None and "first_run" in meta:
            provenance = np.stack(
                [meta["first_run"], meta["first_subrun"], meta["first_event_num"]], axis=1
            ).astype(np.int32)

    aggregator = str(infer_cfg.get("window_aggregator", "group_max_mean"))
    groups = None
    if aggregator == "group_max_mean" and data_cfg.get("channel_map"):
        groups = channel_to_group(data_cfg["channel_map"],
                                  level=str(infer_cfg.get("group_level", "femb")),
                                  num_channels=num_channels)
    elif aggregator == "group_max_mean":
        logger.warning("group_max_mean needs data.channel_map; falling back to mean.")
        aggregator = "mean"
    scores = aggregate_windows(node_scores, aggregator, groups=groups,
                               k=int(infer_cfg.get("topk", 64)))

    # Per-channel summary across all windows (which channels are anomalous
    # overall), alongside the per-window per-channel node_scores and the global
    # per-window scores.
    # Channels never active over all windows are all-NaN columns -> NaN summary
    # (expected); silence the benign "Mean/All-NaN of empty slice" warnings.
    import warnings
    with np.errstate(invalid="ignore", all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        channel_mean_error = np.nanmean(node_scores, axis=0).astype(np.float32) \
            if node_scores.shape[0] else np.full(num_channels, np.nan, np.float32)
        channel_max_error = np.nanmax(node_scores, axis=0).astype(np.float32) \
            if node_scores.shape[0] else np.full(num_channels, np.nan, np.float32)
        channel_active_frac = np.mean(np.isfinite(node_scores), axis=0).astype(np.float32) \
            if node_scores.shape[0] else np.zeros(num_channels, np.float32)

    out = {"node_scores": node_scores, "scores": scores.astype(np.float32),
           "window_index": np.arange(node_scores.shape[0], dtype=np.int64),
           "channel_mean_error": channel_mean_error,
           "channel_max_error": channel_max_error,
           "channel_active_frac": channel_active_frac}
    if provenance is not None:
        out["provenance"] = provenance
    if event_count is not None:
        # Trigger count per window -- lets scores.npz be correlated against
        # the rate directly (e.g. does anomaly score track a rate change).
        out["event_count"] = event_count
    threshold = _score_threshold_from_config(scores, infer_cfg, logger=logger)
    if threshold is not None:
        out["is_anomaly"] = (scores > threshold)
        out["threshold"] = np.array(threshold, dtype=np.float32)

    # Rank the worst channels for a quick console readout.
    if node_scores.shape[0] and np.isfinite(channel_mean_error).any():
        order = np.argsort(np.nan_to_num(channel_mean_error, nan=-np.inf))[::-1]
        top = [c for c in order if np.isfinite(channel_mean_error[c])][:10]
        logger.info("Top-10 channels by mean recon error: %s",
                    ", ".join(f"{int(c)}:{channel_mean_error[c]:.3g}" for c in top))

    out_path = Path(output)
    if out_path.suffix != ".npz":
        out_path = out_path.with_suffix(".npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    logger.info("Saved graph_vae scores for %d windows to %s (aggregator=%s)",
                node_scores.shape[0], out_path, aggregator)

    # Plot the per-window anomaly-score distribution and score-over-time next to
    # the output, unless disabled.
    if bool(infer_cfg.get("plot", True)) and scores.size:
        try:
            from sbn_anomaly.infer.window_score import max_score
            from sbn_anomaly.utils.plotting import (
                save_score_distribution_plot,
                save_score_over_time_plot,
                save_node_mse_plot,
            )
            plot_dir = out_path.parent
            thr = float(threshold) if threshold is not None else None
            save_score_distribution_plot(
                scores, plot_dir, filename=out_path.stem + "_score_distribution.png",
                threshold=thr, title=f"Window score distribution ({aggregator})",
            )
            save_score_over_time_plot(
                scores, max_score(node_scores), plot_dir,
                filename=out_path.stem + "_score_over_time.png", threshold=thr,
                title="Window anomaly score over time",
            )
            save_node_mse_plot(
                channel_mean_error, plot_dir,
                filename=out_path.stem + "_channel_error.png",
                title="Per-channel mean reconstruction error",
            )
            logger.info("Saved score + per-channel plots next to %s", out_path)
        except Exception as exc:
            logger.warning("Failed to save score plots: %s", exc)


def _infer_gnn(cfg: dict, checkpoint: str, output: str, input_override: str | None = None) -> None:
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
        input_override
        or infer_cfg.get("input_path")
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
    channel_map = data_cfg.get("channel_map")
    edge_mode = str(data_cfg.get("edge_mode", "sequential"))

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
            channel_map=channel_map,
            edge_mode=edge_mode,
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
            channel_map=channel_map,
            edge_mode=edge_mode,
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
    threshold = _score_threshold_from_config(scores_mean, infer_cfg, logger=logger)
    if threshold is not None:
        save_dict["is_anomaly"] = scores_mean > threshold
        save_dict["threshold"] = np.array(threshold, dtype=np.float32)
    node_feat_names = data_cfg.get("node_features")
    if node_feat_names:
        save_dict["node_feature_names"] = np.asarray(node_feat_names, dtype=str)

    # Attach per-window provenance (run/subrun/event/filename) when available.
    # event_count needs no provenance data (just window bounds) so it's
    # attached separately from the has-real-provenance check below.
    if hasattr(base_dataset, "window_metadata"):
        provenance = base_dataset.window_metadata(indices=scored_indices)
        if "event_count" in provenance:
            save_dict["event_count"] = provenance["event_count"]
        has_provenance = "first_run" in provenance
        if has_provenance:
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
    threshold_percentile = infer_cfg.get("threshold_percentile")
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
        threshold_percentile=threshold_percentile,
        normalize=normalize,
    )


if __name__ == "__main__":
    sys.exit(main())
