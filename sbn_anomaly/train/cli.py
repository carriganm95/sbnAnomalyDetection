"""Training CLI entry-point.

Usage::

    sbn-train --config configs/tpc.yaml
    sbn-train --config configs/tpc.yaml --root-files /data/run1.root /data/run2.root
    python -m sbn_anomaly.train.cli --config configs/pmt.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import random
from pathlib import Path

import yaml

from sbn_anomaly.utils.logging import setup_logging


def _split_dataset_for_validation(dataset, validation_split: float, seed: int):
    """Split a map-style dataset into train/validation subsets when requested."""
    if validation_split <= 0.0 or validation_split >= 1.0:
        return dataset, None
    try:
        dataset_size = len(dataset)
    except TypeError:
        return dataset, None
    if dataset_size < 2:
        return dataset, None

    validation_size = int(round(dataset_size * validation_split))
    if validation_size <= 0 or validation_size >= dataset_size:
        return dataset, None

    from torch.utils.data import random_split
    import torch

    generator = torch.Generator().manual_seed(seed)
    train_size = dataset_size - validation_size
    return random_split(dataset, [train_size, validation_size], generator=generator)


def _split_root_files_for_validation(root_files: list[str], validation_split: float, seed: int):
    """Split ROOT file lists for streaming validation when no explicit eval set is given."""
    if validation_split <= 0.0 or validation_split >= 1.0:
        return list(root_files), []
    if len(root_files) < 2:
        return list(root_files), []

    validation_size = int(round(len(root_files) * validation_split))
    if validation_size <= 0 or validation_size >= len(root_files):
        return list(root_files), []

    shuffled = list(root_files)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled[:-validation_size], shuffled[-validation_size:]


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the appropriate trainer."""
    parser = argparse.ArgumentParser(
        description="Train an SBN anomaly detection model."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML configuration file (e.g. configs/tpc.yaml).",
    )
    parser.add_argument(
        "--root-files",
        nargs="+",
        default=None,
        metavar="PATH",
        help=(
            "One or more ROOT file paths (glob patterns accepted). "
            "When provided with model_type=tpc, streams directly from ROOT "
            "instead of loading pre-computed .npy feature files."
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
            "patterns. This is useful for train/test file manifests."
        ),
    )
    parser.add_argument(
        "--eval-root-files",
        nargs="+",
        default=None,
        metavar="PATH",
        help=(
            "Optional validation ROOT file paths or glob patterns for TPC training. "
            "When supplied, validation loss is computed on this held-out set."
        ),
    )
    parser.add_argument(
        "--eval-root-file-list",
        nargs="+",
        default=None,
        metavar="FILE",
        help=(
            "Optional validation ROOT file list(s), one file path per line, for TPC training."
        ),
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

    root_files: list[str] | None = None
    if args.root_files or args.root_file_list:
        from sbn_anomaly.data.root_files import resolve_root_files

        root_inputs = []
        if args.root_files:
            root_inputs.extend(args.root_files)
        if args.root_file_list:
            root_inputs.extend(args.root_file_list)
        root_files = resolve_root_files(root_inputs)

    eval_root_files: list[str] | None = None
    if args.eval_root_files or args.eval_root_file_list:
        from sbn_anomaly.data.root_files import resolve_root_files

        eval_inputs = []
        if args.eval_root_files:
            eval_inputs.extend(args.eval_root_files)
        if args.eval_root_file_list:
            eval_inputs.extend(args.eval_root_file_list)
        eval_root_files = resolve_root_files(eval_inputs)

    model_type = cfg.get("model_type", "").lower()
    logger.info("Starting training for model_type='%s'", model_type)

    if model_type == "tpc":
        _train_tpc(cfg, root_files=root_files, eval_root_files=eval_root_files)
    elif model_type == "pmt":
        _train_pmt(cfg)
    elif model_type == "fusion":
        _train_fusion(cfg)
    elif model_type == "window":
        _train_window(cfg)
    elif model_type == "gnn":
        _train_gnn(cfg, root_files=root_files)
    else:
        logger.error(
            "Unknown model_type '%s'. Choose from: tpc, pmt, fusion, window.", model_type
        )
        return 1

    return 0


def _train_tpc(
    cfg: dict,
    root_files: list[str] | None = None,
    eval_root_files: list[str] | None = None,
) -> None:
    import numpy as np
    from torch.utils.data import DataLoader

    from sbn_anomaly.data.dataset import TPCDataset
    from sbn_anomaly.models.tpc_model import TPCAutoencoder
    from sbn_anomaly.train.tpc_trainer import TPCTrainer

    logger = logging.getLogger(__name__)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    validation_split = float(train_cfg.get("validation_split", 0.0) or 0.0)
    validation_seed = int(train_cfg.get("validation_seed", 42))

    input_dim = model_cfg.get("input_dim", 256)

    if root_files:
        from sbn_anomaly.data.stream_dataset import TPCStreamDataset

        train_files = list(root_files)
        val_files: list[str] = []
        if eval_root_files:
            val_files = list(eval_root_files)
        elif validation_split > 0.0:
            train_files, val_files = _split_root_files_for_validation(
                train_files,
                validation_split,
                validation_seed,
            )

        logger.info(
            "Streaming TPC training from %d training ROOT file(s)%s",
            len(train_files),
            f" and {len(val_files)} validation ROOT file(s)" if val_files else "",
        )
        dataset = TPCStreamDataset(
            file_paths=train_files,
            tree_name=data_cfg.get("tree_name", "sbn_tree"),
            waveform_branch=data_cfg.get("waveform_branch", "tpc_waveform"),
            hit_branches=data_cfg.get("hit_branches"),
            branches=data_cfg.get("tpc_branches"),
            input_dim=input_dim,
            batch_size=data_cfg.get("batch_size_stream", 512),
            normalize=data_cfg.get("normalize", False),
            max_events=train_cfg.get("max_events"),
        )
        loader = DataLoader(
            dataset,
            batch_size=train_cfg.get("batch_size", 256),
            # shuffle is not supported for IterableDataset
            num_workers=int(train_cfg.get("num_workers", 0)),
        )
        validation_loader = None
        if val_files:
            validation_dataset = TPCStreamDataset(
                file_paths=val_files,
                tree_name=data_cfg.get("tree_name", "sbn_tree"),
                waveform_branch=data_cfg.get("waveform_branch", "tpc_waveform"),
                hit_branches=data_cfg.get("hit_branches"),
                branches=data_cfg.get("tpc_branches"),
                input_dim=input_dim,
                batch_size=data_cfg.get("batch_size_stream", 512),
                normalize=data_cfg.get("normalize", False),
                max_events=train_cfg.get("validation_max_events", train_cfg.get("max_events")),
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=train_cfg.get("batch_size", 256),
                num_workers=int(train_cfg.get("num_workers", 0)),
            )
        steps_per_epoch = train_cfg.get(
            "steps_per_epoch",
            # Default of 100 batches per epoch is a sensible starting point for
            # streaming; tune via training.steps_per_epoch in the YAML config.
            100,
        )
    else:
        features = np.load(data_cfg["features_path"], allow_pickle=True)
        dataset = TPCDataset(
            features,
            input_dim=input_dim,
            normalize=data_cfg.get("normalize", False),
        )
        train_dataset, val_dataset = _split_dataset_for_validation(dataset, validation_split, validation_seed)
        # Honor configured steps_per_epoch for map-style datasets so users can
        # override the default number of batches per epoch (useful for large
        # datasets or repeated sampling experiments). If not set, iterate
        # through the full dataset once per epoch.
        steps_per_epoch = train_cfg.get("steps_per_epoch", None)

        # Configure sampling for map-style datasets. By default we allow
        # sampling with replacement so `steps_per_epoch` batches can be
        # produced even when steps_per_epoch * batch_size > dataset size.
        sampling_with_replacement = bool(train_cfg.get("sampling_with_replacement", True))
        sampling_seed = train_cfg.get("sampling_seed", None)
        batch_size = int(train_cfg.get("batch_size", 256))
        if steps_per_epoch is not None and sampling_with_replacement:
            # Sample with replacement to reach steps_per_epoch * batch_size
            from torch.utils.data import RandomSampler
            import torch

            num_samples = int(steps_per_epoch) * batch_size
            if sampling_seed is not None:
                gen = torch.Generator().manual_seed(int(sampling_seed))
                sampler = RandomSampler(train_dataset, replacement=True, num_samples=num_samples, generator=gen)
            else:
                sampler = RandomSampler(train_dataset, replacement=True, num_samples=num_samples)
            loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=int(train_cfg.get("num_workers", 0)),
            )
        else:
            # Default behavior: shuffle without replacement (one epoch = one pass)
            loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=int(train_cfg.get("num_workers", 0)),
            )
        validation_loader = None
        if val_dataset is not None:
            validation_loader = DataLoader(
                val_dataset,
                batch_size=train_cfg.get("batch_size", 256),
                shuffle=False,
                num_workers=int(train_cfg.get("num_workers", 0)),
            )
        # Honor configured steps_per_epoch for map-style datasets so users can
        # override the default number of batches per epoch (useful for large
        # datasets or repeated sampling experiments). If not set, iterate
        # through the full dataset once per epoch.
        steps_per_epoch = train_cfg.get("steps_per_epoch", None)

    model = TPCAutoencoder(
        input_dim=input_dim,
        latent_dim=model_cfg.get("latent_dim", 32),
    )
    trainer = TPCTrainer(
        model=model,
        lr=train_cfg.get("lr", 1e-3),
        max_epochs=train_cfg.get("max_epochs", 50),
        checkpoint_dir=train_cfg.get("checkpoint_dir"),
        steps_per_epoch=steps_per_epoch,
        anomaly_threshold=train_cfg.get("anomaly_threshold"),
        reconstruction_plot_max_values=train_cfg.get(
            "reconstruction_plot_max_values", 50000
        ),
        save_best_only=train_cfg.get("save_best_only", False),
    )
    trainer.train(
        loader,
        validation_loader=validation_loader,
        metrics_max_samples=train_cfg.get("metrics_max_samples", 20000),
    )
    output = train_cfg.get("output_path")
    if output:
        trainer.save(output)

    plot_dir = train_cfg.get("checkpoint_dir")
    if plot_dir is None and output:
        plot_dir = str(Path(output).parent)
    trainer.save_training_history(plot_dir)
    trainer.save_training_plots(plot_dir, bins=train_cfg.get("reconstruction_hist2d_bins"))
    logger.info("TPC training complete.")


def _train_pmt(cfg: dict) -> None:
    import numpy as np
    from torch.utils.data import DataLoader

    from sbn_anomaly.data.dataset import PMTDataset
    from sbn_anomaly.models.pmt_model import PMTAutoencoder
    from sbn_anomaly.train.pmt_trainer import PMTTrainer

    logger = logging.getLogger(__name__)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    features = np.load(data_cfg["features_path"], allow_pickle=True)
    dataset = PMTDataset(features)

    model = PMTAutoencoder(
        input_dim=model_cfg.get("input_dim", 128),
        latent_dim=model_cfg.get("latent_dim", 16),
    )
    trainer = PMTTrainer(
        model=model,
        lr=train_cfg.get("lr", 1e-3),
        max_epochs=train_cfg.get("max_epochs", 50),
        checkpoint_dir=train_cfg.get("checkpoint_dir"),
        anomaly_threshold=train_cfg.get("anomaly_threshold"),
        reconstruction_plot_max_values=train_cfg.get(
            "reconstruction_plot_max_values", 50000
        ),
        save_best_only=train_cfg.get("save_best_only", False),
    )
    validation_split = float(train_cfg.get("validation_split", 0.0) or 0.0)
    validation_seed = int(train_cfg.get("validation_seed", 42))
    train_dataset, val_dataset = _split_dataset_for_validation(dataset, validation_split, validation_seed)
    loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 256),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    validation_loader = None
    if val_dataset is not None:
        validation_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg.get("batch_size", 256),
            shuffle=False,
            num_workers=int(train_cfg.get("num_workers", 0)),
        )
    trainer.train(
        loader,
        validation_loader=validation_loader,
        metrics_max_samples=train_cfg.get("metrics_max_samples", 20000),
    )
    output = train_cfg.get("output_path")
    if output:
        trainer.save(output)

    plot_dir = train_cfg.get("checkpoint_dir")
    if plot_dir is None and output:
        plot_dir = str(Path(output).parent)
    trainer.save_training_history(plot_dir)
    trainer.save_training_plots(plot_dir, bins=train_cfg.get("reconstruction_hist2d_bins"))
    logger.info("PMT training complete.")


def _train_fusion(cfg: dict) -> None:
    import numpy as np
    from torch.utils.data import DataLoader

    from sbn_anomaly.data.dataset import FusionDataset
    from sbn_anomaly.models.fusion_model import FusionAutoencoder
    from sbn_anomaly.train.fusion_trainer import FusionTrainer

    logger = logging.getLogger(__name__)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    tpc_features = np.load(data_cfg["tpc_features_path"], allow_pickle=True)
    pmt_features = np.load(data_cfg["pmt_features_path"], allow_pickle=True)
    dataset = FusionDataset(tpc_features, pmt_features)
    validation_split = float(train_cfg.get("validation_split", 0.0) or 0.0)
    validation_seed = int(train_cfg.get("validation_seed", 42))
    train_dataset, val_dataset = _split_dataset_for_validation(dataset, validation_split, validation_seed)
    loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 256),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    validation_loader = None
    if val_dataset is not None:
        validation_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg.get("batch_size", 256),
            shuffle=False,
            num_workers=int(train_cfg.get("num_workers", 0)),
        )

    model = FusionAutoencoder(
        tpc_input_dim=model_cfg.get("tpc_input_dim", 256),
        pmt_input_dim=model_cfg.get("pmt_input_dim", 128),
        latent_dim=model_cfg.get("latent_dim", 32),
    )
    trainer = FusionTrainer(
        model=model,
        lr=train_cfg.get("lr", 1e-3),
        max_epochs=train_cfg.get("max_epochs", 50),
        checkpoint_dir=train_cfg.get("checkpoint_dir"),
        anomaly_threshold=train_cfg.get("anomaly_threshold"),
        reconstruction_plot_max_values=train_cfg.get(
            "reconstruction_plot_max_values", 50000
        ),
        save_best_only=train_cfg.get("save_best_only", False),
    )
    trainer.train(
        loader,
        validation_loader=validation_loader,
        metrics_max_samples=train_cfg.get("metrics_max_samples", 20000),
    )
    output = train_cfg.get("output_path")
    if output:
        trainer.save(output)

    plot_dir = train_cfg.get("checkpoint_dir")
    if plot_dir is None and output:
        plot_dir = str(Path(output).parent)
    trainer.save_training_history(plot_dir)
    trainer.save_training_plots(plot_dir, bins=train_cfg.get("reconstruction_hist2d_bins"))
    logger.info("Fusion training complete.")


def _train_window(cfg: dict) -> None:
    import numpy as np
    from torch.utils.data import DataLoader

    from sbn_anomaly.data.dataset import WindowDataset
    from sbn_anomaly.models.window_model import WindowAutoencoder
    from sbn_anomaly.train.window_trainer import WindowTrainer

    logger = logging.getLogger(__name__)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    signal = np.load(data_cfg["signal_path"], allow_pickle=True)
    window_size = model_cfg.get("window_size", 256)
    dataset = WindowDataset(signal, window_size=window_size)
    validation_split = float(train_cfg.get("validation_split", 0.0) or 0.0)
    validation_seed = int(train_cfg.get("validation_seed", 42))
    train_dataset, val_dataset = _split_dataset_for_validation(dataset, validation_split, validation_seed)
    loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 256),
        shuffle=True,
    )
    validation_loader = None
    if val_dataset is not None:
        validation_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg.get("batch_size", 256),
            shuffle=False,
        )

    model = WindowAutoencoder(
        window_size=window_size,
        n_channels=model_cfg.get("n_channels", 1),
        latent_dim=model_cfg.get("latent_dim", 64),
    )
    trainer = WindowTrainer(
        model=model,
        lr=train_cfg.get("lr", 1e-3),
        max_epochs=train_cfg.get("max_epochs", 50),
        checkpoint_dir=train_cfg.get("checkpoint_dir"),
        anomaly_threshold=train_cfg.get("anomaly_threshold"),
        reconstruction_plot_max_values=train_cfg.get(
            "reconstruction_plot_max_values", 50000
        ),
        save_best_only=train_cfg.get("save_best_only", False),
    )
    trainer.train(
        loader,
        validation_loader=validation_loader,
        metrics_max_samples=train_cfg.get("metrics_max_samples", 20000),
    )
    output = train_cfg.get("output_path")
    if output:
        trainer.save(output)

    plot_dir = train_cfg.get("checkpoint_dir")
    if plot_dir is None and output:
        plot_dir = str(Path(output).parent)
    trainer.save_training_history(plot_dir)
    trainer.save_training_plots(plot_dir, bins=train_cfg.get("reconstruction_hist2d_bins"))
    logging.getLogger(__name__).info("Window training complete.")


def _train_gnn(cfg: dict, root_files: list[str] | None = None) -> None:
    import numpy as np
    import time

    from torch_geometric.loader import DataLoader as PyGDataLoader

    from sbn_anomaly.data.graph_window_dataset_pyg import GraphWindowDatasetPyG
    from sbn_anomaly.data.sparse_window_dataset import SparseWindowDatasetPyG
    from sbn_anomaly.models.gnn_forecaster_pyg import GNNForecasterPyG
    from sbn_anomaly.train.gnn_trainer import GNNTrainerPyG
    from sbn_anomaly.utils.plotting import (
        save_node_mse_plot,
        save_score_distribution_plot,
        save_score_over_time_plot,
    )

    logger = logging.getLogger(__name__)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    history = int(model_cfg.get("history", 4))
    stride = int(data_cfg.get("stride", 1))
    radius = int(data_cfg.get("adjacency_radius", 4))
    node_feature_names = data_cfg.get("node_features") or None
    window_size = int(data_cfg.get("window_size", 20))
    n_bins = int(data_cfg.get("n_temporal_bins", 4))

    events_path = data_cfg.get("events_path")

    if root_files or events_path:
        # --- Sparse path: load events once, form windows lazily ---
        dataset_kwargs = dict(
            history=history,
            window_size=window_size,
            n_bins=n_bins,
            stride=stride,
            radius=radius,
            node_features=node_feature_names,
            prune_inactive=True,
        )
        if events_path and not root_files:
            logger.info("Loading sparse events from %s ...", events_path)
            t_load = time.perf_counter()
            dataset = SparseWindowDatasetPyG.from_npz(events_path, **dataset_kwargs)
            logger.info("Events loaded in %.1fs", time.perf_counter() - t_load)
        else:
            logger.info("Streaming events from %d ROOT file(s)...", len(root_files))
            t_load = time.perf_counter()
            dataset = SparseWindowDatasetPyG.from_root(
                root_files=root_files,
                tree_name=data_cfg.get("tree_name", "sbn_tree"),
                hit_branches=data_cfg.get("hit_branches", []),
                n_channels=data_cfg.get("n_channels") or None,
                sort_events=True,
                tpc_branches=data_cfg.get("tpc_branches") or None,
                max_events=data_cfg.get("max_events") or None,
                **dataset_kwargs,
            )
            logger.info("Events loaded in %.1fs", time.perf_counter() - t_load)
            # Optionally cache the compact events file for future runs
            save_events_path = train_cfg.get("save_events_path") or data_cfg.get("save_events_path")
            if save_events_path:
                dataset.save_events(save_events_path)
                logger.info("Saved compact events to %s for future runs", save_events_path)

        num_channels = dataset.num_nodes
        frame_feat_dim = dataset.node_feat_dim

    else:
        # --- Dense pre-materialized path (legacy) ---
        logger.info("Loading windows from %s ...", data_cfg["windows_path"])
        t_load = time.perf_counter()
        windows_path = data_cfg["windows_path"]
        if str(windows_path).endswith(".npy"):
            windows = np.load(windows_path, mmap_mode="r")
        else:
            windows_archive = np.load(windows_path, allow_pickle=True)
            if isinstance(windows_archive, np.lib.npyio.NpzFile):
                windows = windows_archive["windows"] if "windows" in windows_archive else windows_archive["features"]
            else:
                windows = np.asarray(windows_archive)
        logger.info("Windows loaded in %.1fs", time.perf_counter() - t_load)

        if windows.ndim == 4:
            N, N_channels, n_bins_loaded, n_features = windows.shape
            windows = windows.reshape(N, N_channels, n_bins_loaded * n_features)
            logger.info(
                "Reshaped windows (%d, %d, %d, %d) -> (%d, %d, %d)",
                N, N_channels, n_bins_loaded, n_features, N, N_channels, n_bins_loaded * n_features,
            )

    if not (root_files or events_path):
        dataset = GraphWindowDatasetPyG(
            windows,
            history=history,
            stride=stride,
            radius=radius,
            prune_inactive=True,
            node_feature_names=node_feature_names,
        )
        num_channels = dataset.num_nodes
        frame_feat_dim = dataset.node_feat_dim

    logger.info("Splitting train/validation ...")
    validation_split = float(train_cfg.get("validation_split", 0.0) or 0.0)
    validation_seed = int(train_cfg.get("validation_seed", 42))
    train_dataset, val_dataset = _split_dataset_for_validation(dataset, validation_split, validation_seed)

    batch_size = int(train_cfg.get("batch_size", 4))
    num_workers = int(train_cfg.get("num_workers", 4))

    n_train = len(train_dataset)
    if n_train == 0:
        raise ValueError(
            "Training dataset is empty. Check your windows file, history, and validation_split settings."
        )
    if batch_size > n_train:
        logger.warning(
            "batch_size=%d exceeds training dataset size=%d; clamping to %d",
            batch_size, n_train, n_train,
        )
        batch_size = n_train

    logger.info(
        "Train samples: %d  Val samples: %s  Batch size: %d  Num workers: %d",
        n_train,
        len(val_dataset) if val_dataset is not None else "none",
        batch_size, num_workers,
    )
    persistent = num_workers > 0
    loader = PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=persistent)
    # Ordered loader for post-training score collection (no shuffle = temporal order preserved)
    eval_loader = PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=persistent)
    validation_loader = None
    if val_dataset is not None:
        val_batch_size = min(batch_size, len(val_dataset)) if len(val_dataset) > 0 else 1
        validation_loader = PyGDataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, persistent_workers=persistent)

    logger.info("Building model ...")
    model = GNNForecasterPyG(
        frame_feat_dim=frame_feat_dim,
        target_dim=frame_feat_dim,
        gnn_hidden=int(model_cfg.get("gnn_hidden", 64)),
        gnn_layers=int(model_cfg.get("gnn_layers", 2)),
        gru_hidden=int(model_cfg.get("gru_hidden", 128)),
        gru_layers=int(model_cfg.get("gru_layers", 1)),
        history=history,
        dropout=float(model_cfg.get("dropout", 0.1)),
        norm_type=str(model_cfg.get("norm_type", "none")),
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model: GNNForecasterPyG  params=%d  frame_feat_dim=%d  gnn_layers=%d  "
        "gnn_hidden=%d  gru_layers=%d  gru_hidden=%d  history=%d",
        n_params, frame_feat_dim,
        int(model_cfg.get("gnn_layers", 2)), int(model_cfg.get("gnn_hidden", 64)),
        int(model_cfg.get("gru_layers", 1)), int(model_cfg.get("gru_hidden", 128)),
        history,
    )

    trainer = GNNTrainerPyG(
        model=model,
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
        max_epochs=int(train_cfg.get("max_epochs", 20)),
        checkpoint_dir=train_cfg.get("checkpoint_dir"),
        log_interval=int(train_cfg.get("log_interval", 50)),
        anomaly_threshold=train_cfg.get("anomaly_threshold"),
        save_best_only=bool(train_cfg.get("save_best_only", False)),
        use_amp=bool(train_cfg.get("use_amp", False)),
        score_mode=str(train_cfg.get("score_mode", "mean")),
    )

    logger.info("Starting training (%d epochs) ...", int(train_cfg.get("max_epochs", 20)))
    t_train = time.perf_counter()
    trainer.train(
        loader,
        validation_loader=validation_loader,
        metrics_max_samples=int(train_cfg.get("metrics_max_samples", 20000)),
    )
    logger.info("Training complete in %.1fs", time.perf_counter() - t_train)

    output = train_cfg.get("output_path")
    if output:
        trainer.save(output)

    plot_dir = train_cfg.get("checkpoint_dir")
    if plot_dir is None and output:
        plot_dir = str(Path(output).parent)

    trainer.save_training_history(plot_dir)
    trainer.save_training_plots(plot_dir)

    threshold = train_cfg.get("anomaly_threshold")

    # Collect scores in window order using the non-shuffled eval loader
    try:
        scores_mean, scores_max = trainer.collect_scores(eval_loader)

        if plot_dir and scores_mean.size:
            save_score_distribution_plot(
                scores_mean,
                plot_dir,
                filename="score_distribution.png",
                threshold=threshold,
                title="Window anomaly score distribution (training set)",
            )
            logger.info("Saved score distribution plot to %s/score_distribution.png", plot_dir)

            save_score_over_time_plot(
                scores_mean,
                scores_max,
                plot_dir,
                filename="score_over_time.png",
                threshold=threshold,
                title="Anomaly score over time (training set, window order)",
            )
            logger.info("Saved score-over-time plot to %s/score_over_time.png", plot_dir)
    except Exception as exc:
        logger.warning("Failed to generate score plots: %s", exc)

    # Per-channel MSE plot — shows which wires the model struggles to predict
    try:
        channel_mse = trainer.collect_channel_mse(eval_loader, num_channels=num_channels)
        if plot_dir:
            save_node_mse_plot(
                channel_mse.numpy(),
                plot_dir,
                filename="node_mse.png",
                title="Per-channel average prediction MSE (training set)",
            )
            logger.info("Saved per-channel MSE plot to %s/node_mse.png", plot_dir)
    except Exception as exc:
        logger.warning("Failed to generate per-channel MSE plot: %s", exc)

    logger.info("GNN training complete.")


if __name__ == "__main__":
    sys.exit(main())
