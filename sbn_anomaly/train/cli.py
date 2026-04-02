"""Training CLI entry-point.

Usage::

    sbn-train --config configs/tpc.yaml
    sbn-train --config configs/tpc.yaml --root-files /data/run1.root /data/run2.root
    python -m sbn_anomaly.train.cli --config configs/pmt.yaml
"""

from __future__ import annotations

import argparse
import glob as _glob
import logging
import sys
from pathlib import Path

import yaml

from sbn_anomaly.utils.logging import setup_logging


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

    # Expand any glob patterns supplied via --root-files.
    root_files: list[str] | None = None
    if args.root_files:
        root_files = []
        for pattern in args.root_files:
            matches = _glob.glob(pattern)
            if matches:
                root_files.extend(matches)
            else:
                # Treat as a literal path; RootStreamer will raise if missing.
                root_files.append(pattern)

    model_type = cfg.get("model_type", "").lower()
    logger.info("Starting training for model_type='%s'", model_type)

    if model_type == "tpc":
        _train_tpc(cfg, root_files=root_files)
    elif model_type == "pmt":
        _train_pmt(cfg)
    elif model_type == "fusion":
        _train_fusion(cfg)
    elif model_type == "window":
        _train_window(cfg)
    else:
        logger.error(
            "Unknown model_type '%s'. Choose from: tpc, pmt, fusion, window.", model_type
        )
        return 1

    return 0


def _train_tpc(cfg: dict, root_files: list[str] | None = None) -> None:
    import numpy as np
    from torch.utils.data import DataLoader

    from sbn_anomaly.data.dataset import TPCDataset
    from sbn_anomaly.models.tpc_model import TPCAutoencoder
    from sbn_anomaly.train.tpc_trainer import TPCTrainer

    logger = logging.getLogger(__name__)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    input_dim = model_cfg.get("input_dim", 256)

    if root_files:
        from sbn_anomaly.data.stream_dataset import TPCStreamDataset

        logger.info(
            "Streaming TPC training from %d ROOT file(s): %s",
            len(root_files),
            root_files,
        )
        dataset = TPCStreamDataset(
            file_paths=root_files,
            tree_name=data_cfg.get("tree_name", "sbn_tree"),
            waveform_branch=data_cfg.get("waveform_branch", "tpc_waveform"),
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
        )
        steps_per_epoch = train_cfg.get(
            "steps_per_epoch",
            # Default of 100 batches per epoch is a sensible starting point for
            # streaming; tune via training.steps_per_epoch in the YAML config.
            100,
        )
    else:
        features = np.load(data_cfg["features_path"])
        dataset = TPCDataset(features)
        loader = DataLoader(
            dataset, batch_size=train_cfg.get("batch_size", 256), shuffle=True
        )
        steps_per_epoch = None

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
    )
    trainer.train(loader)
    if output := train_cfg.get("output_path"):
        trainer.save(output)
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

    features = np.load(data_cfg["features_path"])
    dataset = PMTDataset(features)
    loader = DataLoader(dataset, batch_size=train_cfg.get("batch_size", 256), shuffle=True)

    model = PMTAutoencoder(
        input_dim=model_cfg.get("input_dim", 128),
        latent_dim=model_cfg.get("latent_dim", 16),
    )
    trainer = PMTTrainer(
        model=model,
        lr=train_cfg.get("lr", 1e-3),
        max_epochs=train_cfg.get("max_epochs", 50),
        checkpoint_dir=train_cfg.get("checkpoint_dir"),
    )
    trainer.train(loader)
    if output := train_cfg.get("output_path"):
        trainer.save(output)
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

    tpc_features = np.load(data_cfg["tpc_features_path"])
    pmt_features = np.load(data_cfg["pmt_features_path"])
    dataset = FusionDataset(tpc_features, pmt_features)
    loader = DataLoader(dataset, batch_size=train_cfg.get("batch_size", 256), shuffle=True)

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
    )
    trainer.train(loader)
    if output := train_cfg.get("output_path"):
        trainer.save(output)
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

    signal = np.load(data_cfg["signal_path"])
    window_size = model_cfg.get("window_size", 256)
    dataset = WindowDataset(signal, window_size=window_size)
    loader = DataLoader(dataset, batch_size=train_cfg.get("batch_size", 256), shuffle=True)

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
    )
    trainer.train(loader)
    if output := train_cfg.get("output_path"):
        trainer.save(output)
    logging.getLogger(__name__).info("Window training complete.")


if __name__ == "__main__":
    sys.exit(main())
