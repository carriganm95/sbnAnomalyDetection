"""Train the TPC autoencoder (Option A – standalone job).

Usage::

    python -m sbn_anomaly_detection.train.train_tpc --config configs/tpc_train.yaml
    # or after pip install -e .:
    sbn-train-tpc --config configs/tpc_train.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, random_split

from sbn_anomaly_detection.data.dataset import EventDataset
from sbn_anomaly_detection.models.model import TPCAutoencoder
from sbn_anomaly_detection.utils.checkpointing import save_checkpoint
from sbn_anomaly_detection.utils.logging import setup_logging
from sbn_anomaly_detection.utils.normalization import Normalizer

logger = logging.getLogger(__name__)


def train(cfg: dict) -> None:
    setup_logging(level=logging.INFO, log_file=cfg["training"].get("log_file"))
    device = torch.device(cfg.get("device", "cpu"))
    logger.info("Training TPC autoencoder on device: %s", device)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_cfg = cfg["data"]
    dataset_kwargs = dict(
        root_files=data_cfg["root_files"],
        branches=data_cfg["tpc_branches"],
        tree_name=data_cfg.get("tree_name", "events"),
        step_size=data_cfg.get("step_size", 1000),
        max_events=data_cfg.get("max_events"),
    )
    full_dataset = EventDataset(dataset_kwargs)
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * data_cfg.get("val_fraction", 0.1)))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    # Fit normaliser on the raw training data
    train_arr = full_dataset._data[list(train_ds.indices)]
    normalizer = Normalizer.fit(train_arr)
    norm_path = Path(cfg["training"]["checkpoint_dir"]) / "tpc_normalizer.npz"
    normalizer.save(norm_path)

    # Wrap dataset with normalisation transform
    full_dataset.transform = normalizer.transform_tensor

    batch_size = data_cfg.get("batch_size", 512)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_cfg = cfg["model"]
    model = TPCAutoencoder(
        input_dim=full_dataset.n_features,
        hidden_dims=model_cfg.get("hidden_dims", [256, 128, 64]),
        latent_dim=model_cfg.get("latent_dim", 32),
    ).to(device)
    logger.info("Model: %s", model)

    train_cfg = cfg["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    ckpt_dir = train_cfg["checkpoint_dir"]
    epochs = train_cfg.get("epochs", 50)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            loss = model.reconstruction_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                val_loss += model.reconstruction_loss(batch).item() * len(batch)
        val_loss /= n_val
        scheduler.step(val_loss)

        logger.info(
            "Epoch %d/%d  train_loss=%.6f  val_loss=%.6f",
            epoch + 1, epochs, train_loss, val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, ckpt_dir, "tpc_best.pt")

    save_checkpoint(model, optimizer, epochs - 1, val_loss, ckpt_dir, "tpc_last.pt")
    logger.info("TPC training complete. Best val loss: %.6f", best_val_loss)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TPC autoencoder")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)


if __name__ == "__main__":
    main()
