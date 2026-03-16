"""Train the Fusion autoencoder (Option A – standalone job).

Requires pre-trained TPC and PMT checkpoints to extract frozen latent codes.

Usage::

    sbn-train-fusion --config configs/fusion_train.yaml \\
                     --tpc-checkpoint checkpoints/tpc_best.pt \\
                     --pmt-checkpoint checkpoints/pmt_best.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset, random_split

from sbn_anomaly_detection.data.dataset import EventDataset
from sbn_anomaly_detection.models.model import FusionAutoencoder, PMTAutoencoder, TPCAutoencoder
from sbn_anomaly_detection.utils.checkpointing import load_checkpoint, save_checkpoint
from sbn_anomaly_detection.utils.logging import setup_logging
from sbn_anomaly_detection.utils.normalization import Normalizer

logger = logging.getLogger(__name__)


@torch.no_grad()
def _extract_latents(
    model: torch.nn.Module,
    dataset: EventDataset,
    device: torch.device,
    batch_size: int = 512,
) -> torch.Tensor:
    """Run encoder over the full dataset and return concatenated latent codes."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latents = []
    for batch in loader:
        batch = batch.to(device)
        _, z = model(batch)
        latents.append(z.cpu())
    return torch.cat(latents, dim=0)


def train(cfg: dict, tpc_ckpt: str, pmt_ckpt: str) -> None:
    setup_logging(level=logging.INFO, log_file=cfg["training"].get("log_file"))
    device = torch.device(cfg.get("device", "cpu"))
    logger.info("Training Fusion autoencoder on device: %s", device)

    data_cfg = cfg["data"]
    ckpt_dir = cfg["training"]["checkpoint_dir"]

    # ------------------------------------------------------------------
    # Load upstream models
    # ------------------------------------------------------------------
    tpc_norm = Normalizer.load(Path(ckpt_dir) / "tpc_normalizer.npz")
    pmt_norm = Normalizer.load(Path(ckpt_dir) / "pmt_normalizer.npz")

    tpc_ds = EventDataset(
        dict(
            root_files=data_cfg["root_files"],
            branches=data_cfg["tpc_branches"],
            tree_name=data_cfg.get("tree_name", "events"),
            step_size=data_cfg.get("step_size", 1000),
            max_events=data_cfg.get("max_events"),
        ),
        transform=tpc_norm.transform_tensor,
    )
    pmt_ds = EventDataset(
        dict(
            root_files=data_cfg["root_files"],
            branches=data_cfg["pmt_branches"],
            tree_name=data_cfg.get("tree_name", "events"),
            step_size=data_cfg.get("step_size", 1000),
            max_events=data_cfg.get("max_events"),
        ),
        transform=pmt_norm.transform_tensor,
    )

    model_cfg = cfg["model"]
    tpc_model = TPCAutoencoder(
        input_dim=tpc_ds.n_features,
        hidden_dims=model_cfg.get("tpc_hidden_dims", [256, 128, 64]),
        latent_dim=model_cfg.get("tpc_latent_dim", 32),
    ).to(device)
    load_checkpoint(tpc_model, tpc_ckpt, device=device)
    tpc_model.eval()

    pmt_model = PMTAutoencoder(
        input_dim=pmt_ds.n_features,
        hidden_dims=model_cfg.get("pmt_hidden_dims", [128, 64]),
        latent_dim=model_cfg.get("pmt_latent_dim", 32),
    ).to(device)
    load_checkpoint(pmt_model, pmt_ckpt, device=device)
    pmt_model.eval()

    # ------------------------------------------------------------------
    # Extract and align latents
    # ------------------------------------------------------------------
    logger.info("Extracting TPC latents …")
    z_tpc = _extract_latents(tpc_model, tpc_ds, device)
    logger.info("Extracting PMT latents …")
    z_pmt = _extract_latents(pmt_model, pmt_ds, device)

    # Align by index (assumes both datasets loaded events in the same order)
    min_n = min(len(z_tpc), len(z_pmt))
    z_tpc, z_pmt = z_tpc[:min_n], z_pmt[:min_n]

    fusion_dataset = TensorDataset(z_tpc, z_pmt)
    n_total = len(fusion_dataset)
    n_val = max(1, int(n_total * data_cfg.get("val_fraction", 0.1)))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(fusion_dataset, [n_train, n_val])

    batch_size = data_cfg.get("batch_size", 512)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # Fusion model
    # ------------------------------------------------------------------
    fusion_model = FusionAutoencoder(
        tpc_latent_dim=z_tpc.shape[1],
        pmt_latent_dim=z_pmt.shape[1],
        hidden_dims=model_cfg.get("fusion_hidden_dims", [128, 64]),
        latent_dim=model_cfg.get("fusion_latent_dim", 64),
    ).to(device)

    train_cfg = cfg["training"]
    optimizer = torch.optim.Adam(
        fusion_model.parameters(),
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    epochs = train_cfg.get("epochs", 50)

    for epoch in range(epochs):
        fusion_model.train()
        train_loss = 0.0
        for zt, zp in train_loader:
            zt, zp = zt.to(device), zp.to(device)
            loss = fusion_model.reconstruction_loss(zt, zp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(zt)
        train_loss /= n_train

        fusion_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for zt, zp in val_loader:
                zt, zp = zt.to(device), zp.to(device)
                val_loss += fusion_model.reconstruction_loss(zt, zp).item() * len(zt)
        val_loss /= n_val
        scheduler.step(val_loss)

        logger.info(
            "Epoch %d/%d  train_loss=%.6f  val_loss=%.6f",
            epoch + 1, epochs, train_loss, val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                fusion_model, optimizer, epoch, val_loss, ckpt_dir, "fusion_best.pt"
            )

    save_checkpoint(
        fusion_model, optimizer, epochs - 1, val_loss, ckpt_dir, "fusion_last.pt"
    )
    logger.info("Fusion training complete. Best val loss: %.6f", best_val_loss)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Fusion autoencoder")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--tpc-checkpoint", required=True, help="Path to TPC .pt checkpoint")
    parser.add_argument("--pmt-checkpoint", required=True, help="Path to PMT .pt checkpoint")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, args.tpc_checkpoint, args.pmt_checkpoint)


if __name__ == "__main__":
    main()
