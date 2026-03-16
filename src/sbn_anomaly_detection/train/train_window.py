"""Train the Window autoencoder (Option A – standalone job).

Requires a pre-trained Fusion checkpoint so that the window model operates
on fusion-level latent codes assembled into temporal windows.

Usage::

    sbn-train-window --config configs/window_train.yaml \\
                     --fusion-checkpoint checkpoints/fusion_best.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, random_split

from sbn_anomaly_detection.data.dataset import EventDataset, WindowDataset
from sbn_anomaly_detection.models.model import (
    FusionAutoencoder,
    PMTAutoencoder,
    TPCAutoencoder,
    WindowAutoencoder,
)
from sbn_anomaly_detection.utils.checkpointing import load_checkpoint, save_checkpoint
from sbn_anomaly_detection.utils.logging import setup_logging
from sbn_anomaly_detection.utils.normalization import Normalizer

logger = logging.getLogger(__name__)


@torch.no_grad()
def _build_fusion_latents_array(
    tpc_model: TPCAutoencoder,
    pmt_model: PMTAutoencoder,
    fusion_model: FusionAutoencoder,
    tpc_ds: EventDataset,
    pmt_ds: EventDataset,
    device: torch.device,
    batch_size: int = 512,
) -> torch.Tensor:
    """Return fusion latent codes for all aligned events as a 2-D tensor."""
    tpc_loader = DataLoader(tpc_ds, batch_size=batch_size, shuffle=False)
    pmt_loader = DataLoader(pmt_ds, batch_size=batch_size, shuffle=False)

    latents = []
    for x_tpc, x_pmt in zip(tpc_loader, pmt_loader):
        x_tpc, x_pmt = x_tpc.to(device), x_pmt.to(device)
        _, z_tpc = tpc_model(x_tpc)
        _, z_pmt = pmt_model(x_pmt)
        z_fusion = fusion_model.encode(z_tpc, z_pmt)
        latents.append(z_fusion.cpu())
    return torch.cat(latents, dim=0)


def train(cfg: dict, fusion_ckpt: str) -> None:
    setup_logging(level=logging.INFO, log_file=cfg["training"].get("log_file"))
    device = torch.device(cfg.get("device", "cpu"))
    logger.info("Training Window autoencoder on device: %s", device)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    ckpt_dir = train_cfg["checkpoint_dir"]

    # ------------------------------------------------------------------
    # Load upstream models (frozen)
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

    tpc_model = TPCAutoencoder(
        input_dim=tpc_ds.n_features,
        hidden_dims=model_cfg.get("tpc_hidden_dims", [256, 128, 64]),
        latent_dim=model_cfg.get("tpc_latent_dim", 32),
    ).to(device)
    load_checkpoint(
        tpc_model,
        str(Path(ckpt_dir) / "tpc_best.pt"),
        device=device,
    )
    tpc_model.eval()

    pmt_model = PMTAutoencoder(
        input_dim=pmt_ds.n_features,
        hidden_dims=model_cfg.get("pmt_hidden_dims", [128, 64]),
        latent_dim=model_cfg.get("pmt_latent_dim", 32),
    ).to(device)
    load_checkpoint(
        pmt_model,
        str(Path(ckpt_dir) / "pmt_best.pt"),
        device=device,
    )
    pmt_model.eval()

    fusion_model = FusionAutoencoder(
        tpc_latent_dim=model_cfg.get("tpc_latent_dim", 32),
        pmt_latent_dim=model_cfg.get("pmt_latent_dim", 32),
        hidden_dims=model_cfg.get("fusion_hidden_dims", [128, 64]),
        latent_dim=model_cfg.get("fusion_latent_dim", 64),
    ).to(device)
    load_checkpoint(fusion_model, fusion_ckpt, device=device)
    fusion_model.eval()

    # ------------------------------------------------------------------
    # Build fusion latents → WindowDataset
    # ------------------------------------------------------------------
    logger.info("Building fusion latents for window training …")
    with torch.no_grad():
        fusion_latents = _build_fusion_latents_array(
            tpc_model, pmt_model, fusion_model, tpc_ds, pmt_ds, device
        ).numpy()

    window_size = model_cfg.get("window_size", 64)
    stride = model_cfg.get("stride", 1)
    window_ds = WindowDataset(fusion_latents, window_size=window_size, stride=stride)

    n_total = len(window_ds)
    n_val = max(1, int(n_total * data_cfg.get("val_fraction", 0.1)))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(window_ds, [n_train, n_val])

    batch_size = data_cfg.get("batch_size", 128)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # Window model
    # ------------------------------------------------------------------
    n_features = fusion_latents.shape[1]
    window_model = WindowAutoencoder(
        n_features=n_features,
        window_size=window_size,
        channels=model_cfg.get("window_channels", [32, 64]),
        latent_dim=model_cfg.get("window_latent_dim", 128),
        kernel_size=model_cfg.get("kernel_size", 3),
    ).to(device)

    optimizer = torch.optim.Adam(
        window_model.parameters(),
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
        window_model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            loss = window_model.reconstruction_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)
        train_loss /= n_train

        window_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                val_loss += window_model.reconstruction_loss(batch).item() * len(batch)
        val_loss /= n_val
        scheduler.step(val_loss)

        logger.info(
            "Epoch %d/%d  train_loss=%.6f  val_loss=%.6f",
            epoch + 1, epochs, train_loss, val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                window_model, optimizer, epoch, val_loss, ckpt_dir, "window_best.pt"
            )

    save_checkpoint(
        window_model, optimizer, epochs - 1, val_loss, ckpt_dir, "window_last.pt"
    )
    logger.info("Window training complete. Best val loss: %.6f", best_val_loss)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Window autoencoder")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--fusion-checkpoint", required=True, help="Path to Fusion .pt checkpoint"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, args.fusion_checkpoint)


if __name__ == "__main__":
    main()
