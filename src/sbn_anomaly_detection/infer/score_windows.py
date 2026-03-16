"""Score sliding windows with the Window autoencoder.

Outputs a CSV with columns: window_start, window_end, recon_error, anomaly_score.

Usage::

    sbn-score-windows --config configs/window_train.yaml \\
                      --checkpoint checkpoints/window_best.pt \\
                      --output scores_windows.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from sbn_anomaly_detection.data.dataset import EventDataset, WindowDataset
from sbn_anomaly_detection.models.model import (
    FusionAutoencoder,
    PMTAutoencoder,
    TPCAutoencoder,
    WindowAutoencoder,
)
from sbn_anomaly_detection.utils.checkpointing import load_checkpoint
from sbn_anomaly_detection.utils.logging import setup_logging
from sbn_anomaly_detection.utils.normalization import Normalizer

logger = logging.getLogger(__name__)


@torch.no_grad()
def score_windows(cfg: dict, window_ckpt: str, output_path: str) -> pd.DataFrame:
    """Score every window and return a DataFrame of anomaly scores.

    Parameters
    ----------
    cfg:
        Loaded YAML config dict.
    window_ckpt:
        Path to the window autoencoder checkpoint.
    output_path:
        Where to write the output CSV.

    Returns
    -------
    pd.DataFrame
        Columns: ``window_start``, ``window_end``, ``recon_error``, ``anomaly_score``.
    """
    setup_logging(level=logging.INFO)
    device = torch.device(cfg.get("device", "cpu"))
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    ckpt_dir = cfg["training"]["checkpoint_dir"]

    # Upstream normalisers + models
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
    load_checkpoint(tpc_model, str(Path(ckpt_dir) / "tpc_best.pt"), device=device)
    tpc_model.eval()

    pmt_model = PMTAutoencoder(
        input_dim=pmt_ds.n_features,
        hidden_dims=model_cfg.get("pmt_hidden_dims", [128, 64]),
        latent_dim=model_cfg.get("pmt_latent_dim", 32),
    ).to(device)
    load_checkpoint(pmt_model, str(Path(ckpt_dir) / "pmt_best.pt"), device=device)
    pmt_model.eval()

    fusion_model = FusionAutoencoder(
        tpc_latent_dim=model_cfg.get("tpc_latent_dim", 32),
        pmt_latent_dim=model_cfg.get("pmt_latent_dim", 32),
        hidden_dims=model_cfg.get("fusion_hidden_dims", [128, 64]),
        latent_dim=model_cfg.get("fusion_latent_dim", 64),
    ).to(device)
    load_checkpoint(fusion_model, str(Path(ckpt_dir) / "fusion_best.pt"), device=device)
    fusion_model.eval()

    # ------------------------------------------------------------------
    # Build fusion latents
    # ------------------------------------------------------------------
    batch_size = data_cfg.get("batch_size", 512)
    tpc_loader = DataLoader(tpc_ds, batch_size=batch_size, shuffle=False)
    pmt_loader = DataLoader(pmt_ds, batch_size=batch_size, shuffle=False)

    fusion_latents: list[torch.Tensor] = []
    for x_tpc, x_pmt in zip(tpc_loader, pmt_loader):
        x_tpc, x_pmt = x_tpc.to(device), x_pmt.to(device)
        _, z_tpc = tpc_model(x_tpc)
        _, z_pmt = pmt_model(x_pmt)
        z_fusion = fusion_model.encode(z_tpc, z_pmt)
        fusion_latents.append(z_fusion.cpu())

    fusion_arr = torch.cat(fusion_latents, dim=0).numpy()

    # ------------------------------------------------------------------
    # Window dataset + Window model
    # ------------------------------------------------------------------
    window_size = model_cfg.get("window_size", 64)
    stride = model_cfg.get("stride", 1)
    window_ds = WindowDataset(fusion_arr, window_size=window_size, stride=stride)

    window_model = WindowAutoencoder(
        n_features=fusion_arr.shape[1],
        window_size=window_size,
        channels=model_cfg.get("window_channels", [32, 64]),
        latent_dim=model_cfg.get("window_latent_dim", 128),
        kernel_size=model_cfg.get("kernel_size", 3),
    ).to(device)
    load_checkpoint(window_model, window_ckpt, device=device)
    window_model.eval()

    win_loader = DataLoader(window_ds, batch_size=data_cfg.get("batch_size", 128), shuffle=False)

    recon_errors: list[float] = []
    for batch in win_loader:
        batch = batch.to(device)
        recon, _ = window_model(batch)
        err = ((recon - batch) ** 2).mean(dim=(1, 2))
        recon_errors.extend(err.cpu().numpy().tolist())

    errors = np.array(recon_errors, dtype=np.float32)
    anomaly_score = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)

    starts = window_ds._indices
    ends = [s + window_size for s in starts]
    df = pd.DataFrame(
        {
            "window_start": starts,
            "window_end": ends,
            "recon_error": errors,
            "anomaly_score": anomaly_score,
        }
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Window scores written → %s  (%d windows)", out, len(df))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Score windows with Window autoencoder")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", required=True, help="Path to Window .pt checkpoint")
    parser.add_argument("--output", default="scores_windows.csv", help="Output CSV path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    score_windows(cfg, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
