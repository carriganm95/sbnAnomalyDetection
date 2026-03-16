"""Score individual events with the Fusion autoencoder.

Outputs a CSV with columns: run, event, reconstruction_error, anomaly_score.

Usage::

    sbn-score-events --config configs/fusion_train.yaml \\
                     --checkpoint checkpoints/fusion_best.pt \\
                     --output scores_events.csv
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

from sbn_anomaly_detection.data.dataset import EventDataset
from sbn_anomaly_detection.models.model import FusionAutoencoder, PMTAutoencoder, TPCAutoencoder
from sbn_anomaly_detection.utils.checkpointing import load_checkpoint
from sbn_anomaly_detection.utils.logging import setup_logging
from sbn_anomaly_detection.utils.normalization import Normalizer

logger = logging.getLogger(__name__)


@torch.no_grad()
def score_events(cfg: dict, fusion_ckpt: str, output_path: str) -> pd.DataFrame:
    """Score every event and return a DataFrame of anomaly scores.

    Parameters
    ----------
    cfg:
        Loaded YAML config dict.
    fusion_ckpt:
        Path to the fusion autoencoder checkpoint.
    output_path:
        Where to write the output CSV.

    Returns
    -------
    pd.DataFrame
        Columns: ``run``, ``event``, ``recon_error``, ``anomaly_score``.
    """
    setup_logging(level=logging.INFO)
    device = torch.device(cfg.get("device", "cpu"))
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    ckpt_dir = cfg["training"]["checkpoint_dir"]

    # Load normalisers
    tpc_norm = Normalizer.load(Path(ckpt_dir) / "tpc_normalizer.npz")
    pmt_norm = Normalizer.load(Path(ckpt_dir) / "pmt_normalizer.npz")

    # Datasets
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

    # Models
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
    load_checkpoint(fusion_model, fusion_ckpt, device=device)
    fusion_model.eval()

    batch_size = data_cfg.get("batch_size", 512)
    tpc_loader = DataLoader(tpc_ds, batch_size=batch_size, shuffle=False)
    pmt_loader = DataLoader(pmt_ds, batch_size=batch_size, shuffle=False)

    recon_errors: list[float] = []
    for x_tpc, x_pmt in zip(tpc_loader, pmt_loader):
        x_tpc, x_pmt = x_tpc.to(device), x_pmt.to(device)
        _, z_tpc = tpc_model(x_tpc)
        _, z_pmt = pmt_model(x_pmt)
        recon, _, target = fusion_model(z_tpc, z_pmt)
        err = ((recon - target) ** 2).mean(dim=-1)
        recon_errors.extend(err.cpu().numpy().tolist())

    n = len(recon_errors)
    errors = np.array(recon_errors, dtype=np.float32)
    # Normalise to [0, 1] as anomaly score
    anomaly_score = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)

    df = pd.DataFrame(
        {
            "event_idx": np.arange(n),
            "recon_error": errors,
            "anomaly_score": anomaly_score,
        }
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Scores written → %s  (%d events)", out, n)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Score events with Fusion autoencoder")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", required=True, help="Path to Fusion .pt checkpoint")
    parser.add_argument("--output", default="scores_events.csv", help="Output CSV path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    score_events(cfg, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
