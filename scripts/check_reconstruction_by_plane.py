#!/usr/bin/env python3
"""Check whether graph_vae reconstructs one plane worse than others on GOOD data.

`GraphVAETrainer.compute_loss` is `((x_hat - y) ** 2).mean()` -- an unweighted
mean over every active node/feature in the batch. If one plane has fewer
active channels per window than another (plausible: SBND collection is a
minority of total channels vs. the two induction planes), its share of the
total gradient signal is smaller purely from channel count, independent of
anything wrong with its data -- the model could end up under-fitting that
plane simply because it contributes less to the loss during training.

This script measures the observable symptom directly rather than guessing:
per-channel reconstruction MSE on good data, averaged over many windows,
grouped by plane, using the trained model as-is (no retraining). It reuses
`GraphVAETrainer.collect_channel_mse` -- the same accumulation the training
loop could log per epoch -- rather than reimplementing the forward pass.

How to read the result
-----------------------
- If per-plane mean MSE is comparable across planes, the unweighted loss
  isn't the bottleneck -- collection is being fit about as well as induction,
  and whatever compresses good/bad separation there is downstream of
  training (scoring/aggregation, or the message-passing effect discussed
  separately).
- If one plane's mean MSE is persistently higher than the others' *on good
  data*, that's under-fitting from the loss/channel-count imbalance, and the
  fix is a plane-balanced loss (mean of per-plane means instead of one global
  mean) -- not more features or a bigger model.

This requires the same environment as training (torch + torch_geometric),
unlike compare_events_distributions.py.

Usage
-----
    python scripts/check_reconstruction_by_plane.py \\
        --config configs/graph_vae.yaml \\
        --checkpoint checkpoints/graph_vae/v7/graph_vae_final.pt \\
        --events data/good_events_val.npz

Prefer a held-out validation events npz if you have one -- scoring against
data the model trained on will underestimate under-fitting on every plane
uniformly, but a systematic *per-plane gap* should still show up either way.
Plane comes from `planes_flat` in the events npz if present, otherwise
--channel-map (defaults to data.channel_map in --config).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Pure-numpy helpers shared with the other diagnostic scripts (no torch needed
# for these two functions specifically).
from compare_events_distributions import _channel_plane_lookup, _channel_to_plane_lut  # noqa: E402

logger = logging.getLogger("check_reconstruction_by_plane")


def run(
    config_path: str,
    checkpoint_path: str,
    events_path: str,
    channel_map: Optional[str],
    batch_size: int,
    min_plane_samples: int,
    output_csv: Optional[str],
) -> None:
    import yaml
    import torch
    from torch_geometric.loader import DataLoader as PyGDataLoader

    from sbn_anomaly.data.sparse_window_dataset import SparseWindowDatasetPyG
    from sbn_anomaly.models.graph_vae import GraphVAE
    from sbn_anomaly.train.graph_vae_trainer import GraphVAETrainer

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    channel_map = channel_map or data_cfg.get("channel_map")

    # Reuse training's exact standardization (saved next to the checkpoint) so
    # the reconstruction MSE we measure reflects what the model actually
    # trained against, not a freshly-refit approximation of it.
    std_path = str(Path(checkpoint_path).parent / "standardization.npz")
    feat_mean = feat_std = None
    if Path(std_path).exists():
        s = np.load(std_path)
        feat_mean, feat_std = s["feature_mean"], s["feature_std"]
        logger.info("Loaded standardization from %s", std_path)
    else:
        logger.warning(
            "No standardization.npz next to checkpoint (%s) -- refitting from "
            "--events instead. This may not exactly match what the model "
            "trained against; results are still indicative but less precise.",
            std_path)

    logger.info("Loading events from %s ...", events_path)
    dataset = SparseWindowDatasetPyG.from_npz(
        events_path,
        history=1,  # unused in reconstruction mode
        window_size=int(data_cfg.get("window_size", 20)),
        n_bins=int(data_cfg.get("n_temporal_bins", 4)),
        stride=int(data_cfg.get("stride", 1)),
        radius=int(data_cfg.get("adjacency_radius", 4)),
        node_features=data_cfg.get("node_features") or None,
        prune_inactive=bool(data_cfg.get("prune_inactive", True)),
        channel_map=data_cfg.get("channel_map"),
        edge_mode=str(data_cfg.get("edge_mode", "sequential")),
        reconstruction=True,
        standardize=bool(data_cfg.get("standardize", True)),
        standardize_by=str(data_cfg.get("standardize_by", "global")),
        min_plane_samples=int(data_cfg.get("min_plane_samples", 20)),
        feature_mean=feat_mean, feature_std=feat_std,
        log_features=data_cfg.get("log_features") or None,
        n_channels=data_cfg.get("n_channels"),
    )
    logger.info("Dataset: %d channels, %d windows", dataset.num_nodes, len(dataset))

    model = GraphVAE(
        in_dim=dataset.node_feat_dim,
        latent_dim=int(model_cfg.get("latent_dim", 12)),
        encoder_hidden_dims=model_cfg.get("encoder_hidden_dims", [128, 64]),
        decoder_hidden_dims=model_cfg.get("decoder_hidden_dims", [32, 64]),
        dropout=float(model_cfg.get("dropout", 0.1)),
        mask_ratio=0.0,  # no masking when evaluating
        use_channel_idx=bool(model_cfg.get("use_channel_idx", True)),
        conv=str(model_cfg.get("conv", "sage")),
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)

    # GraphVAETrainer moves the model to device internally; we only use it
    # for collect_channel_mse (no optimizer step ever happens).
    trainer = GraphVAETrainer(model=model, device="auto")

    loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=False)
    logger.info("Running forward pass over %d windows (batch_size=%d) ...", len(dataset), batch_size)
    channel_mse = trainer.collect_channel_mse(loader, num_channels=dataset.num_nodes).numpy()

    # Plane lookup, same helper pattern as the other diagnostic scripts.
    if dataset._planes_flat is not None:
        channel_plane = _channel_to_plane_lut(dataset._channels_flat, dataset._planes_flat, dataset.num_nodes)
    elif channel_map:
        channel_plane = _channel_plane_lookup(channel_map, dataset.num_nodes)
    else:
        raise ValueError(
            "No plane info: events npz has no 'planes_flat' and no --channel-map "
            "given (and none in --config data.channel_map).")

    valid = ~np.isnan(channel_mse)
    if not valid.any():
        raise ValueError("No channel ever active across all windows -- nothing to report.")
    overall = channel_mse[valid]
    print(f"\nAll channels: n={overall.size}  mean_mse={overall.mean():.5g}  "
          f"median_mse={np.median(overall):.5g}  std={overall.std():.5g}")

    planes = sorted(int(p) for p in set(channel_plane.tolist()) if p >= 0)
    print(f"\n{'plane':<8}{'n_channels':>12}{'mean_mse':>14}{'median_mse':>14}{'std_mse':>12}")
    rows_report = []
    plane_means: dict = {}
    for p in planes:
        mask = (channel_plane == p) & valid
        vals = channel_mse[mask]
        if vals.size < min_plane_samples:
            print(f"{p:<8}{vals.size:>12}  (< --min-plane-samples={min_plane_samples} -- skipping)")
            continue
        mean_mse, median_mse, std_mse = vals.mean(), np.median(vals), vals.std()
        plane_means[p] = mean_mse
        print(f"{p:<8}{vals.size:>12}{mean_mse:>14.5g}{median_mse:>14.5g}{std_mse:>12.5g}")
        rows_report.append({
            "plane": p, "n_channels": int(vals.size),
            "mean_mse": mean_mse, "median_mse": median_mse, "std_mse": std_mse,
        })

    if len(plane_means) >= 2:
        worst = max(plane_means, key=plane_means.get)
        best = min(plane_means, key=plane_means.get)
        ratio = plane_means[worst] / max(plane_means[best], 1e-12)
        print(f"\nWorst/best plane MSE ratio: plane {worst} is {ratio:.2f}x plane {best}'s "
              f"reconstruction error on this (good) data.")
        if ratio > 1.5:
            print("This looks like a real per-plane fit imbalance on GOOD data -- "
                  "consider a plane-balanced loss (mean of per-plane means) rather than "
                  "adding more features/capacity.")
        else:
            print("Planes are reconstructed comparably on good data -- the unweighted "
                  "loss doesn't look like the bottleneck here.")
    else:
        print("\nFewer than 2 planes had enough channels to compare -- no verdict.")

    if output_csv:
        import pandas as pd
        pd.DataFrame(rows_report).to_csv(output_csv, index=False)
        logger.info("Wrote per-plane MSE table to %s", output_csv)


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="e.g. configs/graph_vae.yaml")
    ap.add_argument("--checkpoint", required=True, help="Trained model checkpoint (.pt)")
    ap.add_argument("--events", required=True,
                     help="Events .npz to evaluate on (ideally held-out good-run data)")
    ap.add_argument("--channel-map", default=None,
                     help="Override/fallback for plane lookup if the events npz lacks planes_flat "
                          "(default: data.channel_map from --config)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--min-plane-samples", type=int, default=20,
                     help="Skip a plane's row in the report if fewer channels than this had data")
    ap.add_argument("--output-csv", default=None, help="Optional path to dump the per-plane MSE table")
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(
        config_path=args.config, checkpoint_path=args.checkpoint, events_path=args.events,
        channel_map=args.channel_map, batch_size=args.batch_size,
        min_plane_samples=args.min_plane_samples, output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
