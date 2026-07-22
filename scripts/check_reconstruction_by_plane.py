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

Note on --config vs --checkpoint: the model's architecture (encoder/decoder
widths, latent_dim, conv type) is inferred directly from the checkpoint's own
tensor shapes, not from --config's `model` section. This matters because
configs drift over training iterations (e.g. graph_vae.yaml may be on v7 while
you're checking a v5 checkpoint trained under a smaller model section) --
--config only needs to supply the matching `data` section (window/feature
settings) so the events npz is read the same way the checkpoint was trained.
A mismatch between --config's model section and the checkpoint is logged as a
warning, not an error.
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


def _infer_graph_vae_arch(state: dict, in_dim: int) -> dict:
    """Recover GraphVAE's constructor args from a checkpoint's tensor shapes.

    Checkpoints are self-describing: every architecture arg GraphVAE.__init__
    needs (encoder_hidden_dims, decoder_hidden_dims, latent_dim,
    use_channel_idx, conv) is recoverable from parameter shapes alone.
    Configs drift over time (e.g. this repo's configs/graph_vae.yaml has moved
    on to v7's wider encoder/decoder while an older v5 checkpoint trained
    under a smaller model section) -- trusting --config's model section to
    still match a given checkpoint causes exactly the load_state_dict shape
    mismatch this function exists to avoid.
    """
    is_sage = "convs.0.lin_l.weight" in state
    is_gcn = "convs.0.lin.weight" in state
    if not is_sage and not is_gcn:
        raise ValueError(
            "Could not detect conv type from checkpoint -- no convs.0.lin_l.weight "
            "(sage) or convs.0.lin.weight (gcn) key found. Is this really a GraphVAE checkpoint?"
        )
    conv = "sage" if is_sage else "gcn"
    w0_key = "convs.0.lin_l.weight" if is_sage else "convs.0.lin.weight"

    encoder_hidden_dims = []
    i = 0
    while True:
        key = f"convs.{i}.lin_l.weight" if is_sage else f"convs.{i}.lin.weight"
        if key not in state:
            break
        encoder_hidden_dims.append(int(state[key].shape[0]))
        i += 1

    enc_in = int(state[w0_key].shape[1])
    if enc_in == in_dim + 1:
        use_channel_idx = True
    elif enc_in == in_dim:
        use_channel_idx = False
    else:
        raise ValueError(
            f"Checkpoint's first conv layer expects {enc_in} input features, matching "
            f"neither in_dim={in_dim} nor in_dim+1={in_dim + 1} from --events. That's not "
            "just an architecture mismatch -- the node feature count doesn't match what "
            "this checkpoint trained on (wrong events file, node_features list, or "
            "channel count)."
        )

    latent_dim = int(state["fc_mu.weight"].shape[0])

    decoder_idxs = sorted(
        int(k.split(".")[1]) for k in state if k.startswith("decoder.") and k.endswith(".weight")
    )
    if not decoder_idxs:
        raise ValueError("Could not find any decoder.N.weight in checkpoint")
    decoder_out_dims = [int(state[f"decoder.{j}.weight"].shape[0]) for j in decoder_idxs]
    if decoder_out_dims[-1] != in_dim:
        raise ValueError(
            f"Checkpoint's final decoder layer outputs {decoder_out_dims[-1]} features, "
            f"not in_dim={in_dim} -- wrong events file/node_features for this checkpoint."
        )

    return dict(
        conv=conv,
        encoder_hidden_dims=encoder_hidden_dims,
        decoder_hidden_dims=decoder_out_dims[:-1],
        latent_dim=latent_dim,
        use_channel_idx=use_channel_idx,
    )


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

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    # Build the model from the CHECKPOINT's own tensor shapes rather than
    # trusting --config's model section -- configs drift (this repo is on v7
    # while an older checkpoint may have trained under a smaller model
    # section), and a stale --config here would otherwise surface as a
    # confusing load_state_dict shape-mismatch traceback instead of the real
    # cause.
    arch = _infer_graph_vae_arch(state, in_dim=dataset.node_feat_dim)
    cfg_arch = dict(
        conv=str(model_cfg.get("conv", "sage")),
        encoder_hidden_dims=[int(d) for d in model_cfg.get("encoder_hidden_dims", [64, 64])],
        decoder_hidden_dims=[int(d) for d in model_cfg.get("decoder_hidden_dims", [64])],
        latent_dim=int(model_cfg.get("latent_dim", 12)),
        use_channel_idx=bool(model_cfg.get("use_channel_idx", True)),
    )
    if cfg_arch != arch:
        logger.warning(
            "--config's model section doesn't match this checkpoint's actual "
            "architecture -- it was likely trained under a different config "
            "version. Building the model from the CHECKPOINT's shapes instead "
            "of --config so it loads correctly:\n"
            "  --config model section : %s\n"
            "  checkpoint architecture: %s",
            cfg_arch, arch,
        )
    else:
        logger.info("Checkpoint architecture matches --config's model section: %s", arch)

    model = GraphVAE(
        in_dim=dataset.node_feat_dim,
        latent_dim=arch["latent_dim"],
        encoder_hidden_dims=arch["encoder_hidden_dims"],
        decoder_hidden_dims=arch["decoder_hidden_dims"],
        dropout=float(model_cfg.get("dropout", 0.1)),  # not part of state_dict; irrelevant at eval anyway
        mask_ratio=0.0,  # no masking when evaluating
        use_channel_idx=arch["use_channel_idx"],
        conv=arch["conv"],
    )
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
