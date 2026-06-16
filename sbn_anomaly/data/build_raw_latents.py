"""Encode raw ADC waveforms into per-channel VAE latents for the GNN forecaster.

This is the bridge from the trained waveform VAE to the existing graph
forecasting stack.  For every event it produces one latent vector per channel
and stacks them into a dense windows array of shape::

    (num_events, num_channels, latent_dim)

which is exactly the input contract of
:class:`~sbn_anomaly.data.graph_window_dataset_pyg.GraphWindowDatasetPyG`.  The
GNN then treats each *event* as one temporal frame and forecasts the next
event's per-channel latents — reusing ``GNNForecasterPyG`` + GRU unchanged.

Channels are placed at a fixed row given by ``channel_to_node`` (default:
identity up to ``num_channels``), so node indices are stable across events even
when some channels are missing in a given event.

CLI::

    python -m sbn_anomaly.data.build_raw_latents \
        --config configs/raw_gnn.yaml \
        --vae-checkpoint checkpoints/raw_vae/vae_final.pt \
        --root-files raw_run10376.root \
        --output data/raw_latents_run10376.npz
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from sbn_anomaly.data.raw_digit_reader import RawDigitReader
from sbn_anomaly.data.raw_preprocess import preprocess_event
from sbn_anomaly.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def encode_events_to_latents(
    root_files: Sequence[str],
    vae,
    num_channels: int,
    input_length: int,
    *,
    device: str = "cpu",
    tree_name: str = "rawdigits",
    max_events: Optional[int] = None,
    encode_batch: int = 512,
    preprocess_kwargs: Optional[dict] = None,
    channel_to_node: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode every event's channels into latents.

    Returns
    -------
    latents:
        ``(num_events, num_channels, latent_dim)`` float32. Missing channels
        are left as zeros.
    provenance:
        ``(num_events, 3)`` int32 array of ``(run, subrun, event)``.
    """
    import torch

    pp = dict(preprocess_kwargs or {})
    pp.setdefault("n_ticks", input_length)
    latent_dim = int(vae.latent_dim)
    vae = vae.to(device).eval()

    reader = RawDigitReader(root_files, tree_name=tree_name, max_events=max_events)

    event_latents: list[np.ndarray] = []
    provenance: list[tuple[int, int, int]] = []

    for ev in reader:
        wf = preprocess_event(ev.adc, ev.pedestal, **pp)  # (nchan, input_length)
        # Map detector channel id -> fixed node row.
        if channel_to_node is None:
            node_rows = ev.channel.astype(np.int64)
        else:
            node_rows = channel_to_node[ev.channel.astype(np.int64)]
        valid = (node_rows >= 0) & (node_rows < num_channels)
        wf = wf[valid]
        node_rows = node_rows[valid]

        # Encode in mini-batches on the chosen device.
        lat = np.zeros((num_channels, latent_dim), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, wf.shape[0], encode_batch):
                chunk = wf[start:start + encode_batch]
                x = torch.from_numpy(chunk).float().to(device)
                z = vae.encode_latents(x).cpu().numpy()
                lat[node_rows[start:start + encode_batch]] = z
        event_latents.append(lat)
        provenance.append((ev.run, ev.subrun, ev.event))

    if not event_latents:
        logger.warning("No events encoded; producing empty arrays.")
        return (
            np.zeros((0, num_channels, latent_dim), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int32),
        )

    latents = np.stack(event_latents, axis=0).astype(np.float32)
    prov = np.asarray(provenance, dtype=np.int32)
    logger.info(
        "Encoded %d events -> latents %s", latents.shape[0], latents.shape
    )
    return latents, prov


def _load_vae_from_checkpoint(checkpoint: str, model_cfg: dict):
    import torch

    from sbn_anomaly.models.tpc_waveform_vae import TPCWaveformVAE

    vae = TPCWaveformVAE(
        input_length=int(model_cfg.get("input_length", 4096)),
        latent_dim=int(model_cfg.get("latent_dim", 24)),
        base_channels=int(model_cfg.get("base_channels", 16)),
        depth=int(model_cfg.get("depth", 4)),
        kernel_size=int(model_cfg.get("kernel_size", 7)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
    state = torch.load(checkpoint, map_location="cpu")
    # Accept either a raw state_dict or a {'model_state_dict': ...} checkpoint.
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    vae.load_state_dict(state)
    return vae


def main(argv: list[str] | None = None) -> int:
    import yaml

    parser = argparse.ArgumentParser(
        description="Encode raw ADC waveforms into per-channel VAE latents for the GNN."
    )
    parser.add_argument("--config", required=True, help="raw_gnn / raw_vae YAML config.")
    parser.add_argument("--vae-checkpoint", required=True, help="Trained VAE checkpoint (.pt).")
    parser.add_argument("--root-files", nargs="+", required=True, help="Flat raw-ADC ntuple(s).")
    parser.add_argument("--output", required=True, help="Output .npz (key 'windows').")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    setup_logging(args.log_level)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    num_channels = int(data_cfg.get("n_channels", 11264))
    input_length = int(model_cfg.get("input_length", 4096))
    pp = data_cfg.get("preprocess", {}) or {}

    vae = _load_vae_from_checkpoint(args.vae_checkpoint, model_cfg)

    latents, prov = encode_events_to_latents(
        root_files=args.root_files,
        vae=vae,
        num_channels=num_channels,
        input_length=input_length,
        device=args.device,
        tree_name=data_cfg.get("raw_tree_name", "rawdigits"),
        max_events=args.max_events,
        preprocess_kwargs=pp,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, windows=latents, provenance=prov)
    logger.info("Saved latents to %s (windows=%s)", out, latents.shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
