"""Materialise preprocessed per-channel waveforms to an .npz for VAE training.

Streaming from ROOT every epoch re-reads and re-preprocesses the raw ADC each
time. This script does it once and writes a compact array that
:class:`~sbn_anomaly.data.raw_waveform_dataset.RawWaveformArrayDataset` (and
hence ``sbn-train --config configs/raw_vae.yaml`` via ``data.waveforms_path``)
loads directly.

Output ``.npz`` keys:
    waveforms  : (N, input_length) float32  -- preprocessed, one row per channel
    channel    : (N,) int32                 -- detector channel id of each row
    event_idx  : (N,) int32                 -- index of the source event
    provenance : (n_events, 3) int32        -- (run, subrun, event) per event

Only ``waveforms`` is required by the dataset; the rest is provenance you can
use for plotting or bookkeeping.

CLI::

    python -m sbn_anomaly.data.build_raw_waveforms \
        --config configs/raw_vae.yaml \
        --root-files raw_run*.root \
        --output data/raw_waveforms_train.npz \
        --max-waveforms 2000000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from sbn_anomaly.data.raw_digit_reader import RawDigitReader
from sbn_anomaly.data.raw_preprocess import preprocess_event
from sbn_anomaly.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def materialize_waveforms(
    root_files: Optional[Sequence[str]] = None,
    input_length: int = 4096,
    *,
    events: Optional[Iterable] = None,
    tree_name: str = "rawdigits",
    max_events: Optional[int] = None,
    channels_per_event: Optional[int] = None,
    max_waveforms: Optional[int] = None,
    preprocess_kwargs: Optional[dict] = None,
    coherent_groups_path: Optional[str] = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Stream events, preprocess, and stack waveforms into dense arrays.

    Provide either ``root_files`` (read via the flat-ntuple :class:`RawDigitReader`)
    or ``events`` (any iterable of objects with ``.run/.subrun/.event/.channel/
    .pedestal/.adc`` -- e.g. a gallery-based reader on a gpvm). ``events`` takes
    precedence and is assumed to already honour its own event cap.

    Parameters mirror the streaming dataset. ``channels_per_event`` randomly
    subsamples channels per event (balances events, bounds size);
    ``max_waveforms`` caps the total number of rows collected.
    """
    pp = dict(preprocess_kwargs or {})
    pp.setdefault("n_ticks", input_length)
    rng = np.random.default_rng(seed)

    # Optional electronics-aware coherent grouping: a channel-id-indexed array
    # mapping each detector channel to a coherent-noise group (e.g. FEMB id).
    chan_to_group = None
    if coherent_groups_path:
        chan_to_group = np.load(coherent_groups_path)
        logger.info("Loaded coherent-noise groups from %s (%d channels)",
                    coherent_groups_path, chan_to_group.shape[0])

    if events is None:
        if root_files is None:
            raise ValueError("Provide either root_files or events.")
        events = RawDigitReader(root_files, tree_name=tree_name, max_events=max_events)

    wf_chunks: list[np.ndarray] = []
    chan_chunks: list[np.ndarray] = []
    evt_chunks: list[np.ndarray] = []
    provenance: list[tuple] = []
    total = 0

    for ev_idx, ev in enumerate(events):
        provenance.append((ev.run, ev.subrun, ev.event))
        ev_pp = pp
        if chan_to_group is not None:
            # Map this event's channels (in digit order) to their groups.
            ev_pp = {**pp, "coherent_groups": chan_to_group[ev.channel.astype(np.int64)]}
        wf = preprocess_event(ev.adc, ev.pedestal, **ev_pp)  # (nchan, input_length)
        n_chan = wf.shape[0]
        if channels_per_event is not None and channels_per_event < n_chan:
            sel = rng.choice(n_chan, size=channels_per_event, replace=False)
        else:
            sel = np.arange(n_chan)

        if max_waveforms is not None and total + sel.size > max_waveforms:
            sel = sel[: max(0, max_waveforms - total)]
        if sel.size == 0:
            if max_waveforms is not None and total >= max_waveforms:
                break
            continue

        wf_chunks.append(wf[sel].astype(np.float32))
        chan_chunks.append(ev.channel[sel].astype(np.int32))
        evt_chunks.append(np.full(sel.size, ev_idx, dtype=np.int32))
        total += sel.size
        if max_waveforms is not None and total >= max_waveforms:
            logger.info("Reached max_waveforms=%d; stopping.", max_waveforms)
            break

    if not wf_chunks:
        logger.warning("No waveforms collected.")
        return {
            "waveforms": np.zeros((0, input_length), dtype=np.float32),
            "channel": np.zeros(0, dtype=np.int32),
            "event_idx": np.zeros(0, dtype=np.int32),
            "provenance": np.zeros((0, 3), dtype=np.int32),
        }

    out = {
        "waveforms": np.concatenate(wf_chunks, axis=0),
        "channel": np.concatenate(chan_chunks, axis=0),
        "event_idx": np.concatenate(evt_chunks, axis=0),
        "provenance": np.asarray(provenance, dtype=np.int32),
    }
    logger.info(
        "Materialised %d waveforms (length %d) from %d event(s)",
        out["waveforms"].shape[0], input_length, out["provenance"].shape[0],
    )
    return out


def main(argv: list[str] | None = None) -> int:
    import yaml

    parser = argparse.ArgumentParser(
        description="Materialise preprocessed raw waveforms to an .npz for VAE training."
    )
    parser.add_argument("--config", required=True, help="raw_vae YAML config.")
    parser.add_argument("--root-files", nargs="+", default=None,
                        help="Flat raw-ADC ntuple path(s) or glob(s).")
    parser.add_argument("--root-file-list", nargs="+", default=None, metavar="FILE",
                        help="Manifest file(s), one ROOT path per line (# comments ok).")
    parser.add_argument("--output", required=True, help="Output .npz (key 'waveforms').")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--max-waveforms", type=int, default=None,
                        help="Cap total waveforms collected (bounds file size).")
    parser.add_argument("--compress", action="store_true",
                        help="Write compressed npz (smaller, slower to load).")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    setup_logging(args.log_level)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    from sbn_anomaly.data.root_files import resolve_root_files
    inputs = list(args.root_files or []) + list(args.root_file_list or [])
    if not inputs:
        parser.error("provide --root-files and/or --root-file-list")
    root_files = resolve_root_files(inputs)

    input_length = int(model_cfg.get("input_length", 4096))
    out = materialize_waveforms(
        root_files=root_files,
        input_length=input_length,
        tree_name=data_cfg.get("raw_tree_name", "rawdigits"),
        max_events=args.max_events,
        channels_per_event=data_cfg.get("channels_per_event"),
        max_waveforms=args.max_waveforms,
        preprocess_kwargs=data_cfg.get("preprocess", {}) or {},
        coherent_groups_path=data_cfg.get("coherent_groups_path"),
    )

    out_path = Path(args.output)
    if out_path.suffix != ".npz":
        out_path = out_path.with_suffix(".npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saver = np.savez_compressed if args.compress else np.savez
    saver(out_path, **out)
    logger.info("Saved waveforms to %s (waveforms=%s)", out_path, out["waveforms"].shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
