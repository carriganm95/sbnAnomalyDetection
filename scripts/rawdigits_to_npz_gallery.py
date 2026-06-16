#!/usr/bin/env python3
"""One-step: art RawDigit files -> preprocessed waveform .npz, via gallery.

Run on a LArSoft node (gpvm / container) where gallery + lardataobj are set up.
Reads raw ADC waveforms directly with gallery (PyROOT), applies the same
preprocessing as the rest of the pipeline, and writes the training .npz that
`sbn-train --config configs/raw_vae.yaml` (data.waveforms_path) consumes -- with
NO intermediate flat ROOT file.

Example:
    python scripts/rawdigits_to_npz_gallery.py \
        --config configs/raw_vae.yaml \
        --files /exp/icarus/data/.../raw_decoded_reco_094.root \
        --output data/raw_waveforms_train.npz \
        --tag daq --max-waveforms 2000000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    import numpy as np
    import yaml

    from sbn_anomaly.data.gallery_reader import GalleryRawDigitReader
    from sbn_anomaly.data.build_raw_waveforms import materialize_waveforms
    from sbn_anomaly.utils.logging import setup_logging

    p = argparse.ArgumentParser(description="art RawDigits -> waveform npz (gallery, one step)")
    p.add_argument("--config", required=True, help="raw_vae YAML config")
    p.add_argument("--files", nargs="+", required=True, help="art ROOT file(s)")
    p.add_argument("--output", required=True, help="output .npz (key 'waveforms')")
    p.add_argument("--tag", default="daq", help="RawDigit product tag / module label")
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--max-waveforms", type=int, default=None)
    p.add_argument("--compress", action="store_true", help="write compressed npz")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)

    setup_logging(args.log_level)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    input_length = int(model_cfg.get("input_length", 4096))

    reader = GalleryRawDigitReader(
        file_paths=args.files,
        tag=args.tag,
        max_events=args.max_events,
    )

    out = materialize_waveforms(
        events=reader,
        input_length=input_length,
        channels_per_event=data_cfg.get("channels_per_event"),
        max_waveforms=args.max_waveforms,
        preprocess_kwargs=data_cfg.get("preprocess", {}) or {},
    )

    out_path = Path(args.output)
    if out_path.suffix != ".npz":
        out_path = out_path.with_suffix(".npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saver = np.savez_compressed if args.compress else np.savez
    saver(out_path, **out)
    print(f"# Wrote waveforms {out['waveforms'].shape} to {out_path}")
    print(f"# Now set data.waveforms_path: {out_path} in {args.config} and run sbn-train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
