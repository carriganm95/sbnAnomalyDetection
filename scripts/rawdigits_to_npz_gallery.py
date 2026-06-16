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

# Make `import sbn_anomaly` work no matter how the script is launched, by adding
# the repo root (parent of scripts/) to sys.path. Avoids needing PYTHONPATH or
# an editable install just to run the gallery dump in an SL7 container.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main(argv=None) -> int:
    import numpy as np
    import yaml

    from sbn_anomaly.data.gallery_reader import GalleryRawDigitReader
    from sbn_anomaly.data.build_raw_waveforms import write_waveform_shards
    from sbn_anomaly.data.root_files import resolve_root_files
    from sbn_anomaly.utils.logging import setup_logging

    p = argparse.ArgumentParser(description="art RawDigits -> waveform npz (gallery, one step)")
    p.add_argument("--config", required=True, help="raw_vae YAML config")
    p.add_argument("--files", "--root-files", nargs="+", default=None, dest="files",
                   help="art ROOT file path(s) or glob pattern(s)")
    p.add_argument("--root-file-list", nargs="+", default=None, metavar="FILE",
                   help="Manifest file(s) with one ROOT path per line (# comments ok), "
                        "like the GNN's --root-file-list.")
    p.add_argument("--output", required=True,
                   help="output .npz; with --shard-size, shards become <stem>_000.npz, ...")
    p.add_argument("--tag", default="daq", help="RawDigit product tag / module label")
    p.add_argument("--shard-size", type=int, default=None,
                   help="Waveforms per output shard. When set, write <output>_NNN.npz "
                        "every shard-size waveforms and keep processing.")
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--max-waveforms", type=int, default=None,
                   help="Global cap across all shards (None = process everything).")
    p.add_argument("--compress", action="store_true", help="write compressed npz")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)

    setup_logging(args.log_level)

    inputs = list(args.files or []) + list(args.root_file_list or [])
    if not inputs:
        p.error("provide --files / --root-files and/or --root-file-list")
    files = resolve_root_files(inputs)
    if not files:
        p.error("no ROOT files resolved from the given inputs")
    print(f"# resolved {len(files)} ROOT file(s)")

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    input_length = int(model_cfg.get("input_length", 4096))

    # One reader per file so file boundaries (for per-file counts) are known and
    # shards can span files.
    def reader_factory(fpath):
        return GalleryRawDigitReader(file_paths=[fpath], tag=args.tag)

    written = write_waveform_shards(
        file_inputs=files,
        reader_factory=reader_factory,
        input_length=input_length,
        output=args.output,
        shard_size=args.shard_size,
        channels_per_event=data_cfg.get("channels_per_event"),
        max_events=args.max_events,
        max_waveforms=args.max_waveforms,
        preprocess_kwargs=data_cfg.get("preprocess", {}) or {},
        coherent_groups_path=data_cfg.get("coherent_groups_path"),
        compress=args.compress,
    )
    print(f"# Wrote {len(written)} shard file(s): {written}")
    print(f"# Point data.waveforms_path at them (a glob works) and run sbn-train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
