#!/usr/bin/env python3
"""Plot preprocessed waveforms from a raw_waveforms .npz.

The .npz is the output of ``sbn_anomaly.data.build_raw_waveforms`` and contains:
    waveforms  : (N, input_length) float32  -- preprocessed (pedestal-subtracted,
                                                coherent-noise removed, scaled)
    channel    : (N,) int32   -- detector channel id of each row
    event_idx  : (N,) int32   -- source event index
    provenance : (n_events, 3) int32 -- (run, subrun, event) per event

Examples
--------
First 6 waveforms in a grid:
    python scripts/plot_raw_waveforms.py --input data/raw_waveforms_train.npz \
        --output waveforms.png

Specific rows, overlaid:
    python scripts/plot_raw_waveforms.py --input data/raw_waveforms_train.npz \
        --rows 0 5 42 --overlay --output overlay.png

All waveforms of one channel:
    python scripts/plot_raw_waveforms.py --input data/raw_waveforms_train.npz \
        --channel 1234 --max 8 --output chan1234.png

Random sample:
    python scripts/plot_raw_waveforms.py --input data/raw_waveforms_train.npz \
        --random 9 --output sample.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless / file output
import matplotlib.pyplot as plt


def _select_rows(npz, args) -> np.ndarray:
    n = npz["waveforms"].shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)

    if args.rows:
        rows = np.array(args.rows, dtype=int)
    elif args.channel is not None and "channel" in npz:
        rows = np.where(npz["channel"] == args.channel)[0]
    elif args.event is not None and "event_idx" in npz:
        rows = np.where(npz["event_idx"] == args.event)[0]
    elif args.random:
        rng = np.random.default_rng(args.seed)
        rows = rng.choice(n, size=min(args.random, n), replace=False)
        rows.sort()
    else:
        rows = np.arange(min(args.max, n))

    if args.max is not None and rows.size > args.max:
        rows = rows[: args.max]
    valid = (rows >= 0) & (rows < n)
    return rows[valid]


def _label(npz, row: int) -> str:
    parts = [f"row {row}"]
    if "channel" in npz:
        parts.append(f"ch {int(npz['channel'][row])}")
    if "event_idx" in npz and "provenance" in npz and npz["provenance"].size:
        ei = int(npz["event_idx"][row])
        if 0 <= ei < npz["provenance"].shape[0]:
            r, s, e = npz["provenance"][ei]
            parts.append(f"run {r} sr {s} evt {e}")
    return "  ".join(parts)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Plot waveforms from a raw_waveforms .npz")
    p.add_argument("--input", required=True, help="Path to raw_waveforms .npz")
    p.add_argument("--output", default="raw_waveforms.png", help="Output image path")
    p.add_argument("--rows", type=int, nargs="+", help="Explicit row indices to plot")
    p.add_argument("--channel", type=int, default=None, help="Plot rows for this channel id")
    p.add_argument("--event", type=int, default=None, help="Plot rows for this event index")
    p.add_argument("--random", type=int, default=None, help="Plot N random rows")
    p.add_argument("--max", type=int, default=6, help="Max waveforms to plot (default 6)")
    p.add_argument("--overlay", action="store_true",
                   help="Overlay all on one axes instead of a grid")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for --random")
    p.add_argument("--dpi", type=int, default=120)
    args = p.parse_args(argv)

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Input not found: {in_path}", file=sys.stderr)
        return 1

    npz = np.load(in_path, allow_pickle=False)
    if "waveforms" not in npz:
        print("No 'waveforms' key in the npz.", file=sys.stderr)
        return 1

    rows = _select_rows(npz, args)
    if rows.size == 0:
        print("No matching waveforms to plot.", file=sys.stderr)
        return 1

    wf = npz["waveforms"]
    ticks = np.arange(wf.shape[1])

    if args.overlay:
        fig, ax = plt.subplots(figsize=(10, 4))
        for r in rows:
            ax.plot(ticks, wf[r], lw=0.8, label=_label(npz, int(r)))
        ax.set_xlabel("tick")
        ax.set_ylabel("preprocessed ADC (scaled)")
        ax.set_title(f"{in_path.name} — {rows.size} waveform(s)")
        if rows.size <= 12:
            ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
    else:
        ncols = 2 if rows.size > 1 else 1
        nrows = int(np.ceil(rows.size / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 2.2 * nrows),
                                 squeeze=False)
        for ax, r in zip(axes.flat, rows):
            ax.plot(ticks, wf[int(r)], lw=0.8)
            ax.set_title(_label(npz, int(r)), fontsize=8)
            ax.set_xlabel("tick")
            ax.set_ylabel("ADC (scaled)")
            ax.grid(alpha=0.3)
        for ax in axes.flat[rows.size:]:
            ax.axis("off")
        fig.suptitle(in_path.name)

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print(f"Saved {rows.size} waveform(s) to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
