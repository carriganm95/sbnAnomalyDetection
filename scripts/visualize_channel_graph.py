#!/usr/bin/env python3
"""Visualize how the electronics graph (edge_mode: electronics) connects channels.

For one or more channels, prints their FEMB/ASIC grouping and saves a FEMB-layout
plot (ASIC cliques + cross-ASIC readout-chain edges), an offline-channel-id spread
plot, and a geometry plot showing a FEMB spans wire planes. The interactive
version is notebooks/electronics_graph_viz.ipynb.

    python scripts/visualize_channel_graph.py --channels 10 15 --radius 4 --outdir plots/graph
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from sbn_anomaly.data.channel_graph import build_channel_map_edges


def describe(df, edges, offl):
    r = df[df.offlchan == offl].iloc[0]
    nbrs = sorted({int(b) for a, b in zip(edges[0], edges[1]) if int(a) == offl})
    same = [n for n in nbrs if int(df[df.offlchan == n].iloc[0].asic) == int(r.asic)]
    cross = [n for n in nbrs if n not in same]
    print(f"channel {offl}: FEMB {r.FEMBSerialNum}, ASIC {int(r.asic)}, "
          f"FEMBCh {int(r.FEMBCh)}, plane {int(r.plane)}, wire {int(r.wire)}")
    print(f"  ASIC-clique neighbours ({len(same)}): {same}")
    print(f"  readout-chain cross-ASIC neighbours ({len(cross)}): {cross}")
    print(f"  total degree: {len(nbrs)}")


def draw_femb(df, edges, highlight, ax):
    r = df[df.offlchan == highlight].iloc[0]
    femb = r.FEMBSerialNum
    sub = df[df.FEMBSerialNum == femb]
    pos = {int(o): (int(a), 15 - (int(fc) % 16))
           for o, a, fc in zip(sub.offlchan, sub.asic, sub.FEMBCh)}
    for a in range(8):
        ax.add_patch(FancyBboxPatch((a - 0.35, -0.6), 0.7, 16.2,
                     boxstyle="round,pad=0.05", fc="0.93", ec="0.7", zorder=0))
        ax.text(a, 16.1, f"ASIC {a}", ha="center", fontsize=8, color="0.4")
    ordered = sub.sort_values("FEMBCh")
    ax.plot([pos[int(o)][0] for o in ordered.offlchan],
            [pos[int(o)][1] for o in ordered.offlchan], "-", color="0.8", lw=1, zorder=1)
    ax.scatter([pos[int(o)][0] for o in sub.offlchan],
               [pos[int(o)][1] for o in sub.offlchan],
               c=sub.asic, cmap="tab10", s=90, zorder=2, edgecolor="w")
    nbrs = sorted({int(b) for a, b in zip(edges[0], edges[1]) if int(a) == highlight})
    ha = int(r.asic)
    for nb in nbrs:
        if nb not in pos:
            continue
        col = "tab:blue" if int(df[df.offlchan == nb].iloc[0].asic) == ha else "tab:red"
        ax.plot([pos[highlight][0], pos[nb][0]], [pos[highlight][1], pos[nb][1]],
                color=col, lw=1.2, alpha=0.7, zorder=3)
    ax.scatter([pos[highlight][0]], [pos[highlight][1]], s=280, facecolor="none",
               edgecolor="k", lw=2.5, zorder=4)
    ax.set_title(f"FEMB {femb}: ch {highlight} (ASIC {ha}, FEMBCh {int(r.FEMBCh)})  "
                 f"degree={len(nbrs)}\nblue=ASIC clique, red=cross-ASIC chain")
    ax.set_xlabel("ASIC"); ax.set_ylabel("position within ASIC (FEMBCh % 16)")
    ax.set_xticks(range(8)); ax.set_xlim(-0.7, 7.7); ax.set_ylim(-1, 17)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Visualize the electronics channel graph.")
    p.add_argument("--channels", type=int, nargs="+", default=[10, 15])
    p.add_argument("--radius", type=int, default=4)
    p.add_argument("--channel-map", default=str(_REPO / "configs" /
                                                "SBNDTPCChannelMap_v2_with_positions.csv"))
    p.add_argument("--outdir", default="plots/graph")
    args = p.parse_args(argv)

    df = pd.read_csv(args.channel_map)
    edges = build_channel_map_edges(df, mode="electronics", radius=args.radius)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    for c in args.channels:
        describe(df, edges, c)
        fig, ax = plt.subplots(figsize=(9, 6)); draw_femb(df, edges, c, ax)
        fig.tight_layout(); fig.savefig(out / f"femb_ch{c}.png", dpi=120); plt.close(fig)

        nbrs = sorted({int(b) for a, b in zip(edges[0], edges[1]) if int(a) == c})
        fig, ax = plt.subplots(figsize=(10, 2.6))
        ax.axvspan(c - args.radius, c + args.radius, color="tab:blue", alpha=0.15,
                   label=f"sequential +/-{args.radius}")
        ax.scatter(nbrs, [1] * len(nbrs), c="tab:red", s=45, label="electronics neighbours")
        ax.scatter([c], [1], c="k", s=150, marker="*", label=f"channel {c}")
        ax.set_yticks([]); ax.set_xlabel("offline channel id")
        ax.legend(loc="upper center", ncol=3, fontsize=8)
        ax.set_title(f"ch {c}: electronics neighbours span offlchan {min(nbrs)}..{max(nbrs)}")
        fig.tight_layout(); fig.savefig(out / f"offlchan_ch{c}.png", dpi=120); plt.close(fig)

    print(f"# wrote plots to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
