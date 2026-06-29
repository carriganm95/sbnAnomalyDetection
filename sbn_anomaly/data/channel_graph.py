"""Build a geometry/electronics-aware channel adjacency graph for the GNN.

The default GNN graph (`build_sparse_edge_index`) connects channels by sequential
offline-channel index -- a rough electronics proxy. This module builds the
*correct* graph from the SBND channel map CSV
(`configs/SBNDTPCChannelMap_v2_with_positions.csv`), keyed on the offline channel
id (`offlchan`, the GNN node index).

Edge modes
----------
- ``electronics`` (default): connect channels that share readout hardware, where
  coherent noise and board/ASIC failures actually correlate. Two sub-rules,
  both on by default:
    * ASIC clique  -- fully connect the 16 channels on each ASIC (tightest
      common-mode unit).
    * FEMB chain   -- within each FEMB, connect channels adjacent in ``FEMBCh``
      (the readout chain) up to ``radius``.
  Note: a FEMB can span multiple wire planes, so these edges deliberately cross
  planes -- that's the point.
- ``wire``: physical neighbours -- within each (plane, EastWest, SideTop) group,
  connect channels with adjacent ``wire`` numbers up to ``radius``.
- ``both``: union of the two.

All edges are returned undirected (both directions) as an int64 ``(2, E)`` array
in ``offlchan`` space. Endpoints ``>= num_nodes`` are dropped so the graph lines
up with the dataset's node count.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

ELECTRONICS_KEYS = ("FEMBSerialNum", "asic", "FEMBCh")
WIRE_KEYS = ("plane", "EastWest", "SideTop", "wire")
NODE_KEY = "offlchan"


def _load_map(csv: Union[str, Path, "object"]):
    import pandas as pd

    if hasattr(csv, "columns"):  # already a DataFrame
        return csv
    return pd.read_csv(csv)


def _clique_pairs(ids: np.ndarray, out: set) -> None:
    ids = np.unique(ids)
    for i in range(len(ids)):
        a = int(ids[i])
        for j in range(i + 1, len(ids)):
            b = int(ids[j])
            out.add((a, b) if a < b else (b, a))


def _chain_pairs(ids_in_order: np.ndarray, radius: int, out: set) -> None:
    n = len(ids_in_order)
    for i in range(n):
        a = int(ids_in_order[i])
        for d in range(1, radius + 1):
            if i + d < n:
                b = int(ids_in_order[i + d])
                out.add((a, b) if a < b else (b, a))


def build_channel_map_edges(
    csv: Union[str, Path, "object"],
    *,
    mode: str = "electronics",
    radius: int = 2,
    asic_clique: bool = True,
    femb_chain: bool = True,
    num_nodes: Optional[int] = None,
) -> np.ndarray:
    """Return an undirected ``(2, E)`` int64 edge array in offlchan space."""
    if mode not in ("electronics", "wire", "both"):
        raise ValueError(f"mode must be electronics|wire|both, got {mode!r}")
    df = _load_map(csv)
    pairs: set = set()

    if mode in ("electronics", "both"):
        if asic_clique:
            for _, grp in df.groupby(["FEMBSerialNum", "asic"]):
                _clique_pairs(grp[NODE_KEY].to_numpy(), pairs)
        if femb_chain:
            for _, grp in df.groupby("FEMBSerialNum"):
                ordered = grp.sort_values("FEMBCh")[NODE_KEY].to_numpy()
                _chain_pairs(ordered, radius, pairs)

    if mode in ("wire", "both"):
        for _, grp in df.groupby(["plane", "EastWest", "SideTop"]):
            ordered = grp.sort_values("wire")[NODE_KEY].to_numpy()
            _chain_pairs(ordered, radius, pairs)

    if not pairs:
        return np.zeros((2, 0), dtype=np.int64)

    arr = np.fromiter((v for pair in pairs for v in pair), dtype=np.int64)
    arr = arr.reshape(-1, 2)
    a, b = arr[:, 0], arr[:, 1]

    if num_nodes is not None:
        keep = (a < num_nodes) & (b < num_nodes) & (a >= 0) & (b >= 0)
        a, b = a[keep], b[keep]

    # undirected: both directions
    src = np.concatenate([a, b])
    dst = np.concatenate([b, a])
    return np.stack([src, dst]).astype(np.int64)


def build_channel_map_edge_index(
    csv: Union[str, Path, "object"],
    *,
    mode: str = "electronics",
    radius: int = 2,
    asic_clique: bool = True,
    femb_chain: bool = True,
    num_nodes: Optional[int] = None,
):
    """torch ``LongTensor`` wrapper around :func:`build_channel_map_edges`."""
    import torch

    edges = build_channel_map_edges(
        csv, mode=mode, radius=radius, asic_clique=asic_clique,
        femb_chain=femb_chain, num_nodes=num_nodes,
    )
    return torch.from_numpy(edges).long()


def infer_num_nodes(csv: Union[str, Path, "object"]) -> int:
    """Max offlchan + 1 (node-index space), accounting for gaps in offlchan."""
    df = _load_map(csv)
    return int(df[NODE_KEY].max()) + 1
