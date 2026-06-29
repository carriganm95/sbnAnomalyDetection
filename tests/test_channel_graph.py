"""Tests for the geometry/electronics-aware channel graph (numpy-only)."""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from sbn_anomaly.data.channel_graph import (
    build_channel_map_edges,
    infer_num_nodes,
)


def _toy_map():
    # 2 FEMBs x (2 ASICs x 2 channels) = 8 channels, with plane/wire/TPC info.
    rows = []
    offl = 0
    for femb in ("F0", "F1"):
        for asic in (1, 2):
            for fch in (0, 1):
                rows.append(dict(
                    offlchan=offl, FEMBSerialNum=femb, asic=asic, FEMBCh=offl,
                    plane=0, EastWest="East", SideTop="S", wire=offl + 1,
                ))
                offl += 1
    return pd.DataFrame(rows)


def test_electronics_edges_are_intra_femb():
    df = _toy_map()
    e = build_channel_map_edges(df, mode="electronics", radius=2)
    femb = dict(zip(df.offlchan, df.FEMBSerialNum))
    assert e.shape[0] == 2 and e.shape[1] > 0
    assert all(femb[int(a)] == femb[int(b)] for a, b in zip(e[0], e[1]))
    # symmetric (both directions present)
    s = set(map(tuple, e.T.tolist()))
    assert all((b, a) in s for a, b in s)


def test_asic_clique_connects_same_asic():
    df = _toy_map()
    e = build_channel_map_edges(df, mode="electronics", radius=0, femb_chain=False)
    # radius 0 + no chain -> only ASIC cliques; channels 0,1 share an ASIC
    s = {tuple(sorted(p)) for p in zip(e[0].tolist(), e[1].tolist())}
    assert (0, 1) in s and (2, 3) in s
    # channels 1 and 2 are different ASICs -> not connected
    assert (1, 2) not in s


def test_wire_edges_within_plane_and_adjacent():
    df = _toy_map()
    e = build_channel_map_edges(df, mode="wire", radius=1)
    wire = dict(zip(df.offlchan, df.wire))
    assert all(abs(wire[int(a)] - wire[int(b)]) <= 1 for a, b in zip(e[0], e[1]))


def test_num_nodes_clamp():
    df = _toy_map()
    e = build_channel_map_edges(df, mode="electronics", radius=2, num_nodes=4)
    assert e.size == 0 or int(e.max()) < 4


def test_infer_num_nodes():
    df = _toy_map()
    assert infer_num_nodes(df) == 8


def test_empty_when_no_pairs():
    df = _toy_map()
    e = build_channel_map_edges(df, mode="electronics", radius=0,
                                asic_clique=False, femb_chain=False)
    assert e.shape == (2, 0)
