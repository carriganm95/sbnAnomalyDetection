"""Tests for data streaming and event joining (no ROOT files required)."""

from __future__ import annotations

import numpy as np
import pytest

from sbn_anomaly.data.dataset import (
    FusionDataset,
    PMTDataset,
    TPCDataset,
    WindowDataset,
)
from sbn_anomaly.data.event_joiner import EventJoiner
from sbn_anomaly.data.streaming import RootStreamer


# ---------------------------------------------------------------------------
# TPCDataset
# ---------------------------------------------------------------------------


class TestTPCDataset:
    def test_len(self):
        feat = np.random.randn(100, 32).astype(np.float32)
        ds = TPCDataset(feat)
        assert len(ds) == 100

    def test_getitem_without_labels(self):
        feat = np.random.randn(10, 32).astype(np.float32)
        ds = TPCDataset(feat)
        item = ds[0]
        assert isinstance(item, tuple)
        assert item[0].shape == (32,)

    def test_getitem_with_labels(self):
        feat = np.random.randn(10, 32).astype(np.float32)
        labs = np.zeros(10, dtype=np.int64)
        ds = TPCDataset(feat, labels=labs)
        x, y = ds[0]
        assert x.shape == (32,)
        assert y.item() == 0


# ---------------------------------------------------------------------------
# PMTDataset
# ---------------------------------------------------------------------------


class TestPMTDataset:
    def test_len(self):
        feat = np.random.randn(50, 16).astype(np.float32)
        ds = PMTDataset(feat)
        assert len(ds) == 50

    def test_getitem_without_labels(self):
        feat = np.random.randn(5, 16).astype(np.float32)
        ds = PMTDataset(feat)
        (x,) = ds[0]
        assert x.shape == (16,)


# ---------------------------------------------------------------------------
# FusionDataset
# ---------------------------------------------------------------------------


class TestFusionDataset:
    def test_len(self):
        tpc = np.random.randn(80, 32).astype(np.float32)
        pmt = np.random.randn(80, 16).astype(np.float32)
        ds = FusionDataset(tpc, pmt)
        assert len(ds) == 80

    def test_getitem(self):
        tpc = np.random.randn(10, 32).astype(np.float32)
        pmt = np.random.randn(10, 16).astype(np.float32)
        ds = FusionDataset(tpc, pmt)
        x_tpc, x_pmt = ds[0]
        assert x_tpc.shape == (32,)
        assert x_pmt.shape == (16,)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            FusionDataset(
                np.random.randn(10, 8).astype(np.float32),
                np.random.randn(5, 4).astype(np.float32),
            )


# ---------------------------------------------------------------------------
# WindowDataset
# ---------------------------------------------------------------------------


class TestWindowDataset:
    def test_len_stride_1(self):
        signal = np.random.randn(1000).astype(np.float32)
        ds = WindowDataset(signal, window_size=64, stride=1)
        assert len(ds) == 1000 - 64 + 1

    def test_len_stride_32(self):
        signal = np.random.randn(256).astype(np.float32)
        ds = WindowDataset(signal, window_size=64, stride=32)
        expected = len(range(0, 256 - 64 + 1, 32))
        assert len(ds) == expected

    def test_window_shape(self):
        signal = np.random.randn(500).astype(np.float32)
        ds = WindowDataset(signal, window_size=32, stride=16)
        (w,) = ds[0]
        assert w.shape == (32,)


# ---------------------------------------------------------------------------
# EventJoiner
# ---------------------------------------------------------------------------


class TestEventJoiner:
    def _make_batch(self, keys: list[tuple[int, int, int]], n_extra: int = 4):
        """Build a simple awkward array with run/subrun/event + dummy data."""
        import awkward as ak

        run = np.array([k[0] for k in keys], dtype=np.int32)
        subrun = np.array([k[1] for k in keys], dtype=np.int32)
        event = np.array([k[2] for k in keys], dtype=np.int32)
        dummy = np.random.randn(len(keys), n_extra).astype(np.float32)
        return ak.Array({"run": run, "subrun": subrun, "event": event, "feat": dummy})

    def test_inner_join_full_overlap(self):
        keys = [(1, 0, i) for i in range(10)]
        tpc = self._make_batch(keys)
        pmt = self._make_batch(keys)
        joiner = EventJoiner()
        t, p = joiner.join(tpc, pmt)
        assert len(t) == 10
        assert len(p) == 10

    def test_inner_join_partial_overlap(self):
        tpc_keys = [(1, 0, i) for i in range(10)]
        pmt_keys = [(1, 0, i) for i in range(5, 15)]
        tpc = self._make_batch(tpc_keys)
        pmt = self._make_batch(pmt_keys)
        joiner = EventJoiner()
        t, p = joiner.join(tpc, pmt)
        assert len(t) == 5
        assert len(p) == 5

    def test_inner_join_no_overlap(self):
        tpc = self._make_batch([(1, 0, i) for i in range(5)])
        pmt = self._make_batch([(2, 0, i) for i in range(5)])
        joiner = EventJoiner()
        t, p = joiner.join(tpc, pmt)
        assert len(t) == 0
        assert len(p) == 0

    def test_join_preserves_event_alignment(self):
        """After joining, run/subrun/event must match element-wise."""
        import awkward as ak

        tpc = self._make_batch([(1, 0, 0), (1, 0, 1), (1, 0, 2)])
        pmt = self._make_batch([(1, 0, 2), (1, 0, 0)])
        joiner = EventJoiner()
        t, p = joiner.join(tpc, pmt)
        tpc_events = ak.to_numpy(t["event"]).tolist()
        pmt_events = ak.to_numpy(p["event"]).tolist()
        assert tpc_events == pmt_events


class TestRootStreamer:
    def test_xrootd_url_is_preserved(self):
        url = "root://fndcadoor.fnal.gov//store/data/file.root"
        streamer = RootStreamer(url, tree_name="sbn_tree")

        assert streamer.file_paths == [url]
