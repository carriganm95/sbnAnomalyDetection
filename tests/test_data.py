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
from sbn_anomaly.data.sparse_window_dataset import SparseWindowDatasetPyG


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


# ---------------------------------------------------------------------------
# SparseWindowDatasetPyG
# ---------------------------------------------------------------------------


def _make_sparse_dataset(
    n_events: int = 40,
    n_channels: int = 16,
    avg_hits: int = 10,
    seed: int = 0,
    **kwargs,
) -> SparseWindowDatasetPyG:
    """Build a small SparseWindowDatasetPyG from synthetic sparse events."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(0, avg_hits * 2 + 1, size=n_events).astype(np.int64)
    total = int(sizes.sum())
    channels_flat = rng.integers(0, n_channels, size=total).astype(np.int64)
    integrals_flat = rng.random(total).astype(np.float32)
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    defaults = dict(history=2, window_size=4, n_bins=2, stride=1, radius=2)
    defaults.update(kwargs)
    return SparseWindowDatasetPyG(channels_flat, integrals_flat, offsets, n_channels, **defaults)


class TestSparseWindowDatasetPyG:
    def test_len(self):
        # events_per_sample = (history+1)*window_size = 3*4 = 12
        # samples = n_events - events_per_sample + 1 = 40 - 12 + 1 = 29
        ds = _make_sparse_dataset(n_events=40)
        assert len(ds) == 40 - (2 + 1) * 4 + 1

    def test_len_with_stride(self):
        ds = _make_sparse_dataset(n_events=40, stride=2)
        events_per_sample = (2 + 1) * 4
        expected = len(range(0, 40 - events_per_sample + 1, 2))
        assert len(ds) == expected

    def test_getitem_y_shape(self):
        n_bins, n_features = 2, 3
        ds = _make_sparse_dataset(
            n_events=40, n_bins=n_bins,
            node_features=["sum", "count", "max"],
        )
        item = ds[0]
        assert item.y.ndim == 2
        assert item.y.shape[1] == n_bins * n_features

    def test_getitem_x_shape(self):
        # x: (active_nodes, 1 + history * node_feat_dim)
        n_bins, n_features, history = 2, 3, 2
        ds = _make_sparse_dataset(
            n_events=40, n_bins=n_bins, history=history,
            node_features=["sum", "count", "max"],
        )
        item = ds[0]
        expected_x_width = 1 + history * (n_bins * n_features)
        assert item.x.shape[1] == expected_x_width

    def test_getitem_edge_index_shape(self):
        ds = _make_sparse_dataset()
        item = ds[0]
        assert item.edge_index.ndim == 2
        assert item.edge_index.shape[0] == 2

    def test_all_node_features(self):
        ds = _make_sparse_dataset(
            n_events=40,
            node_features=["sum", "min", "max", "mean", "stdev", "count"],
        )
        item = ds[0]
        assert item.y.shape[1] == 2 * 6  # n_bins * n_features

    def test_save_load_roundtrip(self, tmp_path):
        ds = _make_sparse_dataset(n_events=40)
        path = str(tmp_path / "events.npz")
        ds.save_events(path)
        ds2 = SparseWindowDatasetPyG.from_npz(
            path, history=2, window_size=4, n_bins=2, stride=1, radius=2,
        )
        assert len(ds2) == len(ds)
        item1 = ds[0]
        item2 = ds2[0]
        import torch
        assert torch.allclose(item1.y, item2.y)

    def test_empty_events_do_not_crash(self):
        """Dataset with interleaved empty events should construct and yield items."""
        n_channels = 8
        rng = np.random.default_rng(2)
        sizes = np.array([0, 5, 0, 3, 0, 7, 0, 4, 0, 6, 0, 8, 0, 2, 0, 5], dtype=np.int64)
        total = int(sizes.sum())
        channels_flat = rng.integers(0, n_channels, size=total).astype(np.int64)
        integrals_flat = rng.random(total).astype(np.float32)
        offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        ds = SparseWindowDatasetPyG(
            channels_flat, integrals_flat, offsets, n_channels,
            history=1, window_size=2, n_bins=2, stride=1, radius=1,
        )
        if len(ds) > 0:
            _ = ds[0]  # should not raise

    def test_feature_values_finite(self):
        """All feature values in a sample should be finite (no NaN/inf from stdev)."""
        import torch
        ds = _make_sparse_dataset(
            n_events=40,
            node_features=["sum", "min", "max", "mean", "stdev", "count"],
        )
        item = ds[0]
        assert torch.isfinite(item.x).all()
        assert torch.isfinite(item.y).all()
