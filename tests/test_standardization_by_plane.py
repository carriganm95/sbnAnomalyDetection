"""Tests for standardize_by='plane' (per-plane feature standardization)."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from sbn_anomaly.data.sparse_window_dataset import SparseWindowDatasetPyG


def _make_two_plane_dataset(n_events=100, seed=0):
    """channels 0,1,2 -> plane 0 (integral ~ N(10,1)); 3,4,5 -> plane 1 (~ N(50,5))."""
    rng = np.random.default_rng(seed)
    n_channels = 6
    planes_by_chan = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)

    channels_list, integrals_list, planes_list, sizes = [], [], [], []
    for _ in range(n_events):
        n_hits = rng.integers(2, 6)
        ch = rng.integers(0, n_channels, size=n_hits).astype(np.int64)
        pl = planes_by_chan[ch]
        integ = np.where(
            pl == 0,
            rng.normal(10, 1, n_hits),
            rng.normal(50, 5, n_hits),
        ).astype(np.float32)
        channels_list.append(ch)
        integrals_list.append(integ)
        planes_list.append(pl)
        sizes.append(n_hits)

    channels_flat = np.concatenate(channels_list)
    integrals_flat = np.concatenate(integrals_list)
    planes_flat = np.concatenate(planes_list)
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    return channels_flat, integrals_flat, planes_flat, offsets, n_channels


def test_plane_standardization_shape_and_grouping():
    channels_flat, integrals_flat, planes_flat, offsets, n_channels = _make_two_plane_dataset()
    ds = SparseWindowDatasetPyG(
        channels_flat, integrals_flat, offsets, n_channels,
        window_size=10, n_bins=1, stride=10, radius=1,
        reconstruction=True, standardize=True, standardize_by="plane",
        node_features=["mean"], planes_flat=planes_flat, prune_inactive=False,
    )
    assert ds.feature_mean.shape == (n_channels, 1)
    assert ds.feature_std.shape == (n_channels, 1)

    # channels within the same plane must get identical fitted stats
    np.testing.assert_array_equal(ds.feature_mean[0], ds.feature_mean[1])
    np.testing.assert_array_equal(ds.feature_mean[0], ds.feature_mean[2])
    np.testing.assert_array_equal(ds.feature_mean[3], ds.feature_mean[4])
    np.testing.assert_array_equal(ds.feature_mean[4], ds.feature_mean[5])

    # plane 0 (~N(10,1)) and plane 1 (~N(50,5)) should be clearly separated
    assert ds.feature_mean[0, 0] < 20
    assert ds.feature_mean[3, 0] > 40


def test_plane_standardization_vs_global_pooled():
    channels_flat, integrals_flat, planes_flat, offsets, n_channels = _make_two_plane_dataset()
    common = dict(
        window_size=10, n_bins=1, stride=10, radius=1,
        reconstruction=True, standardize=True,
        node_features=["mean"], prune_inactive=False,
    )
    ds_global = SparseWindowDatasetPyG(
        channels_flat, integrals_flat, offsets, n_channels,
        standardize_by="global", planes_flat=planes_flat, **common,
    )
    ds_plane = SparseWindowDatasetPyG(
        channels_flat, integrals_flat, offsets, n_channels,
        standardize_by="plane", planes_flat=planes_flat, **common,
    )
    assert ds_global.feature_mean.shape == (1,)  # unchanged legacy shape
    # pooled global mean should sit strictly between the two plane means
    assert ds_plane.feature_mean[0, 0] < ds_global.feature_mean[0] < ds_plane.feature_mean[3, 0]


def test_plane_standardization_requires_plane_info():
    channels_flat, integrals_flat, planes_flat, offsets, n_channels = _make_two_plane_dataset()
    with pytest.raises(ValueError):
        SparseWindowDatasetPyG(
            channels_flat, integrals_flat, offsets, n_channels,
            window_size=10, n_bins=1, stride=10, radius=1,
            reconstruction=True, standardize=True, standardize_by="plane",
            node_features=["mean"], prune_inactive=False,
            # no planes_flat, no channel_map
        )


def test_invalid_standardize_by_raises():
    channels_flat, integrals_flat, planes_flat, offsets, n_channels = _make_two_plane_dataset()
    with pytest.raises(ValueError):
        SparseWindowDatasetPyG(
            channels_flat, integrals_flat, offsets, n_channels,
            window_size=10, n_bins=1, stride=10, radius=1,
            standardize_by="bogus", node_features=["mean"], prune_inactive=False,
        )


def test_undersampled_plane_falls_back_to_pooled():
    """A plane with far too few active-channel samples should reuse the
    pooled fit rather than a noisy plane-specific one."""
    channels_flat, integrals_flat, planes_flat, offsets, n_channels = _make_two_plane_dataset()

    # Insert one extra hit (channel 6, brand-new plane 2) into event 0's hit
    # range -- the only hit ever seen from plane 2, nowhere near
    # min_plane_samples. Inserting into an existing event (rather than
    # appending a new one after the end) guarantees it actually falls inside
    # a sampled window rather than being truncated off by window_size math.
    insert_at = int(offsets[1])  # end of event 0's hit range
    channels_flat = np.insert(channels_flat, insert_at, 6).astype(np.int64)
    integrals_flat = np.insert(integrals_flat, insert_at, 999.0).astype(np.float32)
    planes_flat = np.insert(planes_flat, insert_at, 2).astype(np.int32)
    offsets = offsets.copy()
    offsets[1:] += 1  # every event after event 0 shifts by the one inserted hit
    n_channels = 7

    ds_plane = SparseWindowDatasetPyG(
        channels_flat, integrals_flat, offsets, n_channels,
        window_size=10, n_bins=1, stride=10, radius=1,
        reconstruction=True, standardize=True, standardize_by="plane",
        node_features=["mean"], planes_flat=planes_flat, prune_inactive=False,
        min_plane_samples=20,
    )
    ds_global = SparseWindowDatasetPyG(
        channels_flat, integrals_flat, offsets, n_channels,
        window_size=10, n_bins=1, stride=10, radius=1,
        reconstruction=True, standardize=True, standardize_by="global",
        node_features=["mean"], planes_flat=planes_flat, prune_inactive=False,
    )
    # channel 6 (plane 2, 1 sample -- way under min_plane_samples) should fall
    # back to the pooled stats, not its own (single-point, std=0) statistics.
    np.testing.assert_allclose(ds_plane.feature_mean[6], ds_global.feature_mean, rtol=1e-5)
