"""Tests for occupancy / hit_rate / timing node features (need torch + PyG)."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from sbn_anomaly.data.sparse_window_dataset import SparseWindowDatasetPyG


def _toy_dataset(node_features, with_times=True):
    # 2 events, channel 5 only. Event0: integrals [10,20] times [3,5];
    # Event1: integral [30] time [7].
    channels = np.array([5, 5, 5], dtype=np.int64)
    integrals = np.array([10, 20, 30], dtype=np.float32)
    offsets = np.array([0, 2, 3], dtype=np.int64)   # event0 -> hits[0:2], event1 -> hits[2:3]
    times = np.array([3, 5, 7], dtype=np.float32) if with_times else None
    return SparseWindowDatasetPyG(
        channels_flat=channels, integrals_flat=integrals, offsets=offsets,
        n_channels=6, window_size=2, n_bins=1, stride=1,
        node_features=node_features, prune_inactive=False,
        times_flat=times,
    )


def test_occupancy_hit_rate_timing_values():
    ds = _toy_dataset(["occupancy", "hit_rate", "time_mean", "time_spread", "count"])
    frame = ds._compute_frame(0, 2).reshape(6, -1)  # (channels, n_bins*F) with n_bins=1
    occ, rate, tmean, tspread, count = frame[5]
    assert occ == pytest.approx(1.0)          # channel hit in both events
    assert rate == pytest.approx(1.5)         # 3 hits / 2 events
    assert tmean == pytest.approx(5.0)        # mean([3,5,7])
    assert tspread == pytest.approx(np.std([3, 5, 7]), abs=1e-5)
    assert count == pytest.approx(3.0)


def test_occupancy_le_one_and_zero_for_inactive():
    ds = _toy_dataset(["occupancy"])
    frame = ds._compute_frame(0, 2).reshape(6, -1)
    assert frame[5, 0] == pytest.approx(1.0)
    assert frame[0, 0] == pytest.approx(0.0)  # channel never hit


def test_timing_without_times_raises():
    with pytest.raises(ValueError):
        _toy_dataset(["time_mean"], with_times=False)


def test_node_feat_dim_scales_with_features():
    ds = _toy_dataset(["sum", "occupancy", "hit_rate", "time_mean"])
    assert ds.node_feat_dim == 1 * 4  # n_bins(1) * n_features(4)


# ---------------------------------------------------------------------------
# width / sumadc / mult / hasSP group features
# ---------------------------------------------------------------------------

def _toy_dataset_extra(node_features, **overrides):
    # Same 2-event / channel-5 toy as _toy_dataset, plus width/sumadc/mult/hasSP.
    channels = np.array([5, 5, 5], dtype=np.int64)
    integrals = np.array([10, 20, 30], dtype=np.float32)
    offsets = np.array([0, 2, 3], dtype=np.int64)
    kwargs = dict(
        widths_flat=np.array([1, 2, 3], dtype=np.float32),
        sumadcs_flat=np.array([100, 200, 300], dtype=np.float32),
        mults_flat=np.array([1, 2, 1], dtype=np.float32),
        hassps_flat=np.array([1, 0, 1], dtype=np.float32),
    )
    kwargs.update(overrides)
    return SparseWindowDatasetPyG(
        channels_flat=channels, integrals_flat=integrals, offsets=offsets,
        n_channels=6, window_size=2, n_bins=1, stride=1,
        node_features=node_features, prune_inactive=False,
        **kwargs,
    )


def test_width_sumadc_mult_sp_group_stats():
    features = [
        "width_sum", "width_mean", "width_min", "width_max", "width_stdev",
        "sumadc_sum", "sumadc_mean", "sumadc_min", "sumadc_max", "sumadc_stdev",
        "mult_sum", "mult_mean", "mult_min", "mult_max", "mult_stdev",
        "sp_fraction",
    ]
    ds = _toy_dataset_extra(features)
    frame = ds._compute_frame(0, 2).reshape(6, -1)
    row = frame[5]

    widths = np.array([1, 2, 3], dtype=np.float32)
    sumadcs = np.array([100, 200, 300], dtype=np.float32)
    mults = np.array([1, 2, 1], dtype=np.float32)
    hassps = np.array([1, 0, 1], dtype=np.float32)
    expected = [
        widths.sum(), widths.mean(), widths.min(), widths.max(), widths.std(),
        sumadcs.sum(), sumadcs.mean(), sumadcs.min(), sumadcs.max(), sumadcs.std(),
        mults.sum(), mults.mean(), mults.min(), mults.max(), mults.std(),
        hassps.mean(),
    ]
    assert row == pytest.approx(expected, abs=1e-4)

    # Channel never hit should stay all-zero.
    assert frame[0] == pytest.approx(np.zeros_like(row))


def test_group_feature_without_matching_flat_array_raises():
    for feat, kw in (
        ("width_mean", "widths_flat"),
        ("sumadc_mean", "sumadcs_flat"),
        ("mult_mean", "mults_flat"),
        ("sp_fraction", "hassps_flat"),
    ):
        overrides = {
            "widths_flat": None, "sumadcs_flat": None,
            "mults_flat": None, "hassps_flat": None,
        }
        with pytest.raises(ValueError):
            _toy_dataset_extra([feat], **overrides)


def test_node_feat_dim_scales_with_group_features():
    ds = _toy_dataset_extra(["width_mean", "sumadc_max", "mult_stdev", "sp_fraction"])
    assert ds.node_feat_dim == 1 * 4  # n_bins(1) * n_features(4)
