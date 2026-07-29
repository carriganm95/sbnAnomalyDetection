"""Tests for window_mode='time' (fixed elapsed-time windows, variable event count).

Companion to test_node_features.py, which covers the (unchanged, default)
window_mode='event' feature math. These tests check that:
  - window_mode='event' (the default) is untouched by the new code path.
  - window_mode='time' derives window boundaries from evt_time instead of a
    fixed event count, including windows with zero events (a quiet gap) and
    a trailing partial window.
  - bin assignment within a time window is by elapsed time, not event index.
  - the new validation errors fire for the unsupported/misconfigured cases.
"""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from sbn_anomaly.data.sparse_window_dataset import SparseWindowDatasetPyG


def _time_dataset(**overrides):
    # 6 events, single channel (0), one hit per event, integral = event index + 1.
    # evt_time has a burst at t=[0,1,2,3], a quiet gap, then a burst at t=[10,11].
    channels = np.array([0, 0, 0, 0, 0, 0], dtype=np.int64)
    integrals = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    offsets = np.arange(7, dtype=np.int64)  # one hit per event
    evt_time = np.array([0, 1, 2, 3, 10, 11], dtype=np.float64)
    kwargs = dict(
        channels_flat=channels, integrals_flat=integrals, offsets=offsets,
        n_channels=1, n_bins=2, node_features=["sum", "count", "occupancy"],
        prune_inactive=False, evt_time=evt_time,
        reconstruction=True, window_mode="time",
        window_duration=4.0, stride_duration=4.0,
    )
    kwargs.update(overrides)
    return SparseWindowDatasetPyG(**kwargs)


def test_time_windows_have_variable_event_counts():
    ds = _time_dataset()
    # windows: [0,4) events 0-3 (4 events), [4,8) no events (gap), [4,6) events 4-5 (2 events)
    assert list(zip(ds._starts, ds._window_ends)) == [(0, 4), (4, 4), (4, 6)]
    assert len(ds) == 3


def test_time_bin_assignment_uses_elapsed_time_not_event_index():
    ds = _time_dataset()
    # Window 0: events at t=[0,1,2,3], window_duration=4, n_bins=2 -> bin width 2.
    # t=0,1 -> bin 0 ; t=2,3 -> bin 1 (not an even 2/2 index split coincidentally
    # matching here, but driven by /bin_width, not np.array_split on indices).
    splits = ds._time_bin_splits(0, 4)
    assert [s.tolist() for s in splits] == [[0, 1], [2, 3]]


def test_empty_time_window_is_all_zero_and_doesnt_crash():
    ds = _time_dataset()
    frame = ds[1]  # the (4, 4) empty window
    assert frame.x.shape[0] >= 1  # prune_inactive=False keeps the single channel
    y = frame.y.numpy()
    assert np.allclose(y, 0.0)


def test_trailing_partial_time_window_has_fewer_events():
    ds = _time_dataset()
    frame = ds._frame_for_bounds(4, 6).reshape(ds.num_nodes, -1)
    sums = frame[0].reshape(ds.n_bins, -1)[:, 0]  # "sum" is feature 0
    assert sums.sum() == pytest.approx(5.0 + 6.0)


def test_event_mode_default_is_unaffected():
    # Same events, default window_mode='event' -- should behave exactly as
    # before this change (fixed window_size, index-based bins).
    channels = np.array([0, 0, 0, 0, 0, 0], dtype=np.int64)
    integrals = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    offsets = np.arange(7, dtype=np.int64)
    ds = SparseWindowDatasetPyG(
        channels_flat=channels, integrals_flat=integrals, offsets=offsets,
        n_channels=1, window_size=3, n_bins=1, stride=3,
        node_features=["sum"], prune_inactive=False,
        reconstruction=True,
    )
    assert ds.window_mode == "event"
    assert ds._window_ends is None
    assert len(ds) == 2  # floor(6/3)
    frame0 = ds._compute_frame(0, 3).reshape(1, -1)
    assert frame0[0, 0] == pytest.approx(1 + 2 + 3)


def test_time_mode_requires_reconstruction():
    with pytest.raises(NotImplementedError):
        _time_dataset(reconstruction=False)


def test_time_mode_requires_evt_time():
    with pytest.raises(ValueError):
        _time_dataset(evt_time=None)


def test_time_mode_requires_positive_duration():
    with pytest.raises(ValueError):
        _time_dataset(window_duration=0.0)


def test_event_count_graph_level_attribute_time_mode():
    ds = _time_dataset()
    # windows: (0,4) -> 4 events, (4,4) -> 0 events, (4,6) -> 2 events
    assert ds[0].event_count.item() == pytest.approx(4.0)
    assert ds[1].event_count.item() == pytest.approx(0.0)
    assert ds[2].event_count.item() == pytest.approx(2.0)


def test_event_count_graph_level_attribute_event_mode():
    channels = np.array([0, 0, 0, 0, 0, 0], dtype=np.int64)
    integrals = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    offsets = np.arange(7, dtype=np.int64)
    ds = SparseWindowDatasetPyG(
        channels_flat=channels, integrals_flat=integrals, offsets=offsets,
        n_channels=1, window_size=3, n_bins=1, stride=3,
        node_features=["sum"], prune_inactive=False, reconstruction=True,
    )
    assert ds[0].event_count.item() == pytest.approx(3.0)
    assert ds[1].event_count.item() == pytest.approx(3.0)


def test_window_metadata_includes_event_count_without_provenance():
    # No evt_run/evt_time provenance loaded -> window_metadata() should still
    # return event_count (it needs no provenance), just not the run/subrun keys.
    channels = np.array([0, 0, 0, 0, 0, 0], dtype=np.int64)
    integrals = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    offsets = np.arange(7, dtype=np.int64)
    ds = SparseWindowDatasetPyG(
        channels_flat=channels, integrals_flat=integrals, offsets=offsets,
        n_channels=1, window_size=3, n_bins=1, stride=3,
        node_features=["sum"], prune_inactive=False, reconstruction=True,
    )
    meta = ds.window_metadata()
    assert "event_count" in meta
    assert "first_run" not in meta
    assert meta["event_count"].tolist() == [3, 3]


def test_window_metadata_event_count_time_mode():
    ds = _time_dataset()
    meta = ds.window_metadata()
    assert meta["event_count"].tolist() == [4, 0, 2]


def test_time_mode_requires_monotonic_evt_time():
    channels = np.array([0, 0, 0], dtype=np.int64)
    integrals = np.array([1, 2, 3], dtype=np.float32)
    offsets = np.arange(4, dtype=np.int64)
    with pytest.raises(ValueError):
        SparseWindowDatasetPyG(
            channels_flat=channels, integrals_flat=integrals, offsets=offsets,
            n_channels=1, n_bins=1, node_features=["sum"], prune_inactive=False,
            evt_time=np.array([0, 5, 2], dtype=np.float64),  # not sorted
            reconstruction=True, window_mode="time", window_duration=4.0,
        )
