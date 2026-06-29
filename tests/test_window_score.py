"""Tests for window-level aggregation of per-channel reconstruction errors."""

import numpy as np
import pytest

from sbn_anomaly.infer.window_score import (
    aggregate_windows,
    group_max_mean_score,
    mean_score,
    max_score,
    topk_mean_score,
)


def _scores():
    # 2 windows x 6 channels; NaN = inactive. Window 1 has a hot 2-channel group.
    return np.array([
        [0.1, 0.2, np.nan, 0.1, 0.15, 0.1],   # quiet
        [0.1, 0.1, 0.1, 0.1, 5.0, 5.1],       # channels 4,5 anomalous
    ], dtype=np.float64)


def test_mean_and_max_ignore_nan():
    s = _scores()
    np.testing.assert_allclose(mean_score(s)[0], np.nanmean(s[0]))
    assert max_score(s)[1] == pytest.approx(5.1)


def test_topk_mean_picks_largest():
    s = _scores()
    out = topk_mean_score(s, k=2)
    assert out[1] == pytest.approx((5.0 + 5.1) / 2)
    # window 0 top-2 of [0.1,0.2,0.1,0.15,0.1] = (0.2+0.15)/2
    assert out[0] == pytest.approx((0.2 + 0.15) / 2)


def test_group_max_mean_localizes_bad_group():
    s = _scores()
    # groups: {0,1}=g0, {2,3}=g1, {4,5}=g2
    groups = np.array([0, 0, 1, 1, 2, 2])
    out = group_max_mean_score(s, groups)
    # window 1: worst group is g2 with mean (5.0+5.1)/2 = 5.05
    assert out[1] == pytest.approx(5.05)
    # window 0: all groups quiet -> small
    assert out[0] < 0.3


def test_group_max_mean_beats_plain_mean_for_localized_fault():
    # Realistic dilution: 40 channels, only one 2-channel group is hot.
    rng = np.random.default_rng(0)
    row = np.abs(rng.normal(0.1, 0.02, 40))
    row[10] = 5.0
    row[11] = 5.1
    s = row[None, :]
    groups = np.arange(40) // 2          # 2 channels per group
    gmm = group_max_mean_score(s, groups)[0]
    plain = mean_score(s)[0]
    # localized fault stands out strongly once diluted across many channels
    assert gmm == pytest.approx(5.05, abs=0.05)
    assert gmm > 10 * plain


def test_dispatch_and_unknown():
    s = _scores()
    np.testing.assert_allclose(aggregate_windows(s, "mean"), mean_score(s))
    with pytest.raises(ValueError):
        aggregate_windows(s, "nope")
    with pytest.raises(ValueError):
        aggregate_windows(s, "group_max_mean")  # missing groups


def test_channel_to_group_from_map():
    pd = pytest.importorskip("pandas")
    from sbn_anomaly.infer.window_score import channel_to_group
    df = pd.DataFrame({
        "offlchan": [0, 1, 2, 3],
        "FEMBSerialNum": ["A", "A", "B", "B"],
        "asic": [1, 1, 1, 2],
    })
    g_femb = channel_to_group(df, level="femb")
    assert g_femb[0] == g_femb[1] and g_femb[2] == g_femb[3] and g_femb[0] != g_femb[2]
    g_asic = channel_to_group(df, level="asic")
    assert g_asic[2] != g_asic[3]  # same FEMB, different ASIC
