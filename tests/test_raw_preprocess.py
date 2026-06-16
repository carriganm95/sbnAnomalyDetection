"""Tests for raw-ADC preprocessing and the flat-ntuple reshape (numpy-only)."""

import numpy as np
import pytest

from sbn_anomaly.data.raw_preprocess import (
    pad_or_truncate,
    pedestal_subtract,
    preprocess_event,
    remove_coherent_noise,
    scale_waveforms,
)
from sbn_anomaly.data.raw_digit_reader import _reshape_adc


def test_pedestal_subtract_with_explicit_pedestal():
    adc = np.array([[105, 110, 95], [205, 200, 195]], dtype=np.float32)
    ped = np.array([100, 200], dtype=np.float32)
    out = pedestal_subtract(adc, ped)
    assert out.shape == adc.shape
    np.testing.assert_allclose(out, [[5, 10, -5], [5, 0, -5]])


def test_pedestal_subtract_median_fallback():
    adc = np.array([[10, 12, 11, 100]], dtype=np.float32)  # median ignores the hit
    out = pedestal_subtract(adc, None)
    # median of [10,12,11,100] = 11.5
    np.testing.assert_allclose(out[0], [-1.5, 0.5, -0.5, 88.5])


def test_remove_coherent_noise_cancels_common_mode():
    # 8 channels in one group share a common-mode sine; channel 0 also has a
    # localized spike. Median over the group is robust to the single spike, so
    # clean channels go to ~0 and the spike survives. (With only 2 channels the
    # median equals the mean and the spike would leak -- hence 8 here.)
    ticks = np.arange(16)
    common = np.sin(ticks / 3.0).astype(np.float32)
    wf = np.stack([common.copy() for _ in range(8)])
    wf[0, 8] += 50.0  # localized signal on channel 0
    out = remove_coherent_noise(wf, group_size=8)
    # Common mode removed -> clean channels are ~0 everywhere.
    assert np.abs(out[1:]).max() < 1e-4
    # The spike on channel 0 survives.
    assert out[0, 8] > 40.0


def test_remove_coherent_noise_explicit_groups():
    wf = np.ones((4, 5), dtype=np.float32)
    groups = np.array([0, 0, 1, 1])
    out = remove_coherent_noise(wf, groups=groups)
    # Within each group all channels identical -> common-mode removal zeros them.
    np.testing.assert_allclose(out, np.zeros_like(wf), atol=1e-6)


def test_scale_and_pad_truncate():
    wf = np.full((2, 4), 64.0, dtype=np.float32)
    np.testing.assert_allclose(scale_waveforms(wf, 64.0), np.ones_like(wf))

    padded = pad_or_truncate(wf, 6)
    assert padded.shape == (2, 6)
    np.testing.assert_allclose(padded[:, 4:], 0.0)

    truncated = pad_or_truncate(wf, 2)
    assert truncated.shape == (2, 2)


def test_preprocess_event_pipeline_shapes():
    rng = np.random.default_rng(0)
    adc = rng.normal(2048, 5, size=(128, 3000)).astype(np.float32)
    ped = np.full(128, 2048, dtype=np.float32)
    out = preprocess_event(
        adc, ped, coherent_group_size=32, scale=64.0, n_ticks=4096
    )
    assert out.shape == (128, 4096)
    assert out.dtype == np.float32
    # zero-padded tail
    np.testing.assert_allclose(out[:, 3000:], 0.0)


def test_reshape_adc_roundtrip():
    nchan, nticks = 5, 7
    flat = np.arange(nchan * nticks)
    mat = _reshape_adc(flat, nchan, nticks)
    assert mat.shape == (nchan, nticks)
    # channel-major: row c, tick t == c*nticks + t
    assert mat[2, 3] == 2 * nticks + 3


def test_reshape_adc_length_mismatch_raises():
    with pytest.raises(ValueError):
        _reshape_adc(np.arange(10), 3, 4)


def test_pedestal_subtract_bad_shape_raises():
    with pytest.raises(ValueError):
        pedestal_subtract(np.zeros(5), None)
