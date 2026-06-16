"""Tests for latent export and its hand-off to the GNN windows contract."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sbn_anomaly.data import build_raw_latents as brl
from sbn_anomaly.data.raw_digit_reader import RawEvent
from sbn_anomaly.models.tpc_waveform_vae import TPCWaveformVAE


class _FakeReader:
    """Stand-in for RawDigitReader yielding synthetic RawEvents."""

    def __init__(self, events):
        self._events = events

    def __iter__(self):
        return iter(self._events)


def _fake_events(n_events=6, n_channels=32, n_ticks=64):
    rng = np.random.default_rng(1)
    events = []
    for e in range(n_events):
        adc = rng.normal(2048, 4, size=(n_channels, n_ticks)).astype(np.float32)
        ped = np.full(n_channels, 2048, dtype=np.float32)
        chans = np.arange(n_channels, dtype=np.int32)
        events.append(RawEvent(run=1, subrun=0, event=e,
                               channel=chans, pedestal=ped, adc=adc))
    return events


def test_encode_events_to_latents_shapes(monkeypatch):
    n_channels, n_ticks, latent_dim = 32, 64, 8
    events = _fake_events(6, n_channels, n_ticks)
    monkeypatch.setattr(brl, "RawDigitReader",
                        lambda *a, **k: _FakeReader(events))

    vae = TPCWaveformVAE(input_length=n_ticks, latent_dim=latent_dim,
                         base_channels=8, depth=3, kernel_size=5).eval()

    latents, prov = brl.encode_events_to_latents(
        root_files=["dummy.root"],
        vae=vae,
        num_channels=n_channels,
        input_length=n_ticks,
        preprocess_kwargs={"remove_coherent": False, "scale": 64.0},
    )
    assert latents.shape == (6, n_channels, latent_dim)
    assert prov.shape == (6, 3)
    assert latents.dtype == np.float32


def test_missing_channels_are_zero(monkeypatch):
    # Event only fills channels 0..15 of a 32-channel detector.
    rng = np.random.default_rng(2)
    adc = rng.normal(2048, 4, size=(16, 64)).astype(np.float32)
    ped = np.full(16, 2048, dtype=np.float32)
    ev = RawEvent(run=1, subrun=0, event=0,
                  channel=np.arange(16, dtype=np.int32), pedestal=ped, adc=adc)
    monkeypatch.setattr(brl, "RawDigitReader", lambda *a, **k: _FakeReader([ev]))

    vae = TPCWaveformVAE(input_length=64, latent_dim=8, base_channels=8,
                         depth=3, kernel_size=5).eval()
    latents, _ = brl.encode_events_to_latents(
        root_files=["dummy.root"], vae=vae, num_channels=32, input_length=64,
        preprocess_kwargs={"remove_coherent": False},
    )
    # Channels 16..31 were never filled -> exactly zero.
    np.testing.assert_array_equal(latents[0, 16:], 0.0)


def test_latents_feed_graph_window_dataset(monkeypatch):
    """The (num_events, n_channels, latent_dim) array must satisfy the existing
    GraphWindowDatasetPyG contract used by the raw_gnn path."""
    pyg = pytest.importorskip("torch_geometric")
    from sbn_anomaly.data.graph_window_dataset_pyg import GraphWindowDatasetPyG

    n_channels, n_ticks, latent_dim = 16, 64, 8
    events = _fake_events(8, n_channels, n_ticks)
    monkeypatch.setattr(brl, "RawDigitReader", lambda *a, **k: _FakeReader(events))
    vae = TPCWaveformVAE(input_length=n_ticks, latent_dim=latent_dim,
                         base_channels=8, depth=3, kernel_size=5).eval()
    latents, _ = brl.encode_events_to_latents(
        root_files=["dummy.root"], vae=vae, num_channels=n_channels,
        input_length=n_ticks, preprocess_kwargs={"remove_coherent": False},
    )

    ds = GraphWindowDatasetPyG(latents, history=3, stride=1, radius=2,
                               prune_inactive=False)
    assert ds.node_feat_dim == latent_dim
    sample = ds[0]
    # x = channel_idx + history*latent_dim ; y = next-frame latents
    assert sample.x.shape[1] == 1 + 3 * latent_dim
    assert sample.y.shape[1] == latent_dim
