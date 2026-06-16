"""Tests for the raw-waveform VAE model and trainer (require torch)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sbn_anomaly.models.tpc_waveform_vae import TPCWaveformVAE


def _make_vae(input_length=64, latent_dim=8, depth=3):
    return TPCWaveformVAE(
        input_length=input_length,
        latent_dim=latent_dim,
        base_channels=8,
        depth=depth,
        kernel_size=5,
    )


def test_forward_shapes():
    vae = _make_vae()
    x = torch.randn(4, 64)
    x_hat, mu, logvar, z = vae(x)
    assert x_hat.shape == (4, 64)
    assert mu.shape == (4, 8)
    assert logvar.shape == (4, 8)
    assert z.shape == (4, 8)


def test_accepts_bcl_and_bl_inputs():
    vae = _make_vae()
    x_bl = torch.randn(3, 64)
    x_bcl = x_bl.unsqueeze(1)
    out_bl, _, _, _ = vae(x_bl)
    out_bcl, _, _, _ = vae(x_bcl)
    assert out_bl.shape == out_bcl.shape == (3, 64)


def test_invalid_length_raises():
    with pytest.raises(ValueError):
        # 100 not divisible by 2**3 = 8
        TPCWaveformVAE(input_length=100, depth=3)


def test_encode_latents_is_mu_and_deterministic():
    vae = _make_vae().eval()
    x = torch.randn(5, 64)
    z1 = vae.encode_latents(x)
    z2 = vae.encode_latents(x)
    mu, _ = vae.encode(x)
    assert z1.shape == (5, 8)
    torch.testing.assert_close(z1, z2)        # deterministic in eval
    torch.testing.assert_close(z1, mu)        # equals the posterior mean


def test_scores_shapes_and_nonneg():
    vae = _make_vae().eval()
    x = torch.randn(6, 64)
    err = vae.reconstruction_error(x)
    score = vae.anomaly_score(x, beta=1.0)
    assert err.shape == (6,)
    assert score.shape == (6,)
    assert torch.all(err >= 0)


def test_training_step_reduces_loss():
    pytest.importorskip("torch")
    from sbn_anomaly.train.vae_trainer import VAETrainer
    from sbn_anomaly.data.raw_waveform_dataset import RawWaveformArrayDataset
    from torch.utils.data import DataLoader

    # A learnable structured signal: each waveform is a shifted ramp.
    rng = np.random.default_rng(0)
    base = np.linspace(-1, 1, 64, dtype=np.float32)
    wf = np.stack([base + rng.normal(0, 0.01, 64).astype(np.float32) for _ in range(256)])
    ds = RawWaveformArrayDataset(wf)
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    vae = _make_vae()
    trainer = VAETrainer(model=vae, lr=1e-3, max_epochs=3, beta=0.001, device="cpu")
    losses = trainer.train(loader)
    assert losses[-1] <= losses[0] + 1e-6  # loss should not increase overall


def test_array_dataset_returns_tuple():
    from sbn_anomaly.data.raw_waveform_dataset import RawWaveformArrayDataset

    wf = np.zeros((10, 64), dtype=np.float32)
    ds = RawWaveformArrayDataset(wf)
    item = ds[0]
    assert isinstance(item, tuple) and len(item) == 1
    assert item[0].shape == (64,)
