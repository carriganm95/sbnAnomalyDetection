"""Tests for TPC, PMT, Fusion, Window, and GNN forecaster models."""

from __future__ import annotations

import pytest
import torch

from sbn_anomaly.models.tpc_model import TPCAutoencoder
from sbn_anomaly.models.pmt_model import PMTAutoencoder
from sbn_anomaly.models.fusion_model import FusionAutoencoder
from sbn_anomaly.models.window_model import WindowAutoencoder
from sbn_anomaly.models.gnn_forecaster_pyg import GNNForecasterPyG


# ---------------------------------------------------------------------------
# TPC Autoencoder
# ---------------------------------------------------------------------------


class TestTPCAutoencoder:
    def test_forward_shape(self):
        model = TPCAutoencoder(input_dim=64, latent_dim=8, hidden_dims=(32, 16))
        x = torch.randn(4, 64)
        x_hat, z = model(x)
        assert x_hat.shape == (4, 64)
        assert z.shape == (4, 8)

    def test_reconstruction_error_shape(self):
        model = TPCAutoencoder(input_dim=64, latent_dim=8)
        x = torch.randn(10, 64)
        errors = model.reconstruction_error(x)
        assert errors.shape == (10,)

    def test_reconstruction_error_non_negative(self):
        model = TPCAutoencoder(input_dim=64, latent_dim=8)
        x = torch.randn(10, 64)
        errors = model.reconstruction_error(x)
        assert (errors >= 0).all()


# ---------------------------------------------------------------------------
# PMT Autoencoder
# ---------------------------------------------------------------------------


class TestPMTAutoencoder:
    def test_forward_shape(self):
        model = PMTAutoencoder(input_dim=32, latent_dim=4, hidden_dims=(16,))
        x = torch.randn(8, 32)
        x_hat, z = model(x)
        assert x_hat.shape == (8, 32)
        assert z.shape == (8, 4)

    def test_reconstruction_error_shape(self):
        model = PMTAutoencoder(input_dim=32, latent_dim=4)
        errors = model.reconstruction_error(torch.randn(5, 32))
        assert errors.shape == (5,)


# ---------------------------------------------------------------------------
# Fusion Autoencoder
# ---------------------------------------------------------------------------


class TestFusionAutoencoder:
    def test_joint_forward_shape(self):
        model = FusionAutoencoder(
            tpc_input_dim=64, pmt_input_dim=32, latent_dim=16, hidden_dims=(48, 24)
        )
        x_tpc = torch.randn(6, 64)
        x_pmt = torch.randn(6, 32)
        recon, z, combined = model(x_tpc, x_pmt)
        assert recon.shape == (6, 96)     # 64 + 32
        assert z.shape == (6, 16)
        assert combined.shape == (6, 96)

    def test_reconstruction_error_shape(self):
        model = FusionAutoencoder(tpc_input_dim=64, pmt_input_dim=32, latent_dim=16)
        errors = model.reconstruction_error(torch.randn(4, 64), torch.randn(4, 32))
        assert errors.shape == (4,)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            FusionAutoencoder(mode="invalid")

    def test_late_mode_requires_encoders(self):
        with pytest.raises(ValueError, match="tpc_encoder"):
            FusionAutoencoder(mode="late")

    def test_late_fusion_forward(self):
        tpc_enc = TPCAutoencoder(input_dim=64, latent_dim=8)
        pmt_enc = PMTAutoencoder(input_dim=32, latent_dim=4)
        model = FusionAutoencoder(
            tpc_input_dim=64,
            pmt_input_dim=32,
            latent_dim=8,
            hidden_dims=(8,),
            mode="late",
            tpc_encoder=tpc_enc,
            pmt_encoder=pmt_enc,
        )
        x_tpc = torch.randn(3, 64)
        x_pmt = torch.randn(3, 32)
        recon, z, combined = model(x_tpc, x_pmt)
        # combined dim = tpc latent + pmt latent = 8 + 4 = 12
        assert recon.shape == (3, 12)
        assert z.shape == (3, 8)


# ---------------------------------------------------------------------------
# Window Autoencoder
# ---------------------------------------------------------------------------


class TestWindowAutoencoder:
    def test_forward_shape(self):
        model = WindowAutoencoder(window_size=64, n_channels=1, latent_dim=16)
        x = torch.randn(4, 1, 64)
        x_hat, z = model(x)
        assert x_hat.shape == (4, 1, 64)
        assert z.shape == (4, 16)

    def test_forward_multi_channel(self):
        model = WindowAutoencoder(window_size=64, n_channels=4, latent_dim=32)
        x = torch.randn(2, 4, 64)
        x_hat, z = model(x)
        assert x_hat.shape == (2, 4, 64)

    def test_reconstruction_error_shape(self):
        model = WindowAutoencoder(window_size=64, n_channels=1, latent_dim=16)
        errors = model.reconstruction_error(torch.randn(5, 1, 64))
        assert errors.shape == (5,)


# ---------------------------------------------------------------------------
# GNN Forecaster (PyG)
# ---------------------------------------------------------------------------


def _make_pyg_batch(n_nodes: int, frame_feat_dim: int, history: int, n_graphs: int = 1):
    """Build a minimal PyG Batch for GNN forward tests."""
    from torch_geometric.data import Data, Batch

    graphs = []
    for _ in range(n_graphs):
        x = torch.randn(n_nodes, 1 + history * frame_feat_dim)
        y = torch.randn(n_nodes, frame_feat_dim)
        src = torch.arange(n_nodes)
        dst = (src + 1) % n_nodes
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
        graphs.append(Data(x=x, y=y, edge_index=edge_index))
    return Batch.from_data_list(graphs)


class TestGNNForecasterPyG:
    def test_forward_shape_single_graph(self):
        model = GNNForecasterPyG(
            frame_feat_dim=6, target_dim=6,
            gnn_hidden=8, gnn_layers=1, gru_hidden=16, history=2,
        )
        batch = _make_pyg_batch(n_nodes=10, frame_feat_dim=6, history=2)
        pred = model(batch)
        assert pred.shape == (10, 6)

    def test_forward_shape_batched(self):
        """Prediction shape sums node counts across graphs in the batch."""
        from torch_geometric.data import Data, Batch

        model = GNNForecasterPyG(
            frame_feat_dim=4, target_dim=4,
            gnn_hidden=8, gnn_layers=1, gru_hidden=16, history=3,
        )
        graphs = []
        for n in [5, 7]:
            src = torch.arange(n)
            dst = (src + 1) % n
            graphs.append(Data(
                x=torch.randn(n, 1 + 3 * 4),
                y=torch.randn(n, 4),
                edge_index=torch.stack([torch.cat([src, dst]), torch.cat([dst, src])]),
            ))
        pred = model(Batch.from_data_list(graphs))
        assert pred.shape == (12, 4)  # 5 + 7 nodes

    def test_forward_eval_mode(self):
        model = GNNForecasterPyG(
            frame_feat_dim=6, target_dim=6,
            gnn_hidden=8, gnn_layers=1, gru_hidden=16, history=2,
        )
        model.eval()
        batch = _make_pyg_batch(n_nodes=8, frame_feat_dim=6, history=2)
        with torch.no_grad():
            pred = model(batch)
        assert pred.shape == (8, 6)

    def test_invalid_temporal_dim_raises(self):
        """x with T*F not divisible by history should raise ValueError."""
        from torch_geometric.data import Data, Batch

        model = GNNForecasterPyG(
            frame_feat_dim=6, target_dim=6, history=4,
        )
        # 1 + 3*6 = 19 temporal features but history=4 expects 1+4*6=25
        batch = Batch.from_data_list([Data(
            x=torch.randn(5, 19),
            y=torch.randn(5, 6),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
        )])
        with pytest.raises(ValueError, match="not divisible"):
            model(batch)
