"""Tests for the BaseTrainer training loop."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from sbn_anomaly.data.dataset import TPCDataset, PMTDataset, FusionDataset
from sbn_anomaly.models.tpc_model import TPCAutoencoder
from sbn_anomaly.models.pmt_model import PMTAutoencoder
from sbn_anomaly.models.fusion_model import FusionAutoencoder
from sbn_anomaly.train.tpc_trainer import TPCTrainer
from sbn_anomaly.train.pmt_trainer import PMTTrainer
from sbn_anomaly.train.fusion_trainer import FusionTrainer
from sbn_anomaly.train.window_trainer import WindowTrainer
from sbn_anomaly.models.window_model import WindowAutoencoder
from sbn_anomaly.data.dataset import WindowDataset
from sbn_anomaly.utils.plotting import _resolve_hist2d_spec
from sbn_anomaly.models.gnn_forecaster_pyg import GNNForecasterPyG
from sbn_anomaly.train.gnn_trainer import GNNTrainerPyG


class TestTPCTrainer:
    def test_train_one_epoch_returns_loss(self, tmp_path):
        feat = np.random.randn(64, 32).astype(np.float32)
        dataset = TPCDataset(feat)
        loader = DataLoader(dataset, batch_size=16)
        model = TPCAutoencoder(input_dim=32, latent_dim=4, hidden_dims=(16,))
        trainer = TPCTrainer(model=model, max_epochs=1, checkpoint_dir=str(tmp_path))
        losses = trainer.train(loader)
        assert len(losses) == 1
        assert losses[0] > 0

    def test_save_and_load(self, tmp_path):
        model = TPCAutoencoder(input_dim=32, latent_dim=4)
        trainer = TPCTrainer(model=model, max_epochs=1)
        ckpt = str(tmp_path / "model.pt")
        trainer.save(ckpt)
        trainer.load(ckpt)

    def test_writes_training_plots(self, tmp_path):
        feat = np.random.randn(64, 32).astype(np.float32)
        dataset = TPCDataset(feat)
        loader = DataLoader(dataset, batch_size=16)
        model = TPCAutoencoder(input_dim=32, latent_dim=4, hidden_dims=(16,))
        trainer = TPCTrainer(
            model=model,
            max_epochs=1,
            checkpoint_dir=str(tmp_path),
            anomaly_threshold=1.0,
        )
        trainer.train(loader)
        csv_path = trainer.save_training_history()
        plot_path = trainer.save_training_plots()
        assert csv_path is not None
        assert csv_path.exists()
        assert plot_path is not None
        assert plot_path.exists()

    def test_validation_loss_is_plotted(self, tmp_path):
        train_feat = np.random.randn(64, 32).astype(np.float32)
        val_feat = np.random.randn(32, 32).astype(np.float32)
        train_dataset = TPCDataset(train_feat)
        val_dataset = TPCDataset(val_feat)
        train_loader = DataLoader(train_dataset, batch_size=16)
        val_loader = DataLoader(val_dataset, batch_size=16)
        model = TPCAutoencoder(input_dim=32, latent_dim=4, hidden_dims=(16,))
        trainer = TPCTrainer(model=model, max_epochs=2, checkpoint_dir=str(tmp_path))

        trainer.train(train_loader, validation_loader=val_loader)
        plot_path = trainer.save_training_plots()

        assert "val_loss" in trainer.history
        assert len(trainer.history["val_loss"]) == 2
        assert np.isfinite(trainer.history["val_loss"]).all()
        assert plot_path is not None
        assert plot_path.exists()


class TestPlottingHistogramSpec:
    def test_feature_spec_with_bins_and_range(self):
        feature_names = ["hits0.h.integral", "hits0.h.time"]
        spec = {
            "hits0.h.integral": {"bins": 50, "range": [0, 500]},
            "hits0.h.time": {"bins": 40, "range": [0, 10000]},
        }

        resolved = _resolve_hist2d_spec(spec, 0, feature_names)

        assert resolved["bins"] == 50
        assert resolved["range"] == [[0.0, 500.0], [0.0, 500.0]]

    def test_feature_spec_defaults_when_missing(self):
        resolved = _resolve_hist2d_spec(None, 0, None)

        assert resolved == {"bins": 10}

    def test_feature_spec_with_underflow_overflow(self):
        feature_names = ["hits0.h.integral"]
        spec = {
            "hits0.h.integral": {
                "bins": 50,
                "range": [0, 500],
                "underflow": -25,
                "overflow": 550,
            }
        }

        resolved = _resolve_hist2d_spec(spec, 0, feature_names)

        assert resolved["bins"] == 50
        assert resolved["range"] == [[0.0, 500.0], [0.0, 500.0]]
        assert resolved["underflow"] == -25.0
        assert resolved["overflow"] == 550.0

    def test_feature_name_spec_applies_to_repeated_hit_indices(self):
        feature_names = ["hits0.h.integral", "hits0.h.time"]
        spec = {
            "hits0.h.integral": {"bins": 50, "range": [0, 500]},
            "hits0.h.time": {"bins": 40, "range": [0, 10000]},
        }

        # idx=2 corresponds to the same variable as idx=0 in the next hit block.
        resolved = _resolve_hist2d_spec(spec, 2, feature_names)

        assert resolved["bins"] == 50
        assert resolved["range"] == [[0.0, 500.0], [0.0, 500.0]]

    def test_feature_index_override_takes_precedence_over_name(self):
        feature_names = ["hits0.h.integral", "hits0.h.time"]
        spec = {
            "hits0.h.integral": {"bins": 50},
            "2": {"bins": 17},
        }

        resolved = _resolve_hist2d_spec(spec, 2, feature_names)

        assert resolved["bins"] == 17

    def test_save_training_plots_with_score_and_perf_metrics(self, tmp_path):
        model = TPCAutoencoder(input_dim=32, latent_dim=4, hidden_dims=(16,))
        trainer = TPCTrainer(model=model, max_epochs=1, checkpoint_dir=str(tmp_path))
        trainer.history = {
            "epoch": [1, 2, 3],
            "loss": [3.0, 2.0, 1.0],
            "score_p95": [0.3, 0.2, 0.1],
            "score_p99": [0.4, 0.3, 0.2],
            "anomaly_fraction_above_threshold": [0.1, 0.2, 0.3],
            "epoch_time_sec": [10.0, 9.0, 8.0],
            "events_per_sec": [100.0, 110.0, 120.0],
        }

        plot_path = trainer.save_training_plots()

        assert plot_path is not None
        assert plot_path.exists()


class TestPMTTrainer:
    def test_train_one_epoch(self):
        feat = np.random.randn(32, 16).astype(np.float32)
        dataset = PMTDataset(feat)
        loader = DataLoader(dataset, batch_size=8)
        model = PMTAutoencoder(input_dim=16, latent_dim=4, hidden_dims=(8,))
        trainer = PMTTrainer(model=model, max_epochs=1)
        losses = trainer.train(loader)
        assert len(losses) == 1


class TestFusionTrainer:
    def test_train_one_epoch(self):
        tpc_feat = np.random.randn(32, 32).astype(np.float32)
        pmt_feat = np.random.randn(32, 16).astype(np.float32)
        dataset = FusionDataset(tpc_feat, pmt_feat)
        loader = DataLoader(dataset, batch_size=8)
        model = FusionAutoencoder(tpc_input_dim=32, pmt_input_dim=16, latent_dim=8)
        trainer = FusionTrainer(model=model, max_epochs=1)
        losses = trainer.train(loader)
        assert len(losses) == 1


class TestWindowTrainer:
    def test_train_one_epoch(self):
        signal = np.random.randn(512).astype(np.float32)
        dataset = WindowDataset(signal, window_size=32, stride=16)
        loader = DataLoader(dataset, batch_size=16)
        model = WindowAutoencoder(window_size=32, n_channels=1, latent_dim=8)
        trainer = WindowTrainer(model=model, max_epochs=1)
        losses = trainer.train(loader)
        assert len(losses) == 1


# ---------------------------------------------------------------------------
# GNN Trainer (PyG)
# ---------------------------------------------------------------------------


def _make_gnn_loader(
    n_nodes: int = 10,
    frame_feat_dim: int = 6,
    history: int = 2,
    n_windows: int = 20,
    batch_size: int = 4,
):
    """Build a DataLoader of synthetic PyG graphs for GNN trainer tests."""
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as PyGDataLoader

    graphs = []
    for _ in range(n_windows):
        src = torch.arange(n_nodes)
        dst = (src + 1) % n_nodes
        graphs.append(Data(
            x=torch.randn(n_nodes, 1 + history * frame_feat_dim),
            y=torch.randn(n_nodes, frame_feat_dim),
            edge_index=torch.stack([torch.cat([src, dst]), torch.cat([dst, src])]),
        ))
    return PyGDataLoader(graphs, batch_size=batch_size, shuffle=False)


class TestGNNTrainerPyG:
    def test_train_one_epoch_returns_loss(self, tmp_path):
        loader = _make_gnn_loader()
        model = GNNForecasterPyG(
            frame_feat_dim=6, target_dim=6,
            gnn_hidden=8, gnn_layers=1, gru_hidden=16, history=2,
        )
        trainer = GNNTrainerPyG(model=model, max_epochs=1, device="cpu",
                                checkpoint_dir=str(tmp_path))
        losses = trainer.train(loader)
        assert len(losses) == 1
        assert losses[0] > 0

    def test_train_multiple_epochs(self, tmp_path):
        loader = _make_gnn_loader()
        model = GNNForecasterPyG(
            frame_feat_dim=6, target_dim=6,
            gnn_hidden=8, gnn_layers=1, gru_hidden=16, history=2,
        )
        trainer = GNNTrainerPyG(model=model, max_epochs=3, device="cpu",
                                checkpoint_dir=str(tmp_path))
        losses = trainer.train(loader)
        assert len(losses) == 3

    def test_save_and_load(self, tmp_path):
        model = GNNForecasterPyG(
            frame_feat_dim=6, target_dim=6,
            gnn_hidden=8, gnn_layers=1, gru_hidden=16, history=2,
        )
        trainer = GNNTrainerPyG(model=model, max_epochs=1, device="cpu")
        ckpt = str(tmp_path / "gnn.pt")
        trainer.save(ckpt)
        trainer.load(ckpt)

    def test_validation_loss_tracked(self, tmp_path):
        train_loader = _make_gnn_loader(n_windows=16)
        val_loader = _make_gnn_loader(n_windows=8)
        model = GNNForecasterPyG(
            frame_feat_dim=6, target_dim=6,
            gnn_hidden=8, gnn_layers=1, gru_hidden=16, history=2,
        )
        trainer = GNNTrainerPyG(model=model, max_epochs=2, device="cpu",
                                checkpoint_dir=str(tmp_path))
        trainer.train(train_loader, validation_loader=val_loader)

        assert "val_loss" in trainer.history
        assert len(trainer.history["val_loss"]) == 2
        assert np.isfinite(trainer.history["val_loss"]).all()

    def test_collect_scores_shape(self):
        loader = _make_gnn_loader(n_windows=12, batch_size=4)
        model = GNNForecasterPyG(
            frame_feat_dim=6, target_dim=6,
            gnn_hidden=8, gnn_layers=1, gru_hidden=16, history=2,
        )
        trainer = GNNTrainerPyG(model=model, max_epochs=1, device="cpu")
        trainer.train(loader)
        scores_mean, scores_max = trainer.collect_scores(loader)
        assert scores_mean.shape == (12,)
        assert scores_max.shape == (12,)
        assert np.all(scores_mean >= 0)
        assert np.all(scores_max >= scores_mean)

    def test_writes_training_plots(self, tmp_path):
        loader = _make_gnn_loader()
        model = GNNForecasterPyG(
            frame_feat_dim=6, target_dim=6,
            gnn_hidden=8, gnn_layers=1, gru_hidden=16, history=2,
        )
        trainer = GNNTrainerPyG(model=model, max_epochs=1, device="cpu",
                                checkpoint_dir=str(tmp_path))
        trainer.train(loader)
        csv_path = trainer.save_training_history(str(tmp_path))
        plot_path = trainer.save_training_plots(str(tmp_path))
        assert csv_path is not None and csv_path.exists()
        assert plot_path is not None and plot_path.exists()
