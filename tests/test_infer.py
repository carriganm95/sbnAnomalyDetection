"""Tests for the AnomalyScorer inference wrapper."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

import sbn_anomaly.data.root_files as root_files_module
from sbn_anomaly.infer.inferrer import AnomalyScorer
from sbn_anomaly.infer import cli as infer_cli
from sbn_anomaly.models.tpc_model import TPCAutoencoder
from sbn_anomaly.models.pmt_model import PMTAutoencoder
from sbn_anomaly.models.fusion_model import FusionAutoencoder
from sbn_anomaly.models.window_model import WindowAutoencoder
from sbn_anomaly.data.root_files import resolve_root_files


class TestAnomalyScorer:
    def test_tpc_score_shape(self):
        model = TPCAutoencoder(input_dim=32, latent_dim=4)
        scorer = AnomalyScorer(model, model_type="tpc")
        features = np.random.randn(10, 32).astype(np.float32)
        scores = scorer.score(features)
        assert scores.shape == (10,)
        assert (scores >= 0).all()

    def test_pmt_score_shape(self):
        model = PMTAutoencoder(input_dim=16, latent_dim=4)
        scorer = AnomalyScorer(model, model_type="pmt")
        features = np.random.randn(8, 16).astype(np.float32)
        scores = scorer.score(features)
        assert scores.shape == (8,)

    def test_fusion_score_shape(self):
        model = FusionAutoencoder(tpc_input_dim=32, pmt_input_dim=16, latent_dim=8)
        scorer = AnomalyScorer(model, model_type="fusion")
        tpc_feat = np.random.randn(6, 32).astype(np.float32)
        pmt_feat = np.random.randn(6, 16).astype(np.float32)
        scores = scorer.score(tpc_feat, pmt_feat)
        assert scores.shape == (6,)

    def test_window_score_shape(self):
        model = WindowAutoencoder(window_size=32, n_channels=1, latent_dim=8)
        scorer = AnomalyScorer(model, model_type="window")
        # Input shape (B, C, L)
        features = np.random.randn(4, 1, 32).astype(np.float32)
        scores = scorer.score(features)
        assert scores.shape == (4,)

    def test_window_score_auto_channel_dim(self):
        model = WindowAutoencoder(window_size=32, n_channels=1, latent_dim=8)
        scorer = AnomalyScorer(model, model_type="window")
        # (B, L) – missing channel dim, should be added automatically
        features = np.random.randn(4, 32).astype(np.float32)
        scores = scorer.score(features)
        assert scores.shape == (4,)

    def test_is_anomaly_above_threshold(self):
        model = TPCAutoencoder(input_dim=32, latent_dim=4)
        scorer = AnomalyScorer(model, model_type="tpc", threshold=0.0)
        features = np.random.randn(5, 32).astype(np.float32)
        flags = scorer.is_anomaly(features)
        assert flags.dtype == bool
        assert flags.shape == (5,)

    def test_is_anomaly_no_threshold_raises(self):
        model = TPCAutoencoder(input_dim=32, latent_dim=4)
        scorer = AnomalyScorer(model, model_type="tpc")
        with pytest.raises(ValueError, match="threshold"):
            scorer.is_anomaly(np.random.randn(5, 32).astype(np.float32))

    def test_invalid_model_type_raises(self):
        model = TPCAutoencoder(input_dim=32, latent_dim=4)
        with pytest.raises(ValueError, match="model_type"):
            AnomalyScorer(model, model_type="unknown")

    def test_from_checkpoint(self, tmp_path):
        model = TPCAutoencoder(input_dim=32, latent_dim=4)
        ckpt = tmp_path / "model.pt"
        torch.save(model.state_dict(), str(ckpt))

        model2 = TPCAutoencoder(input_dim=32, latent_dim=4)
        scorer = AnomalyScorer.from_checkpoint(ckpt, model2, model_type="tpc")
        features = np.random.randn(3, 32).astype(np.float32)
        scores = scorer.score(features)
        assert scores.shape == (3,)

    def test_from_training_checkpoint(self, tmp_path):
        """AnomalyScorer.from_checkpoint should also accept training checkpoint dicts."""
        model = TPCAutoencoder(input_dim=32, latent_dim=4)
        ckpt = tmp_path / "train_ckpt.pt"
        torch.save(
            {"epoch": 1, "model_state_dict": model.state_dict(), "loss": 0.01},
            str(ckpt),
        )
        model2 = TPCAutoencoder(input_dim=32, latent_dim=4)
        scorer = AnomalyScorer.from_checkpoint(ckpt, model2, model_type="tpc")
        scores = scorer.score(np.random.randn(2, 32).astype(np.float32))
        assert scores.shape == (2,)


class TestInferCliRootFiles:
    def test_score_tpc_from_root_waveform_mode(self, monkeypatch):
        import awkward as ak

        class DummyStreamer:
            def __init__(self, *args, **kwargs):
                pass

            def stream(self):
                yield ak.Array({"wf": [[1.0, 2.0, 3.0], [4.0, 5.0]]})

        class DummyScorer:
            def score(self, features):
                # One score per event.
                return features.mean(axis=1)

        monkeypatch.setattr("sbn_anomaly.data.streaming.RootStreamer", DummyStreamer)

        cfg = {
            "model": {"input_dim": 4},
            "data": {
                "tree_name": "tree",
                "waveform_branch": "wf",
                "tpc_branches": ["wf"],
                "batch_size_stream": 16,
                "normalize": False,
            },
            "inference": {},
            "training": {},
        }

        scores = infer_cli._score_tpc_from_root(cfg, DummyScorer(), ["dummy.root"])
        assert scores.shape == (2,)
        # Event 1 feature vector -> [1, 2, 3, 0], mean 1.5
        assert scores[0] == pytest.approx(1.5)

    def test_main_rejects_root_files_for_non_tpc(self, tmp_path):
        cfg_path = tmp_path / "cfg.yaml"
        ckpt_path = tmp_path / "model.pt"

        # Minimal PMT checkpoint for scorer construction.
        model = PMTAutoencoder(input_dim=16, latent_dim=4)
        torch.save(model.state_dict(), str(ckpt_path))

        cfg = {
            "model_type": "pmt",
            "model": {"input_dim": 16, "latent_dim": 4},
            "inference": {"checkpoint_path": str(ckpt_path)},
        }
        cfg_path.write_text(yaml.safe_dump(cfg))

        rc = infer_cli.main(
            [
                "--config",
                str(cfg_path),
                "--root-files",
                "dummy.root",
            ]
        )
        assert rc == 1

    def test_root_file_manifest_resolution(self, tmp_path):
        root_a = tmp_path / "a.root"
        root_b = tmp_path / "b.root"
        root_a.write_text("")
        root_b.write_text("")

        manifest = tmp_path / "train_roots.txt"
        manifest.write_text(
            f"# comment\n{root_a}\n{tmp_path}/*.root\n"
        )

        resolved = resolve_root_files([str(manifest)])
        assert str(root_a) in resolved
        assert str(root_b) in resolved

    def test_root_file_url_is_treated_as_literal(self, monkeypatch):
        url = "root://xrootd.example.org//store/data/file.root"

        def fail_on_path(*args, **kwargs):
            raise AssertionError("Path should not be used for xrootd URLs")

        monkeypatch.setattr(root_files_module, "Path", fail_on_path, raising=False)

        resolved = resolve_root_files([url])
        assert resolved == [url]
