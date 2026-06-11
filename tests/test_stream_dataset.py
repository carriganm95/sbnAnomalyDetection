"""Tests for TPCStreamDataset feature extraction (no ROOT files required)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

from sbn_anomaly.data.stream_dataset import (
    TPCStreamDataset,
    extract_hit_features,
    extract_tpc_features,
)


# ---------------------------------------------------------------------------
# extract_tpc_features – unit tests for padding / truncation
# ---------------------------------------------------------------------------


class TestExtractTPCFeatures:
    def test_truncation(self):
        """Arrays longer than input_dim must be truncated."""
        data = np.arange(300, dtype=np.float32)
        result = extract_tpc_features(data, input_dim=256)
        assert result.shape == (256,)
        np.testing.assert_array_equal(result, data[:256])

    def test_padding(self):
        """Arrays shorter than input_dim must be zero-padded at the end."""
        data = np.arange(100, dtype=np.float32)
        result = extract_tpc_features(data, input_dim=256)
        assert result.shape == (256,)
        np.testing.assert_array_equal(result[:100], data)
        np.testing.assert_array_equal(result[100:], np.zeros(156, dtype=np.float32))

    def test_exact_length(self):
        """Arrays of exactly input_dim must be returned unchanged."""
        data = np.random.randn(256).astype(np.float32)
        result = extract_tpc_features(data, input_dim=256)
        assert result.shape == (256,)
        np.testing.assert_array_equal(result, data)

    def test_output_dtype(self):
        """Output must always be float32."""
        data = np.ones(10, dtype=np.float64)
        result = extract_tpc_features(data, input_dim=20)
        assert result.dtype == np.float32

    def test_empty_input_pads_to_full_zeros(self):
        """Empty waveform should produce a zero-filled vector."""
        data = np.array([], dtype=np.float32)
        result = extract_tpc_features(data, input_dim=16)
        assert result.shape == (16,)
        np.testing.assert_array_equal(result, np.zeros(16, dtype=np.float32))


# ---------------------------------------------------------------------------
# TPCStreamDataset – iterator behavior (mocked streamer)
# ---------------------------------------------------------------------------


def _make_fake_waveform_batch(n_events: int, n_ticks: int = 300):
    """Build a minimal awkward-array-like batch for testing."""
    import awkward as ak

    waveforms = [np.random.randn(n_ticks).astype(np.float32) for _ in range(n_events)]
    return ak.Array({"tpc_waveform": waveforms})


class TestTPCStreamDataset:
    def _dataset(self, **kwargs):
        """Return a TPCStreamDataset pointing at a non-existent dummy file."""
        defaults = dict(
            file_paths=["/tmp/dummy.root"],
            tree_name="sbn_tree",
            waveform_branch="tpc_waveform",
            input_dim=64,
        )
        defaults.update(kwargs)
        return TPCStreamDataset(**defaults)

    def test_xrootd_url_is_preserved(self):
        url = "root://fndcadoor.fnal.gov//icarus/persistent/users/micarrig/DQM/file.root"
        ds = self._dataset(file_paths=[url])

        assert ds.file_paths == [url]

    def test_hit_branches_are_automatically_loaded(self):
        ds = self._dataset(
            hit_branches=["hits0.h.integral", "hits0.h.sumadc"],
            branches=["meta.run", "meta.subrun", "meta.evt"],
            waveform_branch=None,
        )

        assert ds.branches == [
            "meta.run",
            "meta.subrun",
            "meta.evt",
            "hits0.h.integral",
            "hits0.h.sumadc",
        ]

    def test_yields_correct_feature_dim(self):
        """Each yielded tuple must contain a tensor of shape (input_dim,)."""
        ds = self._dataset(input_dim=64)
        batch = _make_fake_waveform_batch(n_events=5, n_ticks=300)

        # Patch RootStreamer so it yields our synthetic batch.
        with patch(
            "sbn_anomaly.data.stream_dataset.RootStreamer"
        ) as MockStreamer:
            instance = MockStreamer.return_value
            instance.stream.return_value = iter([batch])

            items = list(ds)

        assert len(items) == 5
        for (tensor,) in items:
            assert isinstance(tensor, torch.Tensor)
            assert tensor.shape == (64,)
            assert tensor.dtype == torch.float32

    def test_max_events_limits_output(self):
        """max_events must cap the total number of yielded samples."""
        ds = self._dataset(input_dim=32, max_events=3)
        batch = _make_fake_waveform_batch(n_events=10, n_ticks=100)

        with patch(
            "sbn_anomaly.data.stream_dataset.RootStreamer"
        ) as MockStreamer:
            instance = MockStreamer.return_value
            instance.stream.return_value = iter([batch])

            items = list(ds)

        assert len(items) == 3

    def test_normalize_produces_unit_std(self):
        """With normalize=True, features with non-zero variance should have std≈1."""
        ds = self._dataset(input_dim=64, normalize=True)
        # Create a waveform with known non-trivial variance.
        wave = np.arange(64, dtype=np.float32)  # 0..63
        import awkward as ak

        batch = ak.Array({"tpc_waveform": [wave]})

        with patch(
            "sbn_anomaly.data.stream_dataset.RootStreamer"
        ) as MockStreamer:
            instance = MockStreamer.return_value
            instance.stream.return_value = iter([batch])

            (tensor,) = list(ds)[0]

        arr = tensor.numpy()
        assert abs(arr.std() - 1.0) < 1e-4, f"Expected std≈1, got {arr.std()}"

    def test_padding_in_dataset(self):
        """Waveforms shorter than input_dim must be zero-padded."""
        input_dim = 64
        n_ticks = 20  # shorter than input_dim
        ds = self._dataset(input_dim=input_dim)
        import awkward as ak

        wave = np.ones(n_ticks, dtype=np.float32)
        batch = ak.Array({"tpc_waveform": [wave]})

        with patch(
            "sbn_anomaly.data.stream_dataset.RootStreamer"
        ) as MockStreamer:
            instance = MockStreamer.return_value
            instance.stream.return_value = iter([batch])

            (tensor,) = list(ds)[0]

        assert tensor.shape == (input_dim,)
        np.testing.assert_array_equal(tensor.numpy()[:n_ticks], wave)
        np.testing.assert_array_equal(
            tensor.numpy()[n_ticks:], np.zeros(input_dim - n_ticks)
        )

    def test_truncation_in_dataset(self):
        """Waveforms longer than input_dim must be truncated."""
        input_dim = 32
        n_ticks = 100  # longer than input_dim
        ds = self._dataset(input_dim=input_dim)
        import awkward as ak

        wave = np.arange(n_ticks, dtype=np.float32)
        batch = ak.Array({"tpc_waveform": [wave]})

        with patch(
            "sbn_anomaly.data.stream_dataset.RootStreamer"
        ) as MockStreamer:
            instance = MockStreamer.return_value
            instance.stream.return_value = iter([batch])

            (tensor,) = list(ds)[0]

        assert tensor.shape == (input_dim,)
        np.testing.assert_array_equal(tensor.numpy(), wave[:input_dim])

    def test_non_finite_features_warn_once(self, caplog):
        """Repeated non-finite events should only warn once per iterator."""
        import awkward as ak

        ds = self._dataset(input_dim=8)
        batch = ak.Array(
            {
                "tpc_waveform": [
                    np.array([1.0, np.nan], dtype=np.float32),
                    np.array([2.0, np.inf], dtype=np.float32),
                    np.array([3.0, -np.inf], dtype=np.float32),
                ]
            }
        )

        with patch(
            "sbn_anomaly.data.stream_dataset.RootStreamer"
        ) as MockStreamer:
            instance = MockStreamer.return_value
            instance.stream.return_value = iter([batch])

            with caplog.at_level("WARNING", logger="sbn_anomaly.data.stream_dataset"):
                items = list(ds)

        assert len(items) == 3
        warning_messages = [record.message for record in caplog.records if record.levelname == "WARNING"]
        assert len(warning_messages) == 1
        assert "Found non-finite feature values in event" in warning_messages[0]
        for (tensor,) in items:
            assert torch.isfinite(tensor).all()


# ---------------------------------------------------------------------------
# extract_hit_features – unit tests
# ---------------------------------------------------------------------------


class TestExtractHitFeatures:
    def test_single_branch_truncation(self):
        data = {"hit_integral": np.arange(300, dtype=np.float32)}
        result = extract_hit_features(data, ["hit_integral"], input_dim=256)
        assert result.shape == (256,)
        np.testing.assert_array_equal(result, data["hit_integral"][:256])

    def test_single_branch_padding(self):
        data = {"hit_integral": np.ones(10, dtype=np.float32)}
        result = extract_hit_features(data, ["hit_integral"], input_dim=32)
        assert result.shape == (32,)
        np.testing.assert_array_equal(result[:10], np.ones(10, dtype=np.float32))
        np.testing.assert_array_equal(result[10:], np.zeros(22, dtype=np.float32))

    def test_multiple_branches_concatenated(self):
        """Values should be interleaved by hit before padding."""
        data = {
            "hit_integral": np.array([1.0, 2.0], dtype=np.float32),
            "hit_charge": np.array([3.0, 4.0], dtype=np.float32),
        }
        result = extract_hit_features(data, ["hit_integral", "hit_charge"], input_dim=8)
        assert result.shape == (8,)
        # Interleaved: [integral_0, charge_0, integral_1, charge_1, ...]
        np.testing.assert_array_equal(result[:4], [1.0, 3.0, 2.0, 4.0])
        np.testing.assert_array_equal(result[4:], np.zeros(4, dtype=np.float32))

    def test_missing_branch_skipped(self):
        """Branches absent from event_data should be silently skipped."""
        data = {"hit_integral": np.array([5.0], dtype=np.float32)}
        result = extract_hit_features(
            data, ["hit_integral", "nonexistent"], input_dim=4
        )
        assert result.shape == (4,)
        assert result[0] == pytest.approx(5.0)

    def test_empty_event_returns_zeros(self):
        data = {"hit_integral": np.array([], dtype=np.float32)}
        result = extract_hit_features(data, ["hit_integral"], input_dim=16)
        assert result.shape == (16,)
        np.testing.assert_array_equal(result, np.zeros(16, dtype=np.float32))

    def test_output_dtype(self):
        data = {"hit_integral": np.ones(5, dtype=np.float64)}
        result = extract_hit_features(data, ["hit_integral"], input_dim=8)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# TPCStreamDataset – hit mode tests
# ---------------------------------------------------------------------------


def _make_hit_batch(n_events: int, n_hits: int = 10):
    """Build a fake awkward batch with hit-level scalar branches."""
    import awkward as ak

    integrals = [np.random.randn(n_hits).astype(np.float32) for _ in range(n_events)]
    charges = [np.random.randn(n_hits).astype(np.float32) for _ in range(n_events)]
    amplitudes = [np.random.randn(n_hits).astype(np.float32) for _ in range(n_events)]
    return ak.Array(
        {
            "hit_integral": integrals,
            "hit_charge": charges,
            "hit_amplitude": amplitudes,
        }
    )


class TestTPCStreamDatasetHitMode:
    HIT_BRANCHES = ["hit_integral", "hit_charge", "hit_amplitude"]

    def _dataset(self, **kwargs):
        defaults = dict(
            file_paths=["/tmp/dummy.root"],
            tree_name="sbn_tree",
            waveform_branch=None,
            hit_branches=self.HIT_BRANCHES,
            input_dim=64,
        )
        defaults.update(kwargs)
        return TPCStreamDataset(**defaults)

    def test_yields_correct_feature_dim(self):
        ds = self._dataset(input_dim=64)
        batch = _make_hit_batch(n_events=5, n_hits=10)

        with patch("sbn_anomaly.data.stream_dataset.RootStreamer") as MockStreamer:
            MockStreamer.return_value.stream.return_value = iter([batch])
            items = list(ds)

        assert len(items) == 5
        for (tensor,) in items:
            assert tensor.shape == (64,)
            assert tensor.dtype == torch.float32

    def test_max_events_limits_output(self):
        ds = self._dataset(max_events=3)
        batch = _make_hit_batch(n_events=10)

        with patch("sbn_anomaly.data.stream_dataset.RootStreamer") as MockStreamer:
            MockStreamer.return_value.stream.return_value = iter([batch])
            items = list(ds)

        assert len(items) == 3

    def test_branches_requested_from_streamer(self):
        """The streamer should be asked for exactly the hit_branches."""
        ds = self._dataset()
        batch = _make_hit_batch(n_events=1)

        with patch("sbn_anomaly.data.stream_dataset.RootStreamer") as MockStreamer:
            MockStreamer.return_value.stream.return_value = iter([batch])
            _ = list(ds)

        _, init_kwargs = MockStreamer.call_args
        assert set(init_kwargs["branches"]) == set(self.HIT_BRANCHES)

    def test_no_waveform_and_no_hit_branches_raises(self):
        with pytest.raises(ValueError, match="hit_branches"):
            TPCStreamDataset(
                file_paths=["/tmp/dummy.root"],
                waveform_branch=None,
                hit_branches=None,
            )

    def test_normalize_hit_mode(self):
        """normalize=True should produce unit std in hit mode too."""
        ds = self._dataset(input_dim=32, normalize=True)
        import awkward as ak

        hits = np.arange(10, dtype=np.float32)
        batch = ak.Array(
            {
                "hit_integral": [hits],
                "hit_charge": [hits],
                "hit_amplitude": [hits],
            }
        )

        with patch("sbn_anomaly.data.stream_dataset.RootStreamer") as MockStreamer:
            MockStreamer.return_value.stream.return_value = iter([batch])
            (tensor,) = list(ds)[0]

        arr = tensor.numpy()
        assert abs(arr.std() - 1.0) < 1e-4, f"Expected std≈1, got {arr.std()}"


# ---------------------------------------------------------------------------
# CLI argument parsing – verify --root-files is accepted
# ---------------------------------------------------------------------------


class TestCLIRootFilesArg:
    def test_root_files_accepted_by_parser(self):
        """The CLI must parse --root-files without error."""
        import argparse

        # Re-create just the argument subset that matters.
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        parser.add_argument("--root-files", nargs="+", default=None)

        args = parser.parse_args(
            ["--config", "configs/tpc.yaml", "--root-files", "a.root", "b.root"]
        )
        assert args.root_files == ["a.root", "b.root"]

    def test_root_files_optional(self):
        """--root-files must be optional; omitting it should leave it as None."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        parser.add_argument("--root-files", nargs="+", default=None)

        args = parser.parse_args(["--config", "configs/tpc.yaml"])
        assert args.root_files is None

    def test_main_parses_root_files(self, tmp_path):
        """main() must forward --root-files to _train_tpc without crashing on
        argument parsing itself.  We short-circuit before actual training."""
        from sbn_anomaly.train.cli import main

        # Write a minimal valid config so the config-loading step passes.
        cfg = tmp_path / "tpc_test.yaml"
        cfg.write_text(
            "model_type: tpc\n"
            "model: {input_dim: 8, latent_dim: 2}\n"
            "data: {features_path: missing.npy, tree_name: t}\n"
            "training: {max_epochs: 1, batch_size: 4, steps_per_epoch: 1}\n"
        )

        # Patch _train_tpc so training doesn't actually run.
        with patch("sbn_anomaly.train.cli._train_tpc") as mock_train:
            ret = main(
                ["--config", str(cfg), "--root-files", "run1.root", "run2.root"]
            )

        assert ret == 0
        mock_train.assert_called_once()
        _, kwargs = mock_train.call_args
        assert kwargs["root_files"] == ["run1.root", "run2.root"]
