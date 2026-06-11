"""Batch anomaly scorer: loads a trained model and scores new events.

``AnomalyScorer`` wraps any of the four model types and exposes a unified
``score()`` interface.  It also provides a streaming variant
``score_stream()`` that consumes a :class:`~sbn_anomaly.data.RootStreamer`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Optional, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GNNScorer:
    """Per-node anomaly scorer for GNNForecasterPyG.

    Runs inference over a PyG DataLoader and maps node-level prediction errors
    back to the original channel grid, so each output window has a score per
    channel rather than a single graph-level score.

    Inactive channels (pruned for a given window) receive NaN scores so
    downstream code can distinguish "not hit" from "low anomaly score".

    Parameters
    ----------
    model:
        Trained GNNForecasterPyG instance.
    num_channels:
        Total number of channels in the original (unpruned) detector graph.
    device:
        Compute device (``'auto'``, ``'cpu'``, or ``'cuda'``).
    threshold:
        If set, per-window mean score above this value is flagged as anomalous.
    """

    def __init__(
        self,
        model: nn.Module,
        num_channels: int,
        device: str = "auto",
        threshold: Optional[float] = None,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.num_channels = int(num_channels)
        self.threshold = threshold

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        model: nn.Module,
        num_channels: int,
        device: str = "auto",
        threshold: Optional[float] = None,
    ) -> "GNNScorer":
        """Load model weights from a checkpoint and return a GNNScorer."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        state = torch.load(str(checkpoint_path), map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        logger.info("Loaded GNN weights from %s", checkpoint_path)
        return cls(model=model, num_channels=num_channels, device=device, threshold=threshold)

    @torch.no_grad()
    def score_loader(
        self,
        loader,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Score all windows in a DataLoader in iteration order.

        Pass a non-shuffled loader to preserve temporal (window index) ordering.

        Parameters
        ----------
        loader:
            torch_geometric DataLoader over a GraphWindowDatasetPyG.

        Returns
        -------
        node_scores:
            (N_windows, num_channels) float32 — per-channel MSE; NaN = inactive
        window_scores_mean:
            (N_windows,) float32 — mean active-node MSE per window
        window_scores_max:
            (N_windows,) float32 — max active-node MSE per window
        """
        from tqdm import tqdm

        node_scores_list: list[np.ndarray] = []
        window_mean_list: list[float] = []
        window_max_list: list[float] = []

        for batch in tqdm(loader, desc="Scoring windows", unit="batch"):
            data = batch.to(self.device)
            pred = self.model(data)

            per_node_mse = ((pred - data.y.float()) ** 2).mean(dim=-1).cpu()  # (total_nodes,)
            graph_idx = data.batch.cpu()      # (total_nodes,)
            active_mask = data.active_mask.cpu()  # (total_nodes,) original channel indices

            for g in range(int(data.num_graphs)):
                node_sel = graph_idx == g
                g_mse = per_node_mse[node_sel].numpy()
                g_channels = active_mask[node_sel].numpy()

                # Scatter into full-channel array; inactive channels stay NaN
                full = np.full(self.num_channels, np.nan, dtype=np.float32)
                full[g_channels] = g_mse
                node_scores_list.append(full)

                window_mean_list.append(float(g_mse.mean()))
                window_max_list.append(float(g_mse.max()))

        if not node_scores_list:
            empty = np.zeros((0, self.num_channels), dtype=np.float32)
            return empty, np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        node_scores = np.stack(node_scores_list)                            # (N, C)
        window_scores_mean = np.array(window_mean_list, dtype=np.float32)  # (N,)
        window_scores_max = np.array(window_max_list,  dtype=np.float32)   # (N,)
        return node_scores, window_scores_mean, window_scores_max


class AnomalyScorer:
    """Compute reconstruction-based anomaly scores from a trained model.

    Parameters
    ----------
    model:
        Trained PyTorch autoencoder.
    model_type:
        One of ``'tpc'``, ``'pmt'``, ``'fusion'``, ``'window'``.
    device:
        Compute device (``'auto'``, ``'cpu'``, or ``'cuda'``).
    threshold:
        If set, :meth:`score` also returns a boolean anomaly flag per sample.
    """

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        device: str = "auto",
        threshold: Optional[float] = None,
        normalize: bool = False,
    ) -> None:
        if model_type not in ("tpc", "pmt", "fusion", "window", "gnn"):
            raise ValueError(
                f"model_type must be one of tpc/pmt/fusion/window/gnn, got '{model_type}'."
            )
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.model_type = model_type
        self.threshold = threshold
        self.normalize = bool(normalize)

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        model: nn.Module,
        model_type: str,
        device: str = "auto",
        threshold: Optional[float] = None,
        normalize: bool = False,
    ) -> "AnomalyScorer":
        """Load model weights from a checkpoint and return an ``AnomalyScorer``."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        state = torch.load(str(checkpoint_path), map_location=device)
        # Support both raw state-dict and training checkpoint formats.
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        logger.info("Loaded weights from %s", checkpoint_path)
        return cls(model=model, model_type=model_type, device=device, threshold=threshold, normalize=normalize)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score(
        self,
        *arrays: np.ndarray,
    ) -> np.ndarray:
        """Compute per-sample anomaly scores.

        Parameters
        ----------
        arrays:
            For ``tpc``, ``pmt``, ``window``: a single NumPy array of shape
            ``(N, feature_dim)`` or ``(N, channels, length)``.
            For ``fusion``: two arrays ``(tpc_features, pmt_features)``.

        Returns
        -------
        scores:
            1-D array of shape (N,) with per-sample reconstruction MSE.
        """
        def _normalize(x: np.ndarray) -> np.ndarray:
            """Apply per-sample z-score normalization."""
            if not self.normalize:
                return x
            # Compute per-sample mean and std
            mean = x.mean(axis=-1, keepdims=True)
            std = x.std(axis=-1, keepdims=True)
            # Avoid division by zero
            std = np.where(std > 0, std, 1.0)
            return (x - mean) / std
        
        if self.model_type in ("tpc", "pmt"):
            x = arrays[0].astype(np.float32)
            x = _normalize(x)
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
            x_hat, _ = self.model(x)
            scores = ((x - x_hat) ** 2).mean(dim=-1).cpu().numpy()

        elif self.model_type == "window":
            x = arrays[0].astype(np.float32)
            x = _normalize(x)
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            x_hat, _ = self.model(x)
            scores = ((x - x_hat) ** 2).mean(dim=(-2, -1)).cpu().numpy()

        elif self.model_type == "fusion":
            x_tpc = arrays[0].astype(np.float32)
            x_pmt = arrays[1].astype(np.float32)
            x_tpc = _normalize(x_tpc)
            x_pmt = _normalize(x_pmt)
            x_tpc = torch.tensor(x_tpc, dtype=torch.float32).to(self.device)
            x_pmt = torch.tensor(x_pmt, dtype=torch.float32).to(self.device)
            recon, _, combined = self.model(x_tpc, x_pmt)
            scores = ((combined - recon) ** 2).mean(dim=-1).cpu().numpy()

        elif self.model_type == "gnn":
            # Expect arrays: past_windows (N, T, N_nodes, F), target_next (N, N_nodes, F)
            past = arrays[0].astype(np.float32)
            if len(arrays) < 2:
                raise ValueError("GNN scorer requires both past_windows and target_next arrays for scoring")
            target = arrays[1].astype(np.float32)
            past = _normalize(past)
            target = _normalize(target)
            past_t = torch.tensor(past, dtype=torch.float32).to(self.device)
            target_t = torch.tensor(target, dtype=torch.float32).to(self.device)
            # Model forward: predict next window
            with torch.no_grad():
                pred = self.model(past_t, torch.tensor(np.asarray(arrays[2]) if len(arrays) > 2 else np.eye(past.shape[2]), dtype=torch.float32).to(self.device))
                # per-node MSE
                per_node_mse = ((pred - target_t) ** 2).mean(dim=-1)
                # return window-level mean MSE
                scores = per_node_mse.mean(dim=-1).cpu().numpy()
        else:
            raise RuntimeError(f"Unhandled model_type: {self.model_type}")

        return scores

    def is_anomaly(self, *arrays: np.ndarray) -> np.ndarray:
        """Return boolean array; ``True`` where score exceeds ``threshold``.

        Raises ``ValueError`` if ``threshold`` was not set at construction.
        """
        if self.threshold is None:
            raise ValueError("Set a threshold at construction to use is_anomaly().")
        return self.score(*arrays) > self.threshold

    def score_stream(
        self,
        streamer,  # RootStreamer – avoid circular import by duck-typing
        feature_branch: str,
        batch_size: int = 512,
    ) -> Generator[np.ndarray, None, None]:
        """Yield anomaly score arrays, one per streamed batch.

        Parameters
        ----------
        streamer:
            A :class:`~sbn_anomaly.data.RootStreamer` instance.
        feature_branch:
            Branch name to extract from each batch (e.g. ``'tpc_waveform'``).
        batch_size:
            Batch size passed to the model.
        """
        import awkward as ak

        for batch in streamer.stream():
            features = ak.to_numpy(ak.flatten(batch[feature_branch], axis=None))
            # Reshape assuming flat features per event.
            n_events = len(batch)
            feature_dim = features.size // n_events
            features = features.reshape(n_events, feature_dim)
            scores = self.score(features)
            yield scores
