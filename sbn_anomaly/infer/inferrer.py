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
    ) -> None:
        if model_type not in ("tpc", "pmt", "fusion", "window"):
            raise ValueError(
                f"model_type must be one of tpc/pmt/fusion/window, got '{model_type}'."
            )
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.model_type = model_type
        self.threshold = threshold

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
        return cls(model=model, model_type=model_type, device=device, threshold=threshold)

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
        if self.model_type in ("tpc", "pmt"):
            x = torch.tensor(arrays[0], dtype=torch.float32).to(self.device)
            x_hat, _ = self.model(x)
            scores = ((x - x_hat) ** 2).mean(dim=-1).cpu().numpy()

        elif self.model_type == "window":
            x = torch.tensor(arrays[0], dtype=torch.float32).to(self.device)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            x_hat, _ = self.model(x)
            scores = ((x - x_hat) ** 2).mean(dim=(-2, -1)).cpu().numpy()

        elif self.model_type == "fusion":
            x_tpc = torch.tensor(arrays[0], dtype=torch.float32).to(self.device)
            x_pmt = torch.tensor(arrays[1], dtype=torch.float32).to(self.device)
            recon, _, combined = self.model(x_tpc, x_pmt)
            scores = ((combined - recon) ** 2).mean(dim=-1).cpu().numpy()

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
