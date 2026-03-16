"""Feature normalisation utilities (fit on training data, apply everywhere).

Provides a ``Normalizer`` class that stores mean and std computed from a
numpy array and exposes both numpy and torch-tensor transform methods.
Serialised to disk as a plain ``.npz`` file so it can be reloaded without
pickling issues across Python versions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

_EPS = 1e-8  # numerical stability floor


class Normalizer:
    """Z-score normaliser: transform x → (x - mean) / (std + eps).

    Parameters
    ----------
    mean:
        Per-feature mean array of shape ``(n_features,)``.
    std:
        Per-feature standard deviation array of shape ``(n_features,)``.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def fit(cls, data: np.ndarray) -> "Normalizer":
        """Compute mean and std from *data* (shape ``(N, F)``)."""
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        logger.debug(
            "Normalizer.fit: mean in [%.4f, %.4f], std in [%.4f, %.4f]",
            mean.min(),
            mean.max(),
            std.min(),
            std.max(),
        )
        return cls(mean, std)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Normalizer":
        """Load a previously saved normaliser from a ``.npz`` file."""
        data = np.load(path)
        obj = cls(data["mean"], data["std"])
        logger.info("Normalizer loaded ← %s", path)
        return obj

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save mean and std to *path* as a ``.npz`` archive."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, mean=self.mean, std=self.std)
        logger.info("Normalizer saved → %s", out)

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply z-score normalisation (returns float32 array)."""
        return ((x - self.mean) / (self.std + _EPS)).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Reverse z-score normalisation."""
        return (x * (self.std + _EPS) + self.mean).astype(np.float32)

    def transform_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalisation to a torch tensor (returns same device)."""
        mean_t = torch.from_numpy(self.mean).to(x.device)
        std_t = torch.from_numpy(self.std).to(x.device)
        return (x - mean_t) / (std_t + _EPS)

    def inverse_transform_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse normalisation on a torch tensor."""
        mean_t = torch.from_numpy(self.mean).to(x.device)
        std_t = torch.from_numpy(self.std).to(x.device)
        return x * (std_t + _EPS) + mean_t
