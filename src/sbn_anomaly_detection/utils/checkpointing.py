"""Checkpoint save/load helpers."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: str,
    filename: str = "checkpoint.pt",
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save model + optimiser state to disk.

    Parameters
    ----------
    model:
        The PyTorch model to checkpoint.
    optimizer:
        The optimiser whose state should be saved.
    epoch:
        Current training epoch (0-indexed).
    loss:
        Validation loss at this checkpoint.
    checkpoint_dir:
        Directory where checkpoints are written.
    filename:
        File name for the checkpoint.
    extra:
        Any additional key/value pairs to store alongside the checkpoint.

    Returns
    -------
    Path
        Absolute path to the saved checkpoint file.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / filename

    state = {
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra:
        state.update(extra)

    torch.save(state, path)
    logger.info("Checkpoint saved → %s  (epoch=%d, loss=%.6f)", path, epoch, loss)
    return path


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load a checkpoint into *model* (and optionally *optimizer*).

    Parameters
    ----------
    model:
        Model instance whose ``state_dict`` will be replaced.
    path:
        Path to the ``.pt`` checkpoint file.
    optimizer:
        If provided, restores the optimiser state as well.
    device:
        Device to map tensors to. Defaults to the current model device.

    Returns
    -------
    dict
        Full checkpoint dict (contains ``epoch``, ``loss``, etc.).
    """
    map_location = device or next(model.parameters()).device
    state = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    logger.info(
        "Checkpoint loaded ← %s  (epoch=%d, loss=%.6f)",
        path,
        state.get("epoch", -1),
        state.get("loss", float("nan")),
    )
    return state


def best_checkpoint_path(checkpoint_dir: str, prefix: str = "best") -> Optional[Path]:
    """Return the path of the best checkpoint in *checkpoint_dir*, or None."""
    ckpt_dir = Path(checkpoint_dir)
    candidates = sorted(ckpt_dir.glob(f"{prefix}*.pt"))
    return candidates[0] if candidates else None
