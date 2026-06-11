"""Base trainer class with common training loop logic."""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from csv import DictWriter
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Abstract base trainer providing a standard training loop.

    Subclasses must implement :meth:`compute_loss`.

    Parameters
    ----------
    model:
        PyTorch model to train.
    optimizer:
        Optimiser instance.
    device:
        ``'cuda'``, ``'cpu'``, or ``'auto'``.
    max_epochs:
        Number of training epochs.
    checkpoint_dir:
        Directory to save model checkpoints.  ``None`` disables saving.
    log_interval:
        Log training loss every N batches.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "auto",
        max_epochs: int = 50,
        checkpoint_dir: Optional[str] = None,
        log_interval: int = 50,
        steps_per_epoch: Optional[int] = None,
        enable_progress_bar: bool = True,
        anomaly_threshold: Optional[float] = None,
        reconstruction_plot_max_values: int = 50000,
        save_best_only: bool = False,
        use_amp: bool = False,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        logger.info("Using training device: %s", self.device)
        self.max_epochs = max_epochs
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.log_interval = log_interval
        self.steps_per_epoch = steps_per_epoch
        self.enable_progress_bar = bool(enable_progress_bar and tqdm is not None)
        self.anomaly_threshold = anomaly_threshold
        self.reconstruction_plot_max_values = max(int(reconstruction_plot_max_values), 0)
        self.save_best_only = bool(save_best_only)
        self.best_loss: Optional[float] = None
        self.use_amp = bool(use_amp) and self.device.type == "cuda"
        self._scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        logger.info("Mixed precision (AMP): %s", "enabled" if self.use_amp else "disabled")

        # Training history populated by `train()`.
        # Keys are metric names, values are per-epoch lists.
        self.history: dict[str, list[Any]] = {}
        self._reconstruction_plot_data: dict[str, list[torch.Tensor]] = {
            "original": [],
            "reconstruction": [],
        }
        self._reconstruction_feature_names: Optional[list[str]] = None
        self._reconstruction_n_variables: Optional[int] = None
        self._tpc_branch_values: Optional[np.ndarray] = None
        # Keep a reference to the dataset used for training (if available)
        self._training_dataset: Optional[object] = None

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def compute_loss(self, batch: tuple) -> torch.Tensor:
        """Compute scalar loss for one batch."""

    def compute_scores(self, batch: tuple) -> Optional[torch.Tensor]:
        """Return per-sample anomaly scores for *batch*, if supported.

        The default implementation returns ``None``. Subclasses should
        override when they can compute reconstruction error per sample.
        """

        return None

    def compute_reconstruction_pair(
        self,
        batch: tuple,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Return (original, reconstruction) tensors for plotting, if supported."""
        return None

    def _append_history(self, key: str, value: Any) -> None:
        self.history.setdefault(key, []).append(value)

    def _infer_batch_size(self, batch: tuple) -> int:
        """Best-effort extraction of batch size for throughput metrics."""
        if not batch:
            return 0
        first = batch[0]
        if torch.is_tensor(first) and first.dim() > 0:
            return int(first.shape[0])
        try:
            return int(len(first))
        except Exception:
            return 0
    def _compute_classification_metrics(
        self,
        scores: list[float],
        labels: list[int],
    ) -> dict[str, float]:
        """Compute precision/recall/F1 (+AUC) using a threshold that maximizes F1."""
        import numpy as np

        try:
            from sklearn.metrics import roc_auc_score
        except Exception:
            roc_auc_score = None  # type: ignore[assignment]

        scores_arr = np.asarray(scores, dtype=np.float64)
        labels_arr = np.asarray(labels, dtype=np.int64)

        # Degenerate cases: avoid crashing when a batch/epoch has only one class.
        if labels_arr.size == 0 or np.unique(labels_arr).size < 2:
            return {
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "auc": float("nan"),
                "threshold": float("nan"),
            }

        # Find threshold that maximizes F1 over candidate score thresholds.
        # We use sklearn's precision_recall_curve thresholds (monotonic in score).
        from sklearn.metrics import precision_recall_curve

        precision, recall, thresholds = precision_recall_curve(labels_arr, scores_arr)
        # precision/recall have length thresholds+1; align F1 to thresholds.
        f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
        best_idx = int(np.nanargmax(f1)) if f1.size else 0
        best_thr = float(thresholds[best_idx]) if thresholds.size else float("nan")

        # Metrics at best threshold.
        y_pred = (scores_arr >= best_thr).astype(np.int64)
        tp = int(((y_pred == 1) & (labels_arr == 1)).sum())
        fp = int(((y_pred == 1) & (labels_arr == 0)).sum())
        fn = int(((y_pred == 0) & (labels_arr == 1)).sum())

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1_best = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

        auc = float("nan")
        if roc_auc_score is not None:
            try:
                auc = float(roc_auc_score(labels_arr, scores_arr))
            except Exception:
                auc = float("nan")

        return {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1_best),
            "auc": float(auc),
            "threshold": float(best_thr),
        }

    def _evaluate_loader_loss(self, loader: DataLoader) -> float:
        """Return the mean loss over *loader* without updating model weights."""
        was_training = self.model.training
        self.model.eval()
        running_loss = 0.0
        n_batches = 0
        try:
            with torch.no_grad():
                for batch in loader:
                    loss = self.compute_loss(batch)
                    running_loss += float(loss.item())
                    n_batches += 1
        finally:
            if was_training:
                self.model.train()
        return running_loss / max(n_batches, 1)

    def train(
        self,
        loader: DataLoader,
        *,
        validation_loader: Optional[DataLoader] = None,
        metrics_max_samples: int = 20000,
    ) -> list[float]:
        """Run training for ``max_epochs`` epochs.

        Parameters
        ----------
        loader:
            DataLoader yielding batches consumed by :meth:`compute_loss`.

        Returns
        -------
        epoch_losses:
            Mean loss per epoch.
        """
        self.model.train()
        epoch_losses: list[float] = []

        # Reset history and best_loss on each train() call.
        self.history = {
            "epoch": [],
            "loss": [],
            **({"val_loss": []} if validation_loader is not None else {}),
            "score_p95": [],
            "score_p99": [],
            "anomaly_fraction_above_threshold": [],
            "anomaly_threshold": [],
            "epoch_time_sec": [],
            "events_per_sec": [],
        }
        self._reconstruction_plot_data = {"original": [], "reconstruction": []}
        self._reconstruction_feature_names = None
        self._reconstruction_n_variables = None
        self._tpc_branch_values = None

        dataset = getattr(loader, "dataset", None)
        # Unwrap Subset (random_split) to the underlying dataset so metadata
        # like `hit_branches` is discoverable.
        try:
            from torch.utils.data import Subset
        except Exception:
            Subset = None
        underlying_dataset = dataset
        if Subset is not None and isinstance(dataset, Subset):
            try:
                underlying_dataset = dataset.dataset
            except Exception:
                underlying_dataset = dataset

        # Remember dataset for later plotting/inspection.
        self._training_dataset = underlying_dataset
        logger.info(
            "Training dataset type=%s hit_branches=%s waveform_branch=%s",
            type(underlying_dataset),
            getattr(underlying_dataset, "hit_branches", None),
            getattr(underlying_dataset, "waveform_branch", None),
        )

        hit_branches = getattr(underlying_dataset, "hit_branches", None)
        waveform_branch = getattr(underlying_dataset, "waveform_branch", None)
        tpc_branch_values = getattr(underlying_dataset, "tpc_branch_values", None)
        
        if (
            waveform_branch is None
            and isinstance(hit_branches, (list, tuple))
            and len(hit_branches) > 0
        ):
            self._reconstruction_feature_names = [str(branch) for branch in hit_branches]
            self._reconstruction_n_variables = len(self._reconstruction_feature_names)
        
        # Store tpc_branch_values for saving in inference output
        if tpc_branch_values is not None:
            self._tpc_branch_values = tpc_branch_values

        self.best_loss = None

        def _to_2d_batch(tensor: torch.Tensor) -> torch.Tensor:
            tensor = tensor.detach().to("cpu").float()
            if tensor.ndim == 0:
                return tensor.reshape(1, 1)
            if tensor.ndim == 1:
                return tensor.unsqueeze(0)
            return tensor.reshape(tensor.shape[0], -1)

        def _row_count(chunks: list[torch.Tensor]) -> int:
            return sum(int(chunk.shape[0]) for chunk in chunks)

        processed_any = False
        for epoch in range(1, self.max_epochs + 1):
            epoch_t0 = time.perf_counter()
            running_loss = 0.0
            n_batches = 0
            n_events = 0

            # Optional: capture (score, label) pairs to compute epoch-level metrics.
            score_buf: list[float] = []
            label_buf: list[int] = []
            epoch_recon_data: dict[str, list[torch.Tensor]] = {
                "original": [],
                "reconstruction": [],
            }

            epoch_total = self.steps_per_epoch
            if epoch_total is None:
                try:
                    epoch_total = len(loader)
                except TypeError:
                    epoch_total = None

            pbar = None
            if self.enable_progress_bar:
                pbar = tqdm(
                    total=epoch_total,
                    desc=f"Epoch {epoch}/{self.max_epochs}",
                    unit="batch",
                    dynamic_ncols=True,
                    leave=False,
                )

            for batch_idx, batch in enumerate(loader):
                if self.steps_per_epoch is not None and batch_idx >= self.steps_per_epoch:
                    break
                self.optimizer.zero_grad()
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        loss = self.compute_loss(batch)
                else:
                    loss = self.compute_loss(batch)
                # Fail fast on non-finite loss to aid debugging (bad input/NaNs).
                try:
                    loss_val = float(loss.item())
                except Exception:
                    logger.exception("Failed to materialize loss.item() — aborting training")
                    raise
                if not np.isfinite(loss_val):
                    logger.error(
                        "Non-finite loss detected at epoch %d batch %d (loss=%s)."
                        " Check input data, branches, and preprocessing.",
                        epoch,
                        batch_idx,
                        loss_val,
                    )
                    # Surface the problem immediately rather than producing NaN history.
                    raise RuntimeError(f"Non-finite loss detected: {loss_val}")
                if self.use_amp:
                    self._scaler.scale(loss).backward()
                    self._scaler.step(self.optimizer)
                    self._scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item()
                n_batches += 1
                n_events += self._infer_batch_size(batch)

                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(loss=f"{loss.item():.6f}", refresh=False)

                if (batch_idx + 1) % self.log_interval == 0:
                    logger.info(
                        "Epoch %d/%d  batch %d  loss=%.6f",
                        epoch,
                        self.max_epochs,
                        batch_idx + 1,
                        loss.item(),
                    )

                # Collect score distribution statistics for unsupervised metrics.
                if metrics_max_samples > 0 and len(score_buf) < metrics_max_samples:
                    try:
                        scores_t = self.compute_scores(batch)
                        if scores_t is not None:
                            scores_t = scores_t.detach().to("cpu").float().view(-1)
                            n_keep = min(
                                int(metrics_max_samples - len(score_buf)),
                                int(scores_t.numel()),
                            )
                            if n_keep > 0:
                                score_buf.extend(scores_t[:n_keep].tolist())

                                # Optionally collect labels when present and binary.
                                if len(batch) >= 2:
                                    labels_t = batch[-1]
                                    if torch.is_tensor(labels_t):
                                        labels_t = labels_t.detach().to("cpu")
                                        if labels_t.dim() == 2 and labels_t.shape[1] == 1:
                                            labels_t = labels_t.view(-1)
                                        if labels_t.dim() == 1 and labels_t.numel() >= n_keep:
                                            labels_t = labels_t.long()
                                            uniq = torch.unique(labels_t)
                                            if uniq.numel() <= 2:
                                                label_buf.extend(labels_t[:n_keep].tolist())
                    except Exception:
                        # Metrics collection must never break training.
                        pass

                # Save sample original/reconstruction values for 2D histogram plotting.
                if self.reconstruction_plot_max_values > 0:
                    remaining_epoch = self.reconstruction_plot_max_values - _row_count(
                        epoch_recon_data["original"]
                    )
                    if remaining_epoch > 0:
                        try:
                            pair = self.compute_reconstruction_pair(batch)
                            if pair is not None:
                                x, x_hat = pair
                                x_2d = _to_2d_batch(x)
                                x_hat_2d = _to_2d_batch(x_hat)
                                n_keep = min(
                                    remaining_epoch,
                                    int(x_2d.shape[0]),
                                    int(x_hat_2d.shape[0]),
                                )
                                if n_keep > 0:
                                    epoch_recon_data["original"].append(
                                        x_2d[:n_keep].clone()
                                    )
                                    epoch_recon_data["reconstruction"].append(
                                        x_hat_2d[:n_keep].clone()
                                    )

                                    # Keep an aggregate sample for the final summary histogram.
                                    remaining_global = self.reconstruction_plot_max_values - _row_count(
                                        self._reconstruction_plot_data["original"]
                                    )
                                    if remaining_global > 0:
                                        n_keep_global = min(remaining_global, n_keep)
                                        self._reconstruction_plot_data["original"].append(
                                            x_2d[:n_keep_global].clone()
                                        )
                                        self._reconstruction_plot_data["reconstruction"].append(
                                            x_hat_2d[:n_keep_global].clone()
                                        )
                        except Exception:
                            pass

            if pbar is not None:
                pbar.close()

            mean_loss = running_loss / max(n_batches, 1)
            if n_batches > 0:
                processed_any = True
            epoch_losses.append(mean_loss)
            logger.info("Epoch %d/%d  mean_loss=%.6f", epoch, self.max_epochs, mean_loss)

            epoch_time = time.perf_counter() - epoch_t0
            events_per_sec = n_events / epoch_time if epoch_time > 0 else float("nan")

            if score_buf:
                scores_arr = np.asarray(score_buf, dtype=np.float64)
                score_p95 = float(np.percentile(scores_arr, 95))
                score_p99 = float(np.percentile(scores_arr, 99))
                if self.anomaly_threshold is not None:
                    anomaly_fraction = float(np.mean(scores_arr >= self.anomaly_threshold))
                else:
                    anomaly_fraction = float("nan")
            else:
                score_p95 = float("nan")
                score_p99 = float("nan")
                anomaly_fraction = float("nan")

            self._append_history("epoch", epoch)
            self._append_history("loss", float(mean_loss))
            if validation_loader is not None:
                val_loss = self._evaluate_loader_loss(validation_loader)
                self._append_history("val_loss", float(val_loss))
            self._append_history("score_p95", score_p95)
            self._append_history("score_p99", score_p99)
            self._append_history(
                "anomaly_fraction_above_threshold",
                anomaly_fraction,
            )
            self._append_history(
                "anomaly_threshold",
                float(self.anomaly_threshold) if self.anomaly_threshold is not None else float("nan"),
            )
            self._append_history("epoch_time_sec", float(epoch_time))
            self._append_history("events_per_sec", float(events_per_sec))

            if score_buf and label_buf:
                metrics = self._compute_classification_metrics(score_buf, label_buf)
                for k, v in metrics.items():
                    self._append_history(k, v)

            if validation_loader is not None:
                logger.info(
                    "Epoch %d/%d  val_loss=%.6f",
                    epoch,
                    self.max_epochs,
                    self.history["val_loss"][-1],
                )

            # Determine if we should save checkpoint and plots this epoch.
            is_best = self.best_loss is None or mean_loss < self.best_loss
            is_final = epoch == self.max_epochs
            should_save_epoch_artifacts = (not self.save_best_only) or is_best or is_final

            if is_best and self.best_loss is not None:
                logger.info("New best loss: %.6f (was %.6f)", mean_loss, self.best_loss)
            if is_best:
                self.best_loss = mean_loss

            if self.checkpoint_dir:
                if should_save_epoch_artifacts:
                    weights_dir = self.checkpoint_dir / "weights"
                    weights_dir.mkdir(parents=True, exist_ok=True)
                    ckpt_path = weights_dir / f"epoch_{epoch:04d}.pt"
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "loss": mean_loss,
                        },
                        ckpt_path,
                    )
                    logger.debug("Saved checkpoint: %s", ckpt_path)

        # If we reached the end without processing any batches, warn the user.
        if not processed_any:
            logger.warning(
                "Training completed without processing any batches. Check data loader and input files."
            )
        return epoch_losses

    def save_training_history(
        self,
        output_dir: Optional[str | Path] = None,
        filename: str = "training_history.csv",
    ) -> Optional[Path]:
        """Save per-epoch metrics history to CSV."""
        out_dir = Path(output_dir) if output_dir is not None else None
        if out_dir is None:
            out_dir = self.checkpoint_dir if self.checkpoint_dir else Path(os.getcwd())
        out_dir.mkdir(parents=True, exist_ok=True)

        if not self.history or "epoch" not in self.history:
            return None

        preferred_order = [
            "epoch",
            "loss",
            "val_loss",
            "score_p95",
            "score_p99",
            "anomaly_fraction_above_threshold",
            "anomaly_threshold",
            "epoch_time_sec",
            "events_per_sec",
            "precision",
            "recall",
            "f1",
            "auc",
            "threshold",
        ]
        keys = [k for k in preferred_order if k in self.history] + [
            k for k in self.history if k not in preferred_order
        ]

        n_rows = max((len(v) for v in self.history.values()), default=0)
        out_path = out_dir / filename
        with out_path.open("w", newline="") as fh:
            writer = DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for i in range(n_rows):
                row = {
                    k: (self.history[k][i] if i < len(self.history[k]) else "")
                    for k in keys
                }
                writer.writerow(row)
        return out_path

    def save_training_plots(self, output_dir: Optional[str | Path] = None, bins: Any | None = None) -> Optional[Path]:
        """Save training curves (loss + optional metrics) as a PNG.

        Parameters
        ----------
        output_dir:
            Directory to write plots to. Defaults to ``checkpoint_dir`` if set,
            else the current working directory.

        Returns
        -------
        Path to the generated PNG, or ``None`` if plotting is unavailable.
        """
        from sbn_anomaly.utils.plotting import save_reconstruction_hist2d, save_training_curves

        out_dir = Path(output_dir) if output_dir is not None else None
        if out_dir is None:
            out_dir = self.checkpoint_dir if self.checkpoint_dir else Path(os.getcwd())
        out_dir.mkdir(parents=True, exist_ok=True)

        curve_path = None
        # Save curve PNG first — don't let reconstruction hist failures stop it.
        try:
            curve_path = save_training_curves(self.history, out_dir)
        except Exception as exc:
            logger.warning("Failed to save training curves: %s", exc)

        # Attempt reconstruction histograms separately; log but don't fail the
        # overall plotting when they fail.
        try:
            orig = self._reconstruction_plot_data.get("original") or []
            recon = self._reconstruction_plot_data.get("reconstruction") or []
            logger.info("Reconstruction plot data sizes: original=%d reconstruction=%d", 
                        sum(int(x.shape[0]) for x in orig) if orig else 0,
                        sum(int(x.shape[0]) for x in recon) if recon else 0)
            logger.info("Reconstruction feature names (before recover): %s", self._reconstruction_feature_names)
            logger.info("Reconstruction n_variables (before recover): %s", self._reconstruction_n_variables)
            # If feature names weren't populated at train() start, try to
            # recover them now from the training dataset metadata.
            if self._reconstruction_feature_names is None and self._training_dataset is not None:
                hb = getattr(self._training_dataset, "hit_branches", None)
                wf = getattr(self._training_dataset, "waveform_branch", None)
                if wf is None and isinstance(hb, (list, tuple)) and len(hb) > 0:
                    self._reconstruction_feature_names = [str(b) for b in hb]
                    self._reconstruction_n_variables = len(self._reconstruction_feature_names)
                    logger.info("Recovered reconstruction feature names from training dataset: %s", self._reconstruction_feature_names)
                    logger.info("Recovered reconstruction n_variables: %s", self._reconstruction_n_variables)

            # Put reconstruction/epoch histograms under a dedicated `plots/` dir.
            plots_dir = out_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            if orig and recon:
                try:
                    save_reconstruction_hist2d(
                        orig,
                        recon,
                        plots_dir,
                        feature_names=self._reconstruction_feature_names,
                        n_variables=self._reconstruction_n_variables,
                        bins=bins,
                    )
                except Exception as exc2:
                    logger.warning("Failed to save reconstruction hist2d: %s", exc2)
        except Exception as exc:
            logger.warning("Unexpected error when preparing reconstruction hist: %s", exc)

        return curve_path

    def save(self, path: str) -> None:
        """Save model weights to *path*."""
        torch.save(self.model.state_dict(), path)
        logger.info("Model weights saved to %s", path)

    def load(self, path: str) -> None:
        """Load model weights from *path*."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        logger.info("Model weights loaded from %s", path)
