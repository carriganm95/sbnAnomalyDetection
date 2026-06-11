"""Multi-branch anomaly detection combining TPC, PMT, and Window models.

This module scores events using three independent branches and combines them
using a fusion rule that triggers an alert if ANY branch exceeds its threshold.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


class MultiBranchScorer:
    """Loads and scores three independent anomaly detection branches.
    
    Attributes:
        tpc_scorer: AnomalyScorer for TPC branch
        pmt_scorer: AnomalyScorer for PMT branch
        window_scorer: AnomalyScorer for Window branch
        tpc_threshold: Alert threshold for TPC scores
        pmt_threshold: Alert threshold for PMT scores
        window_threshold: Alert threshold for Window scores
    """

    def __init__(
        self,
        tpc_scorer: AnomalyScorer,
        pmt_scorer: AnomalyScorer,
        window_scorer: AnomalyScorer,
        tpc_threshold: float | None = None,
        pmt_threshold: float | None = None,
        window_threshold: float | None = None,
    ):
        self.tpc_scorer = tpc_scorer
        self.pmt_scorer = pmt_scorer
        self.window_scorer = window_scorer
        self.tpc_threshold = tpc_threshold
        self.pmt_threshold = pmt_threshold
        self.window_threshold = window_threshold

    def score(
        self,
        tpc_features: np.ndarray,
        pmt_features: np.ndarray | None = None,
        window_features: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Score all three branches independently.
        
        Args:
            tpc_features: (n_samples, tpc_dim) array of TPC features
            pmt_features: (n_samples, pmt_dim) array of PMT features (if available)
            window_features: (n_samples, window_dim) array of window features (if available)
            
        Returns:
            Dictionary with keys:
                - 'tpc_scores': (n_samples,) TPC anomaly scores
                - 'pmt_scores': (n_samples,) PMT anomaly scores (or zeros if not provided)
                - 'window_scores': (n_samples,) Window anomaly scores (or zeros if not provided)
                - 'max_scores': (n_samples,) Element-wise max across branches
                - 'alert_flags': (n_samples,) Boolean indicating if any branch exceeds threshold
        """
        # Score TPC branch (required)
        tpc_scores = self.tpc_scorer.score(tpc_features)
        
        # Score PMT branch if provided
        if pmt_features is not None:
            pmt_scores = self.pmt_scorer.score(pmt_features)
        else:
            pmt_scores = np.zeros_like(tpc_scores)
            
        # Score Window branch if provided
        if window_features is not None:
            window_scores = self.window_scorer.score(window_features)
        else:
            window_scores = np.zeros_like(tpc_scores)
        
        # Combine branches: alert if any exceeds threshold
        max_scores = np.maximum(np.maximum(tpc_scores, pmt_scores), window_scores)
        
        # Build alert flags using OR logic across thresholds
        alert_flags = np.zeros(len(tpc_scores), dtype=bool)
        if self.tpc_threshold is not None:
            alert_flags |= tpc_scores > self.tpc_threshold
        if self.pmt_threshold is not None:
            alert_flags |= pmt_scores > self.pmt_threshold
        if self.window_threshold is not None:
            alert_flags |= window_scores > self.window_threshold
        
        return {
            'tpc_scores': tpc_scores,
            'pmt_scores': pmt_scores,
            'window_scores': window_scores,
            'max_scores': max_scores,
            'alert_flags': alert_flags,
        }

    @classmethod
    def from_checkpoints(
        cls,
        tpc_checkpoint: str | Path,
        pmt_checkpoint: str | Path,
        window_checkpoint: str | Path,
        tpc_config: dict,
        pmt_config: dict,
        window_config: dict,
        tpc_data_config: dict | None = None,
        pmt_data_config: dict | None = None,
        window_data_config: dict | None = None,
        tpc_threshold: float | None = None,
        pmt_threshold: float | None = None,
        window_threshold: float | None = None,
    ) -> MultiBranchScorer:
        """Load three models from checkpoint files.
        
        Args:
            tpc_checkpoint: Path to TPC model checkpoint
            pmt_checkpoint: Path to PMT model checkpoint
            window_checkpoint: Path to Window model checkpoint
            tpc_config: TPC model config dict
            pmt_config: PMT model config dict
            window_config: Window model config dict
            tpc_data_config: TPC data config dict (for normalize flag)
            pmt_data_config: PMT data config dict (for normalize flag)
            window_data_config: Window data config dict (for normalize flag)
            tpc_threshold: Alert threshold for TPC
            pmt_threshold: Alert threshold for PMT
            window_threshold: Alert threshold for Window
            
        Returns:
            MultiBranchScorer instance with all three branches loaded
        """
        from sbn_anomaly.infer.inferrer import AnomalyScorer
        from sbn_anomaly.models.tpc_model import TPCAutoencoder
        from sbn_anomaly.models.pmt_model import PMTAutoencoder
        from sbn_anomaly.models.window_model import WindowAutoencoder
        
        # Default data configs if not provided
        if tpc_data_config is None:
            tpc_data_config = {}
        if pmt_data_config is None:
            pmt_data_config = {}
        if window_data_config is None:
            window_data_config = {}
        
        # Load TPC model
        tpc_model = TPCAutoencoder(
            input_dim=tpc_config.get("input_dim", 256),
            latent_dim=tpc_config.get("latent_dim", 32),
        )
        tpc_scorer = AnomalyScorer.from_checkpoint(
            checkpoint_path=str(tpc_checkpoint),
            model=tpc_model,
            model_type="tpc",
            threshold=tpc_threshold,
            normalize=tpc_data_config.get("normalize", False),
        )
        
        # Load PMT model
        pmt_model = PMTAutoencoder(
            input_dim=pmt_config.get("input_dim", 128),
            latent_dim=pmt_config.get("latent_dim", 16),
        )
        pmt_scorer = AnomalyScorer.from_checkpoint(
            checkpoint_path=str(pmt_checkpoint),
            model=pmt_model,
            model_type="pmt",
            threshold=pmt_threshold,
            normalize=pmt_data_config.get("normalize", False),
        )
        
        # Load Window model
        window_model = WindowAutoencoder(
            window_size=window_config.get("window_size", 256),
            n_channels=window_config.get("n_channels", 1),
            latent_dim=window_config.get("latent_dim", 64),
        )
        window_scorer = AnomalyScorer.from_checkpoint(
            checkpoint_path=str(window_checkpoint),
            model=window_model,
            model_type="window",
            threshold=window_threshold,
            normalize=window_data_config.get("normalize", False),
        )
        
        return cls(
            tpc_scorer=tpc_scorer,
            pmt_scorer=pmt_scorer,
            window_scorer=window_scorer,
            tpc_threshold=tpc_threshold,
            pmt_threshold=pmt_threshold,
            window_threshold=window_threshold,
        )


# Import for convenience in other modules
from sbn_anomaly.infer.inferrer import AnomalyScorer  # noqa: E402
