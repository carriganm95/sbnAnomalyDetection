"""Autoencoder models for SBN anomaly detection.

Architecture (Option A – modular):
  * ``TPCAutoencoder``   – detects wire-level waveform anomalies
  * ``PMTAutoencoder``   – detects scintillation-light anomalies
  * ``FusionAutoencoder``– joint latent space from TPC + PMT
  * ``WindowAutoencoder``– 1-D convolutional model for time-series windows

All models return both the reconstruction and the latent vector so that
downstream code can use either reconstruction error or latent-space distances
as anomaly scores.
"""

from sbn_anomaly.models.tpc_model import TPCAutoencoder
from sbn_anomaly.models.pmt_model import PMTAutoencoder
from sbn_anomaly.models.fusion_model import FusionAutoencoder
from sbn_anomaly.models.window_model import WindowAutoencoder
from sbn_anomaly.models.gnn_forecaster import GNNForecaster
from sbn_anomaly.models.gnn_forecaster_pyg import GNNForecasterPyG

__all__ = [
    "TPCAutoencoder",
    "PMTAutoencoder",
    "FusionAutoencoder",
    "WindowAutoencoder",
    "GNNForecaster",
    "GNNForecasterPyG",
]
