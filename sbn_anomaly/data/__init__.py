"""Data ingestion and streaming utilities for SBN ROOT files."""

from sbn_anomaly.data.streaming import RootStreamer
from sbn_anomaly.data.event_joiner import EventJoiner
from sbn_anomaly.data.dataset import TPCDataset, PMTDataset, FusionDataset
from sbn_anomaly.data.stream_dataset import TPCStreamDataset
from sbn_anomaly.data.graph_window_dataset import GraphWindowDataset, build_channel_adjacency
from sbn_anomaly.data.materialize_windows import materialize_windows_from_root

__all__ = [
    "RootStreamer",
    "EventJoiner",
    "TPCDataset",
    "PMTDataset",
    "FusionDataset",
    "TPCStreamDataset",
    "GraphWindowDataset",
    "build_channel_adjacency",
    "materialize_windows_from_root",
]
