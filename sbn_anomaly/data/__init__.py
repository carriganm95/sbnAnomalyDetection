"""Data ingestion and streaming utilities for SBN ROOT files.

Lightweight, dependency-minimal pieces (the raw-ADC readers and preprocessing,
which need only numpy -- plus ROOT/uproot lazily) are imported unconditionally
so they work in minimal environments such as an SL7/LArSoft container that only
runs the gallery reader. The heavier components (torch / torch-geometric /
uproot / awkward) are imported best-effort so their absence doesn't break those
lightweight uses.
"""

# ---- Lightweight (numpy only; ROOT/uproot imported lazily inside methods) ----
from sbn_anomaly.data.raw_digit_reader import RawDigitReader, RawEvent

__all__ = ["RawDigitReader", "RawEvent"]

# ---- Heavier optional components (torch / uproot / awkward) ----
try:
    from sbn_anomaly.data.streaming import RootStreamer
    from sbn_anomaly.data.event_joiner import EventJoiner
    from sbn_anomaly.data.dataset import TPCDataset, PMTDataset, FusionDataset
    from sbn_anomaly.data.stream_dataset import TPCStreamDataset
    from sbn_anomaly.data.graph_window_dataset import (
        GraphWindowDataset,
        build_channel_adjacency,
    )
    from sbn_anomaly.data.raw_waveform_dataset import (
        RawWaveformStreamDataset,
        RawWaveformArrayDataset,
    )

    __all__ += [
        "RootStreamer",
        "EventJoiner",
        "TPCDataset",
        "PMTDataset",
        "FusionDataset",
        "TPCStreamDataset",
        "GraphWindowDataset",
        "build_channel_adjacency",
        "RawWaveformStreamDataset",
        "RawWaveformArrayDataset",
    ]
except ImportError as _exc:  # optional deps (torch/uproot/awkward) not installed
    import logging as _logging

    _logging.getLogger(__name__).debug(
        "Optional data components unavailable (%s); lightweight readers only.", _exc
    )
