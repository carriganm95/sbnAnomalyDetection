# sbnAnomalyDetection

Streaming anomaly detection pipeline for the [Short-Baseline Neutrino (SBN)](https://sbn.fnal.gov/) experiment at Fermilab.

## Architecture

Option A — **modular independent autoencoders**:

| Module | Input | Description |
|--------|-------|-------------|
| `TPCAutoencoder` | TPC waveform features | Detects wire-level anomalies |
| `PMTAutoencoder` | PMT waveform features | Detects scintillation-light anomalies |
| `FusionAutoencoder` | TPC + PMT (joint or late) | Multimodal joint anomaly score |
| `WindowAutoencoder` | Raw waveform windows | 1-D conv time-series model |

Each model can be trained **independently** and deployed individually or combined.

## Project Structure

```
sbnAnomalyDetection/
├── sbn_anomaly/
│   ├── data/          # ROOT streaming, event joining, dataset classes
│   ├── models/        # TPC, PMT, Fusion, Window autoencoders
│   ├── train/         # Independent trainers + CLI
│   ├── infer/         # Inference scorer + CLI
│   └── utils/         # Metrics, logging helpers
├── configs/           # YAML configuration files per model type
├── scripts/           # Shell convenience wrappers
├── tests/             # pytest unit tests
├── pyproject.toml
└── requirements.txt
```

## Installation

```bash
pip install -e ".[dev]"
```

## Training

```bash
# Train each model independently (from pre-computed .npy features)
sbn-train --config configs/tpc.yaml
sbn-train --config configs/pmt.yaml
sbn-train --config configs/fusion.yaml
sbn-train --config configs/window.yaml

# Train TPC directly from ROOT files (streaming — no .npy files needed)
sbn-train --config configs/tpc.yaml --root-files /data/run1.root /data/run2.root
# Glob patterns work too (quote to prevent premature shell expansion):
sbn-train --config configs/tpc.yaml --root-files "/data/sbn/*.root"

# Or via shell scripts
bash scripts/train_tpc.sh
bash scripts/train_pmt.sh
bash scripts/train_fusion.sh
bash scripts/train_window.sh
```

## Inference

```bash
# Score TPC events
sbn-infer --config configs/tpc.yaml --input data/processed/tpc_features_test.npy

# Score with fusion model
sbn-infer --config configs/fusion.yaml \
  --tpc-input data/processed/tpc_features_test.npy \
  --pmt-input data/processed/pmt_features_test.npy
```

## Data Streaming

ROOT files are read in batches via [uproot](https://uproot.readthedocs.io):

```python
from sbn_anomaly.data import RootStreamer, EventJoiner

tpc_stream = RootStreamer("run.root", tree_name="sbn_tree",
                          branches=["run", "subrun", "event", "tpc_waveform"])
pmt_stream = RootStreamer("run.root", tree_name="sbn_tree",
                          branches=["run", "subrun", "event", "pmt_waveform"])

joiner = EventJoiner()
for tpc_batch, pmt_batch in zip(tpc_stream.stream(), pmt_stream.stream()):
    tpc_matched, pmt_matched = joiner.join(tpc_batch, pmt_batch)
    # ... score with FusionAutoencoder
```

## Tests

```bash
pytest
```
