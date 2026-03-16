# sbnAnomalyDetection

Anomaly detection pipeline for the Short-Baseline Neutrino (SBN) programme at Fermilab.
The pipeline ingests raw ROOT files from ICARUS / SBND detectors and trains a family of
autoencoders — one per sub-detector modality — to flag unusual neutrino interaction events
and time-windows.

---

## Architecture overview

```
ROOT files (uproot streaming)
        │
        ▼
 data/root_stream.py       ← lazy uproot.iterate reader, yields batches of arrays
        │
        ▼
 data/event_join.py        ← joins TPC + PMT branches by (run, event) key
        │
        ▼
 data/dataset.py           ← torch Dataset wrapping the joined iterator
        │
     ┌──┴──────────────────────────────┐
     │                                 │
     ▼                                 ▼
models/model.py                 models/model.py
  TPCAutoencoder               PMTAutoencoder
     │                                 │
     └──────────┐      ┌───────────────┘
                ▼      ▼
          FusionAutoencoder   (latent concatenation)
                │
                ▼
          WindowAutoencoder   (sliding-window anomaly score)
                │
         ┌──────┴──────┐
         ▼             ▼
infer/score_events.py  infer/score_windows.py
```

### Option A – independent training jobs (recommended)

Each phase is a separate job and can run on its own GPU / time-slot:

| Phase  | Entry-point              | Config                       |
|--------|--------------------------|------------------------------|
| TPC    | `sbn-train-tpc`          | `configs/tpc_train.yaml`     |
| PMT    | `sbn-train-pmt`          | `configs/pmt_train.yaml`     |
| Fusion | `sbn-train-fusion`       | `configs/fusion_train.yaml`  |
| Window | `sbn-train-window`       | `configs/window_train.yaml`  |

---

## Installation

```bash
# Create a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate

# Install the package in editable mode
pip install -e ".[dev]"

# Or just install runtime dependencies
pip install -r requirements.txt
```

---

## Configuration

All hyper-parameters live in the `configs/` directory as YAML files.
Copy the file you want to tune and point the training script at it via `--config`.

Key fields common to every config:

```yaml
data:
  root_files: ["/path/to/files/*.root"]  # glob or list of paths
  tpc_branches: [...]                     # list of TPC branch names to read
  pmt_branches: [...]                     # list of PMT branch names to read
  batch_size: 512
  max_events: null                        # null = read everything

model:
  latent_dim: 32
  hidden_dims: [256, 128, 64]

training:
  epochs: 50
  lr: 1.0e-3
  weight_decay: 1.0e-5
  checkpoint_dir: checkpoints/
  log_dir: runs/

device: cuda  # or cpu
```

---

## Running the training phases

### Shell scripts (recommended for cluster / slurm)

```bash
bash scripts/run_tpc.sh     # trains the TPC autoencoder
bash scripts/run_pmt.sh     # trains the PMT autoencoder
bash scripts/run_fusion.sh  # trains the fusion autoencoder (requires TPC + PMT ckpts)
bash scripts/run_window.sh  # trains the window autoencoder (requires fusion ckpt)
```

### Python entry-points (after `pip install -e .`)

```bash
sbn-train-tpc    --config configs/tpc_train.yaml
sbn-train-pmt    --config configs/pmt_train.yaml
sbn-train-fusion --config configs/fusion_train.yaml \
                 --tpc-checkpoint checkpoints/tpc_best.pt \
                 --pmt-checkpoint checkpoints/pmt_best.pt
sbn-train-window --config configs/window_train.yaml \
                 --fusion-checkpoint checkpoints/fusion_best.pt
```

---

## Inference / scoring

```bash
# Score individual events (returns per-event reconstruction error)
sbn-score-events  --config configs/fusion_train.yaml \
                  --checkpoint checkpoints/fusion_best.pt \
                  --output scores_events.csv

# Score sliding windows (returns per-window anomaly score)
sbn-score-windows --config configs/window_train.yaml \
                  --checkpoint checkpoints/window_best.pt \
                  --output scores_windows.csv
```

---

## Project layout

```
sbnAnomalyDetection/
├── configs/
│   ├── tpc_train.yaml
│   ├── pmt_train.yaml
│   ├── fusion_train.yaml
│   └── window_train.yaml
├── scripts/
│   ├── run_tpc.sh
│   ├── run_pmt.sh
│   ├── run_fusion.sh
│   └── run_window.sh
├── src/
│   └── sbn_anomaly_detection/
│       ├── data/
│       │   ├── root_stream.py   # uproot.iterate streaming reader
│       │   ├── event_join.py    # run/event key join of TPC + PMT arrays
│       │   └── dataset.py       # torch Dataset
│       ├── models/
│       │   └── model.py         # all autoencoder architectures
│       ├── train/
│       │   ├── train_tpc.py
│       │   ├── train_pmt.py
│       │   ├── train_fusion.py
│       │   └── train_window.py
│       ├── infer/
│       │   ├── score_events.py
│       │   └── score_windows.py
│       └── utils/
│           ├── checkpointing.py
│           ├── logging.py
│           └── normalization.py
├── pyproject.toml
└── requirements.txt
```

---

## Contributing

1. Fork the repository and create a feature branch.
2. Run `ruff check src/` and `mypy src/` before opening a PR.
3. Add or update tests under `tests/` for any new functionality.

---

## License

MIT – see `LICENSE` for details.
