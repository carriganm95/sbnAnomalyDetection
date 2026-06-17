# sbnAnomalyDetection

Streaming anomaly detection pipeline for the [Short-Baseline Neutrino (SBN)](https://sbn.fnal.gov/) experiment at Fermilab.

Two complementary tracks share the same GNN+GRU forecasting core:

- **Hit-level** (reconstructed) — node features are per-channel hit aggregates. See below.
- **Raw-ADC** — a 1-D convolutional **VAE** compresses each channel's raw ADC waveform into a latent that feeds the same forecaster. See **[RAW_ADC_WORKFLOW.md](RAW_ADC_WORKFLOW.md)** (model types `raw_vae` and `raw_gnn`, configs `configs/raw_vae.yaml` and `configs/raw_gnn.yaml`).

## Primary Architecture — GNN Forecaster

The current model is a **graph neural network forecaster** (`GNNForecasterPyG`) that treats TPC channels as nodes in a spatial graph and learns to predict the next time window from a history of past windows. Anomaly scores are the per-channel MSE between the predicted and actual next window.

```
Past windows (history × window_size events)
        │
        ▼
  ┌─────────────┐   for each time step
  │  GCN layers │◄── spatial message passing between neighboring channels
  └──────┬──────┘
         │ node embeddings
         ▼
  ┌─────────────┐
  │     GRU     │   temporal encoding over history steps
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │   Linear    │   predict next-window features per node
  └──────┬──────┘
         │
         ▼
  MSE vs actual next window  →  anomaly score per channel and per window
```

### Key design choices

| Component | Detail |
|-----------|--------|
| Input | Sparse TPC hit integrals from ROOT files |
| Graph nodes | TPC channels (wires) |
| Graph edges | Channels within configurable `adjacency_radius` of each other |
| Node features | Per-channel aggregates per temporal bin: `sum`, `min`, `max`, `mean`, `stdev`, `count` |
| Temporal model | GRU over `history` past frames |
| Anomaly score | Per-channel MSE between predicted and actual next frame |
| Pruning | Only active channels (non-zero across history) are included in each graph |

## Project Structure

```
sbnAnomalyDetection/
├── sbn_anomaly/
│   ├── data/
│   │   ├── sparse_window_dataset.py  # SparseWindowDatasetPyG — main GNN dataset
│   │   ├── graph_window_dataset_pyg.py  # PyG Data/Batch construction, edge index
│   │   ├── streaming.py              # RootStreamer — uproot-based ROOT streaming
│   │   ├── materialize_windows.py    # (legacy) dense window materializer
│   │   ├── stream_dataset.py         # TPCStreamDataset for TPC autoencoder
│   │   └── dataset.py                # Map-style datasets (TPC, PMT, Fusion, Window)
│   ├── models/
│   │   ├── gnn_forecaster_pyg.py     # GNNForecasterPyG — primary model
│   │   ├── tpc_model.py              # TPCAutoencoder
│   │   ├── pmt_model.py              # PMTAutoencoder
│   │   ├── fusion_model.py           # FusionAutoencoder
│   │   └── window_model.py           # WindowAutoencoder
│   ├── train/
│   │   ├── cli.py                    # sbn-train entry point
│   │   ├── gnn_trainer.py            # GNNTrainerPyG
│   │   └── trainer.py                # BaseTrainer (shared training loop)
│   ├── infer/
│   │   ├── cli.py                    # sbn-infer entry point
│   │   └── inferrer.py               # GNNScorer, AnomalyScorer
│   └── utils/
├── configs/
│   ├── gnn.yaml                      # GNN configuration (primary)
│   ├── tpc.yaml
│   ├── pmt.yaml
│   └── ...
├── tests/
└── pyproject.toml
```

## Installation

```bash
pixi install
pixi shell
```

Or with pip:

```bash
pip install -e ".[dev]"
```

## GNN Training

### From ROOT files (recommended)

Stream directly from ROOT files. Events are loaded into a `SparseWindowDatasetPyG` and optionally cached as a compact `.npz` for subsequent runs.

```bash
sbn-train --config configs/gnn.yaml \
  --root-files /data/run1.root /data/run2.root

# Or from a file list manifest (one path per line, # comments allowed)
sbn-train --config configs/gnn.yaml \
  --root-file-list data/train_files.txt
```

To cache the loaded events for faster future runs, set `training.save_events_path` in `configs/gnn.yaml`:

```yaml
training:
  save_events_path: data/events_cache.npz
```

### From a cached events file

Once an events NPZ has been saved (either from a prior training run or explicitly), point `data.events_path` at it:

```yaml
data:
  events_path: data/events_cache.npz
```

Then run without `--root-files`:

```bash
sbn-train --config configs/gnn.yaml
```

### Key config parameters

```yaml
model_type: gnn

data:
  events_path: data/events_cache.npz   # sparse events NPZ (or use --root-files)
  tree_name: caloskim/TrackCaloSkim    # ROOT TTree path
  hit_branches:                         # which hit branches to read
    - hits0.h.integral
    - hits0.h.channel
    - hits2.h.integral
    - hits2.h.channel
  window_size: 20        # events per frame
  n_temporal_bins: 4     # bins within each frame
  stride: 5              # step between consecutive windows
  adjacency_radius: 4    # channels within this radius are connected
  node_features:         # per-channel aggregates per bin
    - sum
    - min
    - max
    - mean
    - stdev
    - count

model:
  history: 4             # number of past frames used to predict the next frame
  gnn_hidden: 64
  gnn_layers: 2
  gru_hidden: 128

training:
  batch_size: 32
  num_workers: 4
  lr: 0.001
  max_epochs: 10
  validation_split: 0.1
  checkpoint_dir: checkpoints/gnn/
  output_path: checkpoints/gnn/gnn_final.pt
  save_events_path: data/events_cache.npz  # optional: cache events after loading
```

### Training output

After training, the checkpoint directory contains:
- `gnn_final.pt` — trained model weights
- `training_history.csv` — per-epoch loss and validation loss
- `training_curves.png` — loss curves
- `score_distribution.png` — histogram of window anomaly scores on the training set
- `score_over_time.png` — anomaly score vs window index (temporal trend)
- `node_mse.png` — per-channel average prediction MSE (shows which wires are hardest to predict)

## GNN Inference

```bash
sbn-infer --config configs/gnn.yaml --output scores.npz
```

The `inference.input_path` in `configs/gnn.yaml` points at the windows/events file to score. The output is a compressed `.npz` containing:

| Array | Shape | Description |
|-------|-------|-------------|
| `scores` | `(N_windows,)` | Mean active-channel MSE per window |
| `scores_max` | `(N_windows,)` | Max active-channel MSE per window |
| `node_scores` | `(N_windows, N_channels)` | Per-channel MSE, NaN for inactive channels |
| `event_index` | `(N_windows,)` | Window index |
| `is_anomaly` | `(N_windows,)` | Boolean flag (only when `inference.threshold` is set) |

## Raw-ADC VAE Pipeline

The raw-ADC track compresses each channel's raw waveform with a 1-D convolutional
variational autoencoder (`model_type: raw_vae`); the latents then feed the same
GNN forecaster as `raw_gnn`. Full design notes are in
[RAW_ADC_WORKFLOW.md](RAW_ADC_WORKFLOW.md); this section covers building the
training `.npz` files and training the VAE on them.

### Step 1 — Create the waveform `.npz` (one step, on a LArSoft node)

Raw `raw::RawDigit` ADCs live in art files and can only be decoded by
LArSoft/gallery (uproot cannot read the memberwise-serialized vector). So the
dump runs on a gpvm or in an SL7/LArSoft container where the experiment software
is set up. `scripts/rawdigits_to_npz_gallery.py` reads the digits via gallery,
applies preprocessing, and writes the `.npz` in **one step** — no intermediate
flat ROOT file:

```bash
# on a gpvm / SL7 container, AFTER: source .../setup_icarus.sh ; setup icaruscode ...
python scripts/rawdigits_to_npz_gallery.py \
    --config configs/raw_vae.yaml \
    --root-files /pnfs/.../raw_decoded_reco_056.root \
    --output data/raw_waveforms_train.npz \
    --tag daq
```

Minimal dependencies for this step are just **numpy + PyYAML** (ROOT/PyROOT comes
from the LArSoft setup, not pip). If `import sbn_anomaly` fails, run with
`PYTHONPATH=$PWD` or `pip install -e . --no-deps`.

#### Input file lists (same as the GNN)

You can pass individual files, glob patterns, or a manifest — identical to the
GNN's `--root-file-list`:

```bash
# explicit files or a glob
python scripts/rawdigits_to_npz_gallery.py --config configs/raw_vae.yaml \
    --root-files '/pnfs/.../raw_decoded_reco_*.root' --output data/raw_waveforms_train.npz

# a manifest: one ROOT path per line, blank lines and #-comment lines ignored
python scripts/rawdigits_to_npz_gallery.py --config configs/raw_vae.yaml \
    --root-file-list data/filelist_train.txt --output data/raw_waveforms_train.npz
```

Gallery chains the whole list in one pass; `(run, subrun, event)` provenance is
kept per waveform so events stay distinguishable. After **each input file** the
run logs how many waveforms it contributed:

```
File /pnfs/.../raw_decoded_reco_056.root: added 360448 waveforms (running total 360448)
File /pnfs/.../raw_decoded_reco_057.root: added 360448 waveforms (running total 720896)
```

#### Chunking large file lists into shards

For long file lists the waveforms won't fit in memory, so use `--shard-size` to
flush fixed-size shards and keep processing. Output `data/raw_waveforms_train.npz`
becomes `data/raw_waveforms_train_000.npz`, `_001.npz`, …:

```bash
python scripts/rawdigits_to_npz_gallery.py --config configs/raw_vae.yaml \
    --root-file-list data/filelist_train.txt \
    --output data/raw_waveforms_train.npz \
    --shard-size 2000000 --tag daq
```

Behavior:

- A new shard is written every `--shard-size` **waveforms** (not events). Shards
  span file boundaries, and a single event larger than a shard is split across
  shards. The final shard holds the remainder.
- Omit `--shard-size` to write a single `.npz`.
- `--max-waveforms N` caps the **global** total across all shards (useful to stop
  early); `--max-events N` caps total events.
- `--compress` writes compressed npz (smaller, slower to load).

**`--max-waveforms` counts per-channel waveforms, not events.** Each row of the
output is one channel from one event, so with ~11k channels per event the count
climbs ~11k per event. The `data.channels_per_event` config key controls how many
channels each event contributes (e.g. `1024` to subsample and spread a fixed
budget across more events).

Each shard is self-contained: `waveforms` `(N, input_length)` float32, `channel`
`(N,)`, and `prov` `(N, 3)` = `(run, subrun, event)` per row. The waveforms are
already preprocessed per the `data.preprocess` block (pedestal subtraction,
coherent-noise removal, scaling) — see the comments in `configs/raw_vae.yaml` to
tune each step.

> Already have flat `rawdigits` ntuples (e.g. from `scripts/dump_rawdigits.C`)?
> The same flags work via `python -m sbn_anomaly.data.build_raw_waveforms`
> (numpy + uproot, no LArSoft needed).

### Step 2 — Train the VAE on the `.npz`

Point `data.waveforms_path` in `configs/raw_vae.yaml` at the file — a single
`.npz`, or a **glob across shards** — and run training:

```yaml
# configs/raw_vae.yaml
data:
  waveforms_path: data/raw_waveforms_train_*.npz   # glob picks up all shards
```

```bash
sbn-train --config configs/raw_vae.yaml
```

The dataset loads every matching shard and concatenates them, so sharded output
trains transparently. (This holds the full set in RAM at start; if your shards
exceed memory, train on a subset glob or open an issue for a lazy multi-shard
loader.) Training writes a checkpoint to `training.output_path`
(default `checkpoints/raw_vae/v1/vae_final.pt`) plus loss/score plots.

You can also skip the cached `.npz` and stream straight from flat ntuples with
`sbn-train --config configs/raw_vae.yaml --root-files ...`, but for repeated
training the cached `.npz` is faster.

### Step 3 — Plot / sanity-check waveforms

```bash
python scripts/plot_raw_waveforms.py --input data/raw_waveforms_train_000.npz \
    --random 9 --output sample.png
```

The y-axis is the preprocessed (pedestal-subtracted, coherent-noise-removed,
scaled) waveform; set `data.preprocess.scale: 1.0` and `remove_coherent: false`
in the config if you want raw ADC counts instead.

## Data Pipeline

### Sparse event representation

Raw TPC hits are stored as a CSR-style flat array in the events NPZ:

| Key | Dtype | Description |
|-----|-------|-------------|
| `channels_flat` | int64 | Concatenated channel indices across all events |
| `integrals_flat` | float32 | Corresponding hit integrals |
| `offsets` | int64 | CSR row pointers — event `i` spans `[offsets[i], offsets[i+1])` |
| `n_channels` | int64 | Total channel count |

This format is ~1000× more compact than a dense window array because most channels are inactive in any given event.

### Window construction

`SparseWindowDatasetPyG` assembles windows lazily in `__getitem__`:

```
Window i covers events[start : start + (history+1)*window_size]
  └─ split into (history+1) frames of window_size events each
       └─ each frame split into n_bins temporal bins
            └─ hits per bin aggregated per channel → node features
```

Per-event features are pre-aggregated once at construction time, so `__getitem__` only needs to combine `~events_per_bin` already-deduplicated arrays per bin (no per-sample sorting).

### Streaming from ROOT

```python
from sbn_anomaly.data.sparse_window_dataset import SparseWindowDatasetPyG

dataset = SparseWindowDatasetPyG.from_root(
    root_files=["run1.root", "run2.root"],
    tree_name="caloskim/TrackCaloSkim",
    hit_branches=["hits0.h.integral", "hits0.h.channel"],
    history=4,
    window_size=20,
    n_bins=4,
    stride=5,
    radius=4,
)
dataset.save_events("events_cache.npz")  # cache for future runs

### Materialize windows / events (.npz)

Use the materializer to produce a compact events `.npz` by default. This is
the preferred format for GNN training because windows are built lazily by
`SparseWindowDatasetPyG`. Pass `--windows` only when you explicitly want the
legacy dense `.npy` window array plus a companion `_meta.npz` file.

Sparse events NPZ (default, recommended for GNN training; saved as `data/events_cache.npz`):

```bash
python -m sbn_anomaly.data.materialize_windows \
  --root-files /data/run1.root /data/run2.root \
  --output data/events_cache.npz \
  --window-size 20 \
  --n-bins 4 \
  --stride 5
```

Dense windows (legacy opt-in, saved as `data/windows.npy` + `data/windows_meta.npz`):

```bash
python -m sbn_anomaly.data.materialize_windows \
  --root-files /data/run1.root /data/run2.root \
  --output data/windows \
  --window-size 20 \
  --n-bins 4 \
  --stride 5 \
  --windows
```

Alternatively, build and save the sparse events programmatically (same result):

```bash
python - <<'PY'
from sbn_anomaly.data.sparse_window_dataset import SparseWindowDatasetPyG

ds = SparseWindowDatasetPyG.from_root(
    root_files=["/data/run1.root", "/data/run2.root"],
    tree_name="caloskim/TrackCaloSkim",
    hit_branches=["hits0.h.integral", "hits0.h.channel"],
    history=4,
    window_size=20,
    n_bins=4,
    stride=5,
    radius=4,
)
ds.save_events("data/events_cache.npz")
PY
```

Note on defaults and precedence
--------------------------------

If you omit `--window-size`, `--n-bins` (temporal bins), or `--stride` on the
command line, the materializer and the train/infer CLIs will read those values
from the provided YAML config under the `data` section (`data.window_size`,
`data.n_temporal_bins`, `data.stride`). Command-line flags override values in
the YAML. If neither CLI flags nor the config supply a value, the materializer
falls back to sensible built-in defaults (e.g. `window_size=20`,
`n_temporal_bins=4`, `stride=1`).

This precedence also applies when training or running inference: the CLI will
use values from the config unless you explicitly pass overriding flags.


### Using a ROOT file list (manifest)

You can pass a file containing ROOT paths (one per line, `#` allowed for comments)
instead of listing files on the command line. This is convenient for long runs
or reproducible manifests.

Materialize sparse events from a manifest file:

```bash
python -m sbn_anomaly.data.materialize_windows \
  --root-file-list data/train_files.txt \
  --output data/events_cache.npz \
  --window-size 20 \
  --n-bins 4 \
  --stride 5
```

Materialize dense windows from a manifest file:

```bash
python -m sbn_anomaly.data.materialize_windows \
  --root-file-list data/train_files.txt \
  --output data/windows \
  --window-size 20 \
  --n-bins 4 \
  --stride 5 \
  --windows
```

Train using a manifest (streaming path):

```bash
sbn-train --config configs/gnn.yaml \
  --root-file-list data/train_files.txt
```

Infer/score using a manifest (streaming TPC path):

```bash
sbn-infer --config configs/gnn.yaml \
  --root-file-list data/test_files.txt \
  --output scores_from_manifest.npz
```

```

## Legacy Autoencoder Models

The original per-subsystem autoencoders are still available for comparison or independent deployment:

| Model | `model_type` | Input |
|-------|-------------|-------|
| `TPCAutoencoder` | `tpc` | TPC waveform/hit features |
| `PMTAutoencoder` | `pmt` | PMT waveform features |
| `FusionAutoencoder` | `fusion` | TPC + PMT (joint or late fusion) |
| `WindowAutoencoder` | `window` | Raw waveform windows (1-D conv) |

```bash
sbn-train --config configs/tpc.yaml
sbn-train --config configs/tpc.yaml --root-files /data/run1.root  # stream from ROOT
sbn-infer --config configs/tpc.yaml --root-file-list data/test_files.txt
```

## Tests

```bash
# With pixi environment active
python -m pytest tests/

# Or point directly at the pixi Python
.pixi/envs/default/bin/python -m pytest tests/
```
