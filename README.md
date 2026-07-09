# sbnAnomalyDetection — Graph VAE

Real-time data-quality-monitoring (DQM) anomaly detection for liquid-argon TPCs
(SBND / ICARUS, the [SBN](https://sbn.fnal.gov/) program) using a **graph
variational autoencoder** on reconstructed hit features.

The model is trained on **good runs only** and reconstructs one short window of
detector activity at a time. Windows that reconstruct poorly are anomalous, so
problems can be flagged from a small amount of data — no waiting for a whole run.

- Model type: `graph_vae`
- Base config: [`configs/graph_vae.yaml`](configs/graph_vae.yaml)
- Sweep config generator: [`config_maker.py`](config_maker.py)
- Sweep runner / evaluator: [`run_graph_vae_sweep.py`](run_graph_vae_sweep.py)
- Single-model train: `sbn-train --config configs/graph_vae.yaml`
- Single-model score: `sbn-infer --config configs/graph_vae.yaml --input <events.npz> --output scores.npz`
- Single-model evaluate: `python -m sbn_anomaly.infer.window_score ...`

> The main maintained workflow is now based on `config_maker.py` and
> `run_graph_vae_sweep.py`. Older helper scripts such as `npz_npy_reader.py` and
> `train_test_from_npz.py` are not part of the recommended workflow and are not
> documented here.

---

## Recommended workflow

For tuning, the intended path is:

```text
configs/graph_vae.yaml
        │
        ▼
config_maker.py
        │  writes many YAML files
        ▼
tuning_configs/graph_vae_sweep/*.yaml
        │
        ▼
run_graph_vae_sweep.py
        │  train each config
        │  infer on good test events
        │  infer on bad test events
        │  evaluate good-vs-bad separation
        ▼
checkpoints/graph_vae/<run_name>/
graph_vae_sweep.sqlite3
```

### Step 1: Prepare sparse event NPZ files

The GraphVAE workflow expects sparse-events `.npz` files. The default sweep
runner paths assume:

```text
data/good_events_test.npz
data/bad_events_test.npz
```

The training input is set in each generated YAML through the copied base config,
usually from `data.events_path` in `configs/graph_vae.yaml`.

### Step 2: Edit the base GraphVAE YAML

Start from:

```bash
configs/graph_vae.yaml
```

This base file defines the common dataset, graph, feature, model, training, and
inference settings. `config_maker.py` copies this file and overrides only the
sweep parameters, so anything not listed in `config_maker.py` still comes from
`configs/graph_vae.yaml`.

Make sure the base YAML has the correct:

- `data.events_path` for good-run training events.
- `data.channel_map` and graph settings.
- `data.node_features` and `data.log_features`.
- `model.encoder_hidden_dims`, `model.decoder_hidden_dims`, and `model.latent_dim`.
- `training.beta_warmup_epochs`, `training.validation_split`, and other fixed training settings.

### Step 3: Generate sweep YAMLs with `config_maker.py`

`config_maker.py` is the sweep-config generator. Edit the constants near the top
of the file to define the parameter grid:

```python
BASE_YAML_PATH = Path("configs/graph_vae.yaml")
OUTPUT_YAML_DIR = Path("tuning_configs/graph_vae_sweep")
START_INDEX = 0

WINDOW_SIZES = [50, 100, 200]
STRIDES = [50, 100, 200]
ADJACENCY_RADII = [4]
BATCH_SIZES = [64]
LEARNING_RATES = [0.001]
BETAS = [0.8]

DEFAULT_WEIGHT_DECAY = 1.0e-4
DEFAULT_MAX_EPOCHS = 200
CHECKPOINT_BASE_DIR = Path("checkpoints/graph_vae")
```

Then generate the YAMLs:

```bash
python config_maker.py
```

Common options:

```bash
# Force stride = window_size for every window size
python config_maker.py --same-stride

# Start numbering from a chosen index
python config_maker.py --start 12
```

Each generated YAML gets a run name like:

```text
0000_win50_stride50_rad4_bs64_lr0p001_beta0p8.yaml
```

The run name encodes:

```text
index, window_size, stride, adjacency_radius, batch_size, learning rate, beta
```

`config_maker.py` skips invalid combinations such as `stride > window_size`,
negative adjacency radius, invalid batch size, or negative beta.

### Step 4: Run the sweep with `run_graph_vae_sweep.py`

`run_graph_vae_sweep.py` is the main automation script. For every YAML in the
sweep directory, it:

1. Creates a per-run output directory.
2. Writes a patched `config_run.yaml` so runs do not overwrite each other.
3. Trains the GraphVAE.
4. Runs inference separately on good and bad test NPZ files.
5. Runs `window_score` to compare good vs. bad score distributions.
6. Records parameters, commands, statuses, paths, and metrics in SQLite.

Basic sweep:

```bash
python run_graph_vae_sweep.py --batch --export-summary
```

Useful explicit version:

```bash
python run_graph_vae_sweep.py \
  --config-dir tuning_configs/graph_vae_sweep \
  --runs-root checkpoints/graph_vae \
  --db-path graph_vae_sweep.sqlite3 \
  --good-input data/good_events_test.npz \
  --bad-input data/bad_events_test.npz \
  --channel-map configs/SBNDTPCChannelMap_v2_with_positions.csv \
  --eval-aggregator group_max_mean \
  --eval-percentile 90 \
  --batch \
  --export-summary
```

By default, outputs for one config go to:

```text
checkpoints/graph_vae/<run_name>/
├── config_run.yaml
├── graph_vae_final.pt
├── standardization.npz
├── training_history.csv
├── training_curves.png
└── inference_result/
    ├── scores_good.npz
    ├── scores_bad.npz
    ├── goodvsbad.png
    ├── goodvsbad_eval.txt
    └── goodvsbad_eval.json
```

The sweep database is:

```text
graph_vae_sweep.sqlite3
```

When `--export-summary` is set, the script also writes:

```text
graph_vae_sweep_summary.csv
graph_vae_sweep_summary.xlsx
```

### Important sweep-runner modes

Use these when you already have partial results and do not want to rerun every
stage.

```bash
# Train only; skip both good/bad inference jobs
python run_graph_vae_sweep.py --skip-infer

# Use existing checkpoints and rerun good/bad inference, then evaluation
python run_graph_vae_sweep.py --infer-only

# Only rerun inference for runs missing scores_good.npz or scores_bad.npz
python run_graph_vae_sweep.py --missing-infer-only

# Use existing scores_good.npz and scores_bad.npz; rerun only evaluation
python run_graph_vae_sweep.py --evaluate-only

# Only rerun evaluation when goodvsbad.png, goodvsbad_eval.txt, or
# goodvsbad_eval.json is missing
python run_graph_vae_sweep.py --missing-evaluate-only

# Train and infer, but skip good-vs-bad evaluation
python run_graph_vae_sweep.py --skip-eval
```

Notes:

- `--missing-infer-only` implies `--infer-only`.
- `--missing-evaluate-only` implies `--evaluate-only`.
- `--evaluate-only` cannot be combined with `--infer-only` or `--missing-infer-only`.
- `--evaluate-only` / `--missing-evaluate-only` cannot be combined with `--skip-eval`.
- `--eval-threshold` passes an explicit threshold to `window_score` and overrides `--eval-percentile`.
- `--eval-percentile` passes `--percentile` to `window_score`; the current sweep default is `90.0`.
- `--batch` sets batch/log-friendly environment variables so progress bars are suppressed.
- `--monitor-interval 0` disables periodic CPU/RAM usage logging.
- `--force-rewrite` deletes existing database records for matching run names and reruns them.
- `--missing-rewrite` reruns a run only when its output directory is missing or incomplete.

---

## How the network works

### The problem

A window of events is summarized as a **graph**: nodes are TPC channels, node
features are per-channel aggregates of the reconstructed hits in that window, and
edges connect channels that share readout electronics. The autoencoder learns to
reconstruct the nominal good-run feature graph; on anomalous data the
reconstruction error rises. Because it compares each window to the **absolute
learned nominal** rather than to a forecast of recent history, a persistently bad
condition stays flagged and there is no warm-up latency.

### It is a single model, not two stages

The encoder and decoder **are** graph-neural-network layers. Message passing on
the detector/electronics graph happens inside the autoencoder, trained end to end
as a beta-VAE. It is **not** “a GNN followed by an autoencoder.”

```text
one window  ->  per-channel feature graph
                 nodes    = channels
                 features = [sum, min, max, mean, stdev, count, ...] per temporal bin
                 edges    = electronics / wire / sequential graph
        │
        ▼
  graph encoder
        │
        ▼
  per-node variational bottleneck: mu, logvar -> z
        │
        ▼
  decoder -> reconstructed node features
        │
        ▼
  per-channel reconstruction error
        │
        ▼
  per-window anomaly score
        │
        ▼
  good-vs-bad evaluation / streaming flag
```

### Why a variational autoencoder

The bottleneck, KL term, and masking constrain the model to the nominal manifold,
so it reconstructs good windows well but fails on anomalies. A plain high-capacity
autoencoder can learn to reconstruct bad windows too, reducing anomaly
separation. The main regularization knobs are:

- `model.latent_dim`
- `training.beta`
- `training.beta_warmup_epochs`
- `model.mask_ratio`
- `model.dropout`
- encoder/decoder width and depth

### Scoring

1. **Per-channel error**: reconstruction MSE per channel per window, stored as `node_scores`.
2. **Per-window score**: aggregation over channel errors, usually `group_max_mean`.
3. **Evaluation threshold**: usually a high percentile of good-run scores, controlled by `--eval-percentile` in the sweep runner or `--percentile` in `window_score`.
4. **Optional streaming rule**: `window_score --stream --persist-n N --persist-m M` can evaluate persistent M-of-N alarms.

---

## Installation

```bash
pixi install
pixi shell
# or:
pip install -e ".[dev]"
```

Requires Python >= 3.9, PyTorch, PyTorch-Geometric, uproot, awkward, numpy,
pandas, matplotlib, and PyYAML.

---

## Data processing

Training and inference consume a compact sparse-events `.npz`, not a dense array.
Build it once from reconstructed ROOT files; windows are then formed on the fly.

### ROOT file lists

A file list is a text file with one ROOT path per line. Comments with `#` are
allowed.

```text
# data/good_train.txt
/pnfs/icarus/persistent/users/micarrig/DQM/19305/reco/run19305_evt0.root
/pnfs/.../reco/run19305_evt1.root
```

### Materialize sparse events

```bash
python -m sbn_anomaly.data.materialize_windows \
  --config configs/graph_vae.yaml \
  --root-file-list data/good_train.txt \
  --output data/good_events_train.npz
```

Do this separately for good-training, good-test, and bad-test sets:

```bash
python -m sbn_anomaly.data.materialize_windows \
  --config configs/graph_vae.yaml \
  --root-file-list data/good_test.txt \
  --output data/good_events_test.npz

python -m sbn_anomaly.data.materialize_windows \
  --config configs/graph_vae.yaml \
  --root-file-list data/bad_test.txt \
  --output data/bad_events_test.npz
```

`--root-file-list` takes ROOT paths. `sbn-infer --input` takes the resulting
`.npz`. Do not pass a ROOT file list to `sbn-infer --input`.

### Sparse event format

The events NPZ stores hits in CSR-like form, keeping only channels with hits.

| Key | Meaning |
|---|---|
| `channels_flat` | Channel IDs of all hits, concatenated over events. |
| `integrals_flat` | Hit integrals in the same order. |
| `times_flat` | Hit times, used by timing features when available. |
| `offsets` | Event `i` uses hits `channels_flat[offsets[i]:offsets[i+1]]`. |
| `n_channels` | Detector channel count. |
| `evt_run`, `evt_subrun`, `evt_num` | Per-event provenance. |
| `filenames` / file indices | File provenance, when saved. |

Merge sparse event files with the project merge tool rather than plain
`np.concatenate`, because offsets must be shifted correctly:

```bash
python -m sbn_anomaly.data.merge_events \
  --output data/all_good.npz \
  --glob 'data/good_*.npz'
```

---

## Single-model training

The sweep workflow is recommended, but a single YAML can still be trained
directly.

```bash
sbn-train --config configs/graph_vae.yaml
```

or stream from ROOT:

```bash
sbn-train --config configs/graph_vae.yaml --root-file-list data/good_train.txt
```

Typical training outputs under `training.checkpoint_dir`:

| Output | Meaning |
|---|---|
| `graph_vae_final.pt` | Final model weights. |
| `standardization.npz` | Good-run feature mean/std used again at inference. |
| `training_history.csv` | Per-epoch metrics. |
| `training_curves.png` | Loss, reconstruction/KL, score percentiles, throughput. |
| `plots/reconstruction_hist2d*.png` | Original-vs-reconstructed feature plots. |

---

## Single-model inference

```bash
sbn-infer \
  --config configs/graph_vae.yaml \
  --input data/good_events_test.npz \
  --output scores_good.npz

sbn-infer \
  --config configs/graph_vae.yaml \
  --input data/bad_events_test.npz \
  --output scores_bad.npz
```

Typical output arrays:

| Array | Shape | Description |
|---|---:|---|
| `node_scores` | `(W, C)` | Per-window per-channel reconstruction error; inactive channels may be `NaN`. |
| `scores` | `(W,)` | Per-window aggregated score. |
| `scores_max` | `(W,)` | Max-style score when saved by the inferrer. |
| `channel_mean_error` | `(C,)` | Mean per-channel error over windows. |
| `channel_max_error` | `(C,)` | Max per-channel error over windows. |
| `channel_active_frac` | `(C,)` | Fraction of windows where each channel was active. |
| `provenance` | `(W, 3)` | Usually `(run, subrun, event)` for each window. |
| `is_anomaly` | `(W,)` | Present when an inference threshold is set. |

---

## Single-model evaluation

All evaluation runs on saved score NPZ files, so you can change the aggregator or
threshold without rerunning the model.

```bash
python -m sbn_anomaly.infer.window_score \
  --scores scores_good.npz \
  --compare scores_bad.npz \
  --labels good bad \
  --aggregator group_max_mean \
  --channel-map configs/SBNDTPCChannelMap_v2_with_positions.csv \
  --percentile 90 \
  --plot goodvsbad.png
```

This reports score statistics, threshold-free separation AUC, and thresholded
classification metrics such as precision, recall, F1, and false-positive rate.

For streaming/persistence evaluation:

```bash
python -m sbn_anomaly.infer.window_score \
  --scores scores_good.npz \
  --compare scores_bad.npz \
  --labels good bad \
  --aggregator group_max_mean \
  --channel-map configs/SBNDTPCChannelMap_v2_with_positions.csv \
  --percentile 90 \
  --stream --persist-n 3 --persist-m 2
```

Common aggregators:

- `mean`
- `max`
- `topk_mean`
- `group_max_mean`

`group_max_mean` is usually the default for coherent electronics groups because
it pools channel errors by electronics group and takes the worst group.

---

## Node features

Node features are per-channel aggregates computed per temporal bin. If
`n_temporal_bins = B` and `len(node_features) = F`, the node feature dimension is
approximately `B × F`, plus any optional conditioning features used by the model.

| Feature | Meaning |
|---|---|
| `sum` | Sum of hit integrals. |
| `min` | Minimum hit integral. |
| `max` | Maximum hit integral. |
| `mean` | Mean hit integral. |
| `stdev` | Standard deviation of hit integrals. |
| `count` | Number of hits. |
| `occupancy` | Fraction of events in the bin with at least one hit. |
| `hit_rate` | Average hits per event. |
| `time_mean` | Mean hit time, when time data are available. |
| `time_spread` | Spread of hit time, when time data are available. |

### `log_features`

Integral-magnitude features are often heavy-tailed. Put features such as `sum`,
`min`, `max`, and `mean` in `data.log_features` to apply a sign-preserving
`log1p` transform before standardization:

```text
sign(x) * log1p(abs(x))
```

Training and inference must use the same `log_features` list.

---

## Graph construction

`data.edge_mode` controls graph construction from the channel map.

| Mode | Meaning |
|---|---|
| `electronics` | ASIC/FEMB-style electronics connections; recommended for coherent board faults. |
| `wire` | Adjacent wires within plane/TPC/side groups. |
| `both` | Union of electronics and wire edges. |
| `sequential` | Connect nearby offline channel IDs using `adjacency_radius`. |

`config_maker.py` currently sweeps `data.adjacency_radius`, which matters most
for `sequential` graph construction and any graph builder mode that uses radius.

---

## Configuration reference

### `data`

| Key | Meaning |
|---|---|
| `events_path` | Training sparse-events NPZ. |
| `windows_path` | Optional dense-window input; sparse events are preferred. |
| `window_size` | Number of events per window. |
| `stride` | Number of events between consecutive windows. |
| `n_temporal_bins` | Number of time/event bins inside one window. |
| `n_channels` | Detector channel count. |
| `channel_map` | Channel-map CSV. |
| `edge_mode` | Graph construction mode. |
| `adjacency_radius` | Radius for sequential/radius-based graph edges. |
| `node_features` | Per-channel aggregate feature list. |
| `log_features` | Features transformed before standardization. |
| `standardize` | Whether to z-score features using training statistics. |
| `prune_inactive` | Whether inactive channels are removed from per-window graphs. |

### `model`

| Key | Meaning |
|---|---|
| `latent_dim` | Per-node bottleneck dimension. Smaller is tighter. |
| `encoder_hidden_dims` | Encoder graph-layer widths. Length controls encoder depth. |
| `decoder_hidden_dims` | Decoder MLP widths. Length controls decoder depth. |
| `dropout` | Dropout regularization. |
| `mask_ratio` | Denoising mask fraction. |
| `use_channel_idx` | Whether to condition on channel identity. |
| `conv` | Graph convolution type, such as `sage` or `gcn`. |

Example:

```yaml
model:
  latent_dim: 12
  encoder_hidden_dims: [128, 64, 32]
  decoder_hidden_dims: [32, 64]
  dropout: 0.1
  mask_ratio: 0.15
  use_channel_idx: true
  conv: sage
```

Keep the training and inference architecture identical. If the YAML architecture
changes after training, checkpoint loading can fail with size mismatches.

### `training`

| Key | Meaning |
|---|---|
| `lr` | Learning rate. |
| `weight_decay` | Weight decay. |
| `batch_size` | Training batch size. |
| `max_epochs` | Maximum training epochs. |
| `validation_split` | Fraction of training windows used for validation. |
| `beta` | KL weight for beta-VAE training. |
| `beta_warmup_epochs` | Linear warmup period for beta. |
| `checkpoint_dir` | Output directory for weights and training artifacts. |
| `output_path` | Final checkpoint path. |

### `inference`

| Key | Meaning |
|---|---|
| `checkpoint_path` | Model checkpoint to load. |
| `input_path` | Default input NPZ, overridden by `sbn-infer --input`. |
| `output_path` | Default output NPZ, overridden by `sbn-infer --output`. |
| `batch_size` | Inference batch size. |
| `window_aggregator` | Per-window score aggregator. |
| `group_level` | Grouping level for group-based aggregators. |
| `topk` | `topk_mean` parameter. |
| `threshold` | Optional inference-time threshold. |
| `plot` | Whether to write inference plots. |

---

## Tuning notes

- Tune with saved-score evaluation, not just histogram appearance.
- Start by sweeping `window_size`, `stride`, `adjacency_radius`, `batch_size`, `lr`, and `beta` through `config_maker.py`.
- Use `--evaluate-only` to re-threshold or re-aggregate without retraining or rerunning inference.
- Use `--missing-evaluate-only` after interrupted sweeps when only some evaluation products are missing.
- Use `--missing-infer-only` after interrupted sweeps when some models trained but did not produce both good and bad score files.
- Increasing model capacity can improve reconstruction but may reduce anomaly separation if bad windows are reconstructed too well.
- If reconstruction plots become flat bands, check whether heavy-tailed integral features need `log_features`.
- If KL collapses to zero while reconstruction stalls, inspect `beta`, `beta_warmup_epochs`, latent dimension, and decoder capacity.
- For real-time DQM, streaming latency and false-alarm rate are often more meaningful than raw window-level recall.

---

## Tests

```bash
pytest tests/
# or:
.pixi/envs/default/bin/python -m pytest tests/
```

Relevant suites include node-feature tests, window-score/evaluation tests,
channel-graph tests, and GraphVAE data/model/training tests.
