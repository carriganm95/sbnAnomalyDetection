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
- Plotting and score diagnostics: [`graphing/README.md`](graphing/README.md)
- NPZ splitting, filtering, and validation: [`dataset_preparation/README.md`](dataset_preparation/README.md)
- Single-model train: `sbn-train --config configs/graph_vae.yaml`
- Single-model score: `sbn-infer --config configs/graph_vae.yaml --input <events.npz> --output scores.npz`
- Single-model evaluate: `python -m sbn_anomaly.infer.window_score ...`

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

### Prepare existing NPZ datasets

The scripts in [`dataset_preparation/`](dataset_preparation/README.md) operate on
already-materialized NPZ files and are separate from the ROOT materialization
commands above:

| Script | Purpose |
|---|---|
| [`train_test_from_npz.py`](dataset_preparation/train_test_from_npz.py) | Combine `tpc_data_v3_*.npz` sources and create a run-preserving `good_runs.npz` / `windows_train.npz` split; `--with-bad` also creates `bad_runs.npz`. |
| [`filter.py`](dataset_preparation/filter.py) | Apply the same channel slice to training, good-test, and bad-test NPZ files, reindex channels, and optionally filter runs or retain empty events. |
| [`inspect_run.py`](dataset_preparation/inspect_run.py) | Inspect timestamp ordering, run segments, event counts in timed windows, padding, and fixed-event-window durations. |
| [`check_time_sequence.py`](dataset_preparation/check_time_sequence.py) | Report locations where `evt_time` moves backward. |

See the subdirectory README for every flag, editable constant, input constraint,
and output key. These scripts replace the older ad-hoc repair workflow; there is
no `repair.py` in the current directory.

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
| `scores` | `(W,)` | Per-window score produced with `inference.window_aggregator`. |
| `window_index` | `(W,)` | Sequential zero-based window index. |
| `channel_mean_error` | `(C,)` | Mean per-channel error over windows. |
| `channel_max_error` | `(C,)` | Max per-channel error over windows. |
| `channel_active_frac` | `(C,)` | Fraction of windows where each channel was active. |
| `event_count` | `(W,)` | Events contributing to each sparse-data window, when available. |
| `provenance` | `(W, 3)` | First `(run, subrun, event)` for each GraphVAE window, when source provenance is available. `window_score` also accepts score files with separate `first_run`, `first_subrun`, and `first_event_num` arrays. |
| `is_anomaly` | `(W,)` | Present when an inference threshold is set. |
| `threshold` | scalar | The inference threshold used to create `is_anomaly`, when configured. |

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
  --percentile 95 \
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
  --percentile 95 \
  --stream --persist-n 3 --persist-m 2
```

### Window-score aggregators

| Aggregator | What it computes | Additional input |
|---|---|---|
| `mean` | NaN-aware mean over active channels. | None. |
| `max` | NaN-aware maximum channel error. | None. |
| `topk_mean` | Mean of the largest `k` finite channel errors; useful for localized faults without relying on one noisy channel. | `--k` (default `64`). |
| `group_max_mean` | Mean within each FEMB or ASIC group, followed by the maximum group mean; the recommended default for coherent electronics faults. | `--channel-map` and optionally `--group-level`. |
| `zscore_mean` | Per-channel `(score - mu) / sigma`, followed by a NaN-aware channel mean. This targets small, detector-wide shifts that can be hidden by naturally noisy channels. | `--baseline` made from training-only good scores. |

### Training-baseline inference and `zscore_mean`

Fit the channel baseline only on inference scores from the **training split of
good runs**. Do not fit it on the held-out good file used for evaluation, or the
reported separation will be optimistic.

```bash
# Score the same good-only NPZ used to train the model.
sbn-infer \
  --config configs/graph_vae.yaml \
  --input data/events_train.npz \
  --output scores_train.npz

# Pool one or more training-score files into per-channel mu, sigma, and n.
python -m sbn_anomaly.infer.channel_baseline \
  --scores scores_train.npz \
  --min-count 20 \
  --output channel_baseline.npz

# Apply that fixed baseline to disjoint held-out good and bad scores.
python -m sbn_anomaly.infer.window_score \
  --scores scores_good.npz \
  --compare scores_bad.npz \
  --labels good bad \
  --aggregator zscore_mean \
  --baseline channel_baseline.npz \
  --percentile 95 \
  --plot goodvsbad_zscore.png
```

`channel_baseline` pools all files supplied to `--scores`, saves `mu`, `sigma`,
and finite sample count `n` per channel, and excludes channels with fewer than
`--min-count` usable windows or a non-positive/non-finite standard deviation.

| `channel_baseline` flag | Default | Meaning |
|---|---:|---|
| `--scores PATH [PATH ...]` | required | One or more training-only good score NPZ files; all windows are pooled and channel counts must match. |
| `--min-count N` | `20` | Minimum finite windows needed for a valid channel baseline. |
| `--output PATH` | required | Output NPZ; a `.npz` suffix is added when omitted. |
| `-h`, `--help` | — | Show command help. |

### `window_score` flag reference

| Flag | Default | Meaning |
|---|---:|---|
| `--scores PATH` | required | Primary score NPZ; its `node_scores` (or legacy `scores`) is treated as the good/reference sample. |
| `--compare PATH` | none | Second score NPZ to overlay and treat as the positive/bad sample for AUC and classification metrics. |
| `--labels A B` | `good bad` | Labels for the primary and comparison samples. |
| `--aggregator NAME` | `group_max_mean` | One of `mean`, `max`, `topk_mean`, `group_max_mean`, or `zscore_mean`. |
| `--channel-map PATH` | none | Channel-map CSV required by `group_max_mean`. |
| `--group-level {femb,asic}` | `femb` | Electronics grouping used by `group_max_mean`. |
| `--k N` | `64` | Number of largest channel scores averaged by `topk_mean`. |
| `--baseline PATH` | none | `channel_baseline.npz` required by `zscore_mean`. |
| `--output PATH` | none | Save the primary sample's aggregated per-window scores as `.npy`. |
| `--plot PATH` | none | Save a distribution overlay. The operating threshold is drawn and labeled; with good and bad inputs the plot also displays precision and recall at that threshold. |
| `--threshold VALUE` | none | Explicit operating threshold; overrides `--percentile`. |
| `--percentile P` | `99` | Primary/good-score percentile used when `--threshold` is absent. The sweep explicitly passes `95` in its normal evaluation path. |
| `--per-run` | off | Aggregate windows into one score per run and report run-level AUC/confusion metrics. Requires provenance. |
| `--run-stat {frac_above,max,p99,mean}` | `frac_above` | Run rollup statistic. `frac_above` is the fraction of that run's windows above the window threshold. |
| `--run-key {run,runsubrun}` | `run` | Group windows by run or by run/subrun. |
| `--run-threshold VALUE` | good-run p90 | Explicit run-level decision threshold. This is distinct from the window percentile. |
| `--stream` | off | Evaluate a real-time persistence detector per run, reporting good-run false alarms and bad-run detection rate/latency. Requires provenance. |
| `--persist-n N` | `3` | Number of recent windows in the streaming detector. |
| `--persist-m M` | `2` | Trigger when at least `M` of the last `N` windows exceed the window threshold. |
| `--instant-threshold VALUE` | none | Also trigger immediately when one window exceeds this value. |
| `-h`, `--help` | — | Show command help. |

The current provenance loader accepts both the older combined `provenance`
array and the newer separate `first_run` / `first_subrun` /
`first_event_num` arrays, so `--per-run` and `--stream` work with either score
layout. Without provenance, those analyses are skipped while window-level
evaluation still runs.

---

## Graphing and diagnostics

[`graphing/`](graphing/README.md) contains seven plotting and diagnostic scripts.
The subdirectory README documents every callable function, command-line flag,
editable constant, required array, and output. At the root-workflow level:

| Script | Result | Sweep integration |
|---|---|---|
| [`plot_per_channel_scores.py`](graphing/plot_per_channel_scores.py) | Mean good/bad `node_scores` over all windows versus channel. | `--per-channel-plot` |
| [`plot_per_channel_per_window_scores.py`](graphing/plot_per_channel_per_window_scores.py) | Six per-plane plots for selected good/bad windows; includes training mean/std when `scores_train.npz` exists. | `--per-channel-per-window-plot` |
| [`plot_debug.py`](graphing/plot_debug.py) | Target-good, other-good, and bad integral distributions. | Standalone; edit constants. |
| [`plot_event_hist.py`](graphing/plot_event_hist.py) | Between-event-time histogram and window-duration diagnostics. | Standalone; edit constants. |
| [`plot_mean_median_hist.py`](graphing/plot_mean_median_hist.py) | Per-window/per-channel integral mean and standard-deviation histograms. | Standalone; edit constants. |
| [`plot_pulse.py`](graphing/plot_pulse.py) | Overlay of ROOT waveform histograms. | Standalone; edit constants; requires PyROOT. |
| [`plot_time_window_box_whisker.py`](graphing/plot_time_window_box_whisker.py) | Per-run boxplots of event counts in fixed-duration windows. | Standalone; edit constants. |

`goodvsbad.png` is created by `window_score`, not by a script in `graphing/`.
The sweep's plot-only repair modes coordinate it with the two integrated
per-channel plotters.

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
| `window_mode` | `event` (default) or `time`. `event`: fixed event count per window (`window_size`), split into `n_temporal_bins` by event index. `time`: fixed elapsed-time span per window (`window_duration`), holding however many events occurred in that span — makes trigger-rate changes a direct signal instead of a fixed count regardless of elapsed time. `time` mode requires a time-like `tpc_branches` entry and `reconstruction=True` (graph VAE only, not the GNN forecaster). |
| `window_size` | Number of events per window. Used when `window_mode: event`. |
| `stride` | Number of events between consecutive windows. Used when `window_mode: event`. |
| `window_duration` | Elapsed-time span per window, same units as the time-like `tpc_branches` entry. Used when `window_mode: time`. |
| `stride_duration` | How far the window start advances between samples. Used when `window_mode: time`; defaults to `window_duration` (non-overlapping windows). |
| `n_temporal_bins` | Number of time/event bins inside one window. |
| `n_channels` | Detector channel count. |
| `channel_map` | Channel-map CSV. |
| `edge_mode` | Graph construction mode. |
| `adjacency_radius` | Radius for sequential/radius-based graph edges. |
| `node_features` | Per-channel aggregate feature list. |
| `log_features` | Features transformed before standardization. |
| `standardize` | Whether to z-score features using training statistics. |
| `standardize_by` | `global` (one pooled mean/std, default) or `plane` (separate mean/std per plane — use if `scripts/check_standardization_per_plane.py` shows a plane-biased pooled fit). |
| `min_plane_samples` | Minimum active-channel samples required to fit a plane its own stats under `standardize_by: plane`; under-sampled planes fall back to the pooled fit. Default 20. |
| `prune_inactive` | Whether inactive channels are removed from per-window graphs. |
| `graph_features` | Per-window (graph-level, not per-channel) conditioning features: `event_count`, `log1p_event_count`. Standardized like `node_features` and packed into `Data.graph_attr`; sets `model.graph_dim` automatically (no separate model key). Empty/unset = no conditioning (default). |

### `model`

| Key | Meaning |
|---|---|
| `latent_dim` | Per-node bottleneck dimension. Smaller is tighter. |
| `encoder_hidden_dims` | Encoder graph-layer widths. Length controls encoder depth. |
| `decoder_hidden_dims` | Decoder MLP widths. Length controls decoder depth. |
| `dropout` | Dropout regularization. |
| `mask_ratio` | Denoising mask fraction. |
| `use_channel_idx` | Whether to condition on channel identity. |
| `graph_dim` | Derived automatically from `data.graph_features` (length of that list) — conditions both encoder and decoder on a per-window vector (e.g. event count) via `data.graph_attr`, broadcast to every node in its graph through `data.batch`. Not meant to be set directly in config. |
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

Lessons from tuning this model on SBND runs:

- **Measure every change with the `--compare` AUC (and `--stream` latency)** — not
  the histogram shape. Change **one thing at a time**.
- **The anomaly signal lives in integral magnitude** (`sum`/`min`/`max`/`count`).
  Replacing those with occupancy/timing features *reduced* separation here; add
  occupancy/timing *on top* instead of replacing.
- **This model wants *more* regularization, not less.** Lowering `beta` or
  `mask_ratio` let the model reconstruct anomalies too and hurt separation. If
  the recon/KL panel shows no collapse, push `beta` up (1.5–3), `mask_ratio` up
  (0.2–0.3), and/or `latent_dim` down (8, 6).
- **Watch the reconstruction hist2d.** A flat horizontal band = the model is
  predicting the mean (heavy-tailed features); fix with `log_features`. After
  log-transforming, the `sum`/`min`/`max` panels should climb the `y=x` diagonal.
- **Evaluate for real time.** Window-level recall understates performance because
  bad runs are intermittently bad; use the streaming M-of-N detector's latency +
  false-alarm rate.

---

## Tests

```bash
pytest tests/
# or:
.pixi/envs/default/bin/python -m pytest tests/
```

Relevant suites include node-feature tests, window-score/evaluation tests,
channel-graph tests, and GraphVAE data/model/training tests.

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
data/events_train.npz
data/good_events_test.npz
data/bad_events_test.npz
```

The training input is set in each generated YAML through the copied base config,
usually from `data.events_path` in `configs/graph_vae.yaml`. The sweep's
`--train-input` is a separate path used only for training-baseline inference.

`dataset_preparation/train_test_from_npz.py` currently writes
`windows_train.npz`, `good_runs.npz`, and optionally `bad_runs.npz` under its
configured output directory. Either move/name those files to match the defaults,
or pass their real paths through the YAML, `--train-input`, `--good-input`, and
`--bad-input`. Do not assume the dataset-preparation output names and sweep
defaults are automatically synchronized.

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
  --eval-percentile 95 \
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

With the optional graphing hooks, the inference directory also receives
`channel_mean_node_scores.png` and six
`channel_node_scores_selected_windows_plane_<1-6>.png` files. The JSON summary
is written by the sweep runner from the captured `window_score` log and recorded
metrics; it is not a direct `window_score` CLI output.

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

# Recreate only plot sets that are missing; do not train or infer
python run_graph_vae_sweep.py --missing-plot

# Recreate all managed plots from existing scores, even when they exist
python run_graph_vae_sweep.py --force-replot
```

`--missing-infer-only` implies `--infer-only`, and
`--missing-evaluate-only` implies `--evaluate-only`. Evaluation-only and
inference-only modes are mutually exclusive; evaluation-only also cannot be
combined with `--skip-eval`. `--missing-plot` and `--force-replot` are mutually
exclusive standalone modes and cannot be combined with train/infer/evaluate-only,
baseline, skip-infer, or database rewrite modes.

### Sweep-managed training baseline

The sweep can automate the three-step baseline workflow described earlier:

```bash
python run_graph_vae_sweep.py \
  --infer-only \
  --baseline-infer \
  --train-input data/events_train.npz \
  --baseline-eval-percentile 90 \
  --batch
```

For each trained model, `--baseline-infer` replaces normal good/bad inference
with this sequence:

1. Infer `--train-input` into `inference_result/scores_train.npz`.
2. Run `--baseline-cmd` to create `channel_baseline.npz`.
3. Evaluate the **existing** `scores_good.npz` and `scores_bad.npz` with
   `zscore_mean`, producing `goodvsbad_zscore.png`,
   `goodvsbad_zscore_eval.txt`, and `goodvsbad_zscore_eval.json`.

The model must already have ordinary held-out good/bad score files unless
`--skip-eval` is used. Baseline mode works with `--infer-only` and
`--missing-infer-only`; it cannot be combined with evaluation-only,
`--skip-infer`, `--restore-missing-db`, or either per-channel plotting hook.
The normal evaluation percentile is now **95**. Baseline evaluation has an
independent `--baseline-eval-percentile` setting whose current code default is
`90`.

### Complete sweep flag reference

The tables use the canonical hyphenated spellings. Where aliases exist, they are
shown in the same row.

For completeness, the legacy/mixed-separator aliases accepted by the parser are:
`--evaluate_only`, `--eval_only`, `--missing_evaluate_only`,
`--missing_eval_only`, `--infer_only`, `--inference_only`,
`--missing_infer_only`, `--missing_infer`, `--missing-infer_only`,
`--missing_infer-only`, `--per_channel_plot`, `--per_channel-plot`,
`--per-channel_plot`, `--per-channel-plots`, `--per_channel_plots`,
`--per-channel_plots`, `--per_channel-plots`, `--restore_missing_db`,
`--restore_missing-db`, `--restore-missing-database`,
`--restore_missing_database`, `--restore-missing_database`, and
`--restore_missing-database`. Prefer the canonical spellings shown in the tables
for new commands.

#### Paths and commands

| Flag | Default | Meaning |
|---|---|---|
| `--config-dir PATH` | `tuning_configs/graph_vae_sweep` under `PROJECT_DIR` | Directory searched for sweep YAMLs. |
| `--runs-root PATH` | `checkpoints/graph_vae` under `PROJECT_DIR` | Parent directory for per-config run outputs. |
| `--db-path PATH` | `graph_vae_sweep.sqlite3` under `PROJECT_DIR` | SQLite experiment database. |
| `--good-input PATH` (`--good_input`) | `data/good_events_test.npz` | Held-out good sparse-events NPZ for normal inference. |
| `--bad-input PATH` (`--bad_input`) | `data/bad_events_test.npz` | Bad sparse-events NPZ for normal inference. |
| `--train-input PATH` (`--train_input`) | `data/events_train.npz` | Training-split good NPZ inferred by `--baseline-infer`; this does not change the YAML's training data. |
| `--pattern GLOB` | `*.yaml` | Config filename pattern within `--config-dir`. |
| `--train-cmd COMMAND` | `sbn-train` | Training executable/command string. |
| `--infer-cmd COMMAND` | `sbn-infer` | Inference executable/command string. |
| `--baseline-cmd COMMAND` (`--baseline_cmd`) | `python -m sbn_anomaly.infer.channel_baseline` | Converts `scores_train.npz` to `channel_baseline.npz`. |
| `--eval-cmd COMMAND` | `python -m sbn_anomaly.infer.window_score` | Evaluation command; the sweep appends score paths, labels, aggregator, map, threshold/percentile, and plot arguments. |

#### Evaluation

| Flag | Default | Meaning |
|---|---:|---|
| `--eval-aggregator NAME` (`--eval_aggregator`) | `group_max_mean` | Normal good/bad `window_score` aggregator. |
| `--channel-map PATH` (`--channel_map`) | project SBN channel-map CSV | Map passed to normal `window_score` evaluation. In the sweep this does not override the integrated plotter's own map setting. |
| `--eval-threshold VALUE` (`--eval_threshold`) | none | Explicit normal-evaluation threshold; overrides `--eval-percentile`. |
| `--eval-percentile P` (`--eval_percentile`) | `95` | Good-score percentile passed to normal `window_score` when no explicit threshold is supplied. |
| `--eval-per-run` (`--eval_per_run`) | off | Add `--per-run` to normal or baseline `window_score` evaluation. |
| `--good-label TEXT` (`--good_label`) | `good` | Primary-sample label. |
| `--bad-label TEXT` (`--bad_label`) | `bad` | Comparison/positive-sample label. |
| `--eval-plot-name NAME` (`--eval_plot_name`) | `goodvsbad.png` | Normal evaluation plot filename within `inference_result`. |
| `--eval-log-name NAME` (`--eval_log_name`) | `goodvsbad_eval.txt` | Normal evaluation log filename. |
| `--eval-json-name NAME` (`--eval_json_name`) | `goodvsbad_eval.json` | Normal evaluation JSON summary filename. |
| `--baseline-eval-percentile P` (`--baseline_eval_percentile`) | `90` | Good z-score percentile used only after `--baseline-infer`. |
| `--baseline-eval-plot-name NAME` (`--baseline_eval_plot_name`) | `goodvsbad_zscore.png` | Baseline evaluation plot filename. |
| `--baseline-eval-log-name NAME` (`--baseline_eval_log_name`) | `goodvsbad_zscore_eval.txt` | Baseline evaluation log filename. |
| `--baseline-eval-json-name NAME` (`--baseline_eval_json_name`) | `goodvsbad_zscore_eval.json` | Baseline evaluation JSON filename. |

#### Execution modes

| Flag | Meaning |
|---|---|
| `--evaluate-only` (`--eval-only` and underscore forms) | Skip training and inference; reevaluate existing `scores_good.npz` and `scores_bad.npz`. |
| `--missing-evaluate-only` (`--missing-eval-only` and underscore forms) | Evaluation-only, but only when the configured plot, log, or JSON is missing. |
| `--skip-eval` (`--skip_eval`) | Run the applicable train/inference stages without good-vs-bad evaluation. |
| `--skip-infer` | Train only; do not run good/bad inference. |
| `--infer-only` (`--inference-only` and underscore forms) | Skip training and rerun inference using each run's existing `graph_vae_final.pt`. |
| `--baseline-infer` (`--baseline_infer`) | Switch inference to training-score baseline creation and z-score evaluation. |
| `--missing-infer-only` (`--missing-infer` and underscore forms) | Imply inference-only and process only runs missing required outputs. Normal mode checks good/bad scores; baseline mode checks training scores, baseline, and—unless evaluation is skipped—z-score plot/log/JSON. |

#### Integrated graphing and plot repair

| Flag | Default | Meaning |
|---|---:|---|
| `--per-channel-plot` (underscore, mixed-separator, and plural aliases) | off | After an actual successful good/bad inference pair, run `graphing/plot_per_channel_scores.py`. It does not run in evaluation-only mode. |
| `--per-channel-per-window-plot` (`--per_channel_per_window_plot`) | off | After an actual good/bad inference pair, run `graphing/plot_per_channel_per_window_scores.py`. |
| `--per-channel-window-indices SPEC` (`--per_channel_window_indices`) | plotter default | Zero-based index/range expression such as `1200`, `1,4,8`, or inclusive `1200-1210`. |
| `--per-channel-window-datasets {good,bad,both}` (`--per_channel_window_datasets`) | `both` | Which held-out score files the selected-window plotter loads. |
| `--per-channel-per-window-plot-name NAME` (`--per_channel_per_window_plot_name`) | `channel_node_scores_selected_windows.png` | Base name; the plotter inserts `_plane_1` through `_plane_6`. |
| `--missing-plot` (`--missing_plot`) | off | Standalone repair mode: independently recreate missing all-window channel plot, six selected-window plane plots, and `goodvsbad.png` from existing scores. |
| `--force-replot` (`--force_replot`) | off | Standalone mode that recreates all three managed plot sets whenever good/bad score files exist. |

The sweep does not forward the selected-window plotter's direct CLI options
`--channel-map`, `--plot-style`, or `--good-bad-only`. Configure those by running
the plotter directly. If `scores_train.npz` is present, the integrated
selected-window plots automatically include its green training mean/std band.

#### Runtime, database, and export controls

| Flag | Default | Meaning |
|---|---:|---|
| `--timeout SECONDS` | none | Timeout for each train/infer subprocess. |
| `--stop-on-error` | off | Stop after the first failed config instead of continuing. |
| `--monitor-interval N` (`--monitor_interval`) | `30` | Print subprocess CPU/RAM usage every `N` seconds; `0` disables monitoring. |
| `--batch` | off | Set `SBN_BATCH=1` and `TQDM_DISABLE=1` for log-friendly subprocess output. |
| `--force-rewrite` (`--force_rewrite`) | off | Delete an existing matching database record and rerun, overwriting that run directory's outputs. |
| `--missing-rewrite` (`--missing_rewrite`) | off | For an existing database record, rerun only when its run directory is missing or incomplete. |
| `--restore-missing-db` (database and underscore aliases) | off | Reconstruct missing SQLite rows from completed on-disk run directories, including ignored directories, then reorder experiment IDs by sorted YAML order. Runs no train, inference, evaluation, or plotting. |
| `--export-summary` (`--export_summary`, `--export-csv-xlsx`, `--export_csv_xlsx`) | off | Export `graph_vae_sweep_summary.csv` and `.xlsx`; SQLite is always maintained. |
| `-h`, `--help` | — | Show all accepted spellings and command help. |

Model directories listed in the script's `IGNORED` constant are skipped during
normal sweep activity. Restore mode intentionally still considers them.

---

# Important experiments and updates that are yet to be done
- 