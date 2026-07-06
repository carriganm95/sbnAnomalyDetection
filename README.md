# sbnAnomalyDetection — Graph VAE

Real-time data-quality-monitoring (DQM) anomaly detection for liquid-argon TPCs
(SBND / ICARUS, the [SBN](https://sbn.fnal.gov/) program) using a **graph
variational autoencoder** on reconstructed hit features.

The model is trained on **good runs only** and reconstructs one short window of
detector activity at a time. Windows that reconstruct poorly are anomalous, so
problems can be flagged from a small amount of data — no waiting for a whole run.

- Model type: `graph_vae`
- Config: [`configs/graph_vae.yaml`](configs/graph_vae.yaml)
- Train: `sbn-train --config configs/graph_vae.yaml`
- Score: `sbn-infer --config configs/graph_vae.yaml --input <events.npz> --output scores.npz`
- Evaluate: `python -m sbn_anomaly.infer.window_score ...`

---

## How the network works

### The problem

A window of events is summarized as a **graph**: nodes are TPC channels, node
features are per-channel aggregates of the reconstructed hits in that window, and
edges connect channels that share readout electronics. The autoencoder learns to
reconstruct the nominal (good-run) feature graph; on anomalous data the
reconstruction error rises. Because it compares each window to the **absolute
learned nominal** (not to a forecast of recent history), a persistently-bad
condition stays flagged and there is no warm-up latency.

### It is a single model, not two stages

The encoder and decoder **are** graph-neural-network layers — message passing on
the electronics graph happens *inside* the autoencoder, trained end-to-end as a
β-VAE. It is **not** "a GNN followed by an autoencoder."

```
one window  ->  per-channel feature graph
                 nodes   = channels
                 features= [sum, min, max, mean, stdev, count, ...] per sub-window bin
                 edges   = electronics graph (ASIC cliques + FEMB readout chain)
        │
        ▼
  SAGEConv encoder        message passing keeps a self-term, so a single bad
        │                 channel is not smoothed away by its board-mates
        ▼
  per-node variational bottleneck   (mu, logvar -> z)   ← masking: a fraction of
        │                                                 nodes are rebuilt from
        ▼                                                 neighbours (denoising)
  decoder -> reconstruct node features
        │
        ▼
  per-channel reconstruction error
        │  aggregate over channels (group_max_mean, ...)
        ▼
  per-window anomaly score  ->  streaming M-of-N rule  ->  real-time flag
```

### Why a *variational* autoencoder

The bottleneck, the KL term, and the masking all constrain the model to the
nominal manifold, so it reconstructs good windows well but **fails on
anomalies** — which is exactly the signal you want. A plain, high-capacity
autoencoder tends to reconstruct anomalies too, giving no separation. The KL
regularization (β) and a tight `latent_dim` are the main knobs that keep the
model from memorizing anomalies (see [Tuning](#tuning-notes)).

### Scoring, from channel to real-time alarm

1. **Per-channel error** — reconstruction MSE per channel per window
   (`node_scores`). Localizes *which* channels/boards are off.
2. **Per-window score** — aggregate the channel errors (default
   `group_max_mean`: pool error within each ASIC/FEMB group and take the worst
   group, which is ideal for coherent board/ASIC faults).
3. **Real-time decision** — a streaming *M-of-N* persistence rule over the last
   few windows flags a developing problem within a few windows (not a whole
   run), trading a little latency for far fewer false alarms.

---

## Installation

```bash
pixi install
pixi shell
# or:  pip install -e ".[dev]"
```

Requires Python ≥ 3.9, PyTorch, PyTorch-Geometric, uproot, awkward, numpy,
pandas, matplotlib.

---

## Data processing

Inference and training consume a **compact sparse-events `.npz`** (not a dense
array). You build it once from reconstructed ROOT files; windows are then formed
on the fly. You never need to materialize dense windows.

### 1. Point at your ROOT files

A file list is a text file with one ROOT path per line (`#` comments allowed;
`.txt`/`.lst`/`.list`/`.csv` recognized):

```
# data/good_train.txt
/pnfs/icarus/persistent/users/micarrig/DQM/19305/reco/run19305_evt0.root
/pnfs/.../reco/run19305_evt1.root
```

### 2. Materialize the events npz

`materialize_windows` (default, no `--windows` flag) writes the compact sparse
events file. Use the **same config** you train/score with so the hit branches,
window definition, and channel count match:

```bash
python -m sbn_anomaly.data.materialize_windows --config configs/graph_vae.yaml \
    --root-file-list data/good_train.txt --output data/good_events_train.npz
```

Do this separately for your good-training, good-test, and bad-test sets:

```bash
python -m sbn_anomaly.data.materialize_windows --config configs/graph_vae.yaml \
    --root-file-list data/good_test.txt --output data/good_events_test.npz
python -m sbn_anomaly.data.materialize_windows --config configs/graph_vae.yaml \
    --root-file-list data/bad_test.txt  --output data/bad_events_test.npz
```

> `--root-file-list` takes a manifest of ROOT paths; `--input` (on `sbn-infer`)
> takes the resulting `.npz`. They are different stages — don't pass a file list
> to `--input`.

### Sparse event format

The events npz stores hits in CSR form (only channels with hits are kept, ~1000×
smaller than dense):

| Key | Dtype | Meaning |
|-----|-------|---------|
| `channels_flat` | int64 | channel ids of all hits, concatenated over events |
| `integrals_flat` | float32 | hit integrals (same order) |
| `times_flat` | float32 | hit times (enables timing features) |
| `offsets` | int64 | event `i` = hits `channels_flat[offsets[i]:offsets[i+1]]` |
| `n_channels` | int64 | detector channel count |
| `evt_run/subrun/num` | int32 | per-event provenance (run, subrun, event) |

Merge several events npz correctly (offsets must be shifted — a plain
`np.concatenate` corrupts them):

```bash
python -m sbn_anomaly.data.merge_events --output data/all_good.npz --glob 'data/good_*.npz'
```

See [`data/README.md`](data/README.md) for storage locations and known
good/bad run lists.

---

## Training

Trains on good runs only; from an events npz (`data.events_path`), from ROOT
directly (`--root-file-list`), or from a dense windows array (`data.windows_path`).

```bash
# from a cached events npz (data.events_path in the config)
sbn-train --config configs/graph_vae.yaml

# or stream straight from ROOT
sbn-train --config configs/graph_vae.yaml --root-file-list data/good_train.txt
```

Outputs (under `training.checkpoint_dir`):

- `graph_vae_final.pt` — model weights.
- `standardization.npz` — per-feature mean/std used at inference (must match).
- `training_curves.png` — four panels: **loss**, **recon / KL (with β on a twin
  axis)**, **score percentiles**, **throughput**. The recon/KL panel is how you
  diagnose posterior collapse (KL → 0 while recon plateaus).
- `plots/reconstruction_hist2d*.png` — per-feature original-vs-reconstruction
  histograms. On a healthy model these track the `y=x` diagonal; a flat band
  means the model is predicting the mean (see [Tuning](#tuning-notes)).

---

## Inference / scoring

```bash
sbn-infer --config configs/graph_vae.yaml --input data/good_events_test.npz --output scores_good.npz
sbn-infer --config configs/graph_vae.yaml --input data/bad_events_test.npz  --output scores_bad.npz
```

`--input` overrides `inference.input_path`. Output npz:

| Array | Shape | Description |
|-------|-------|-------------|
| `node_scores` | `(W, C)` | per-window per-channel reconstruction error (NaN = inactive) |
| `scores` | `(W,)` | per-window aggregated score (default `group_max_mean`) |
| `channel_mean_error` / `channel_max_error` | `(C,)` | per-channel summary over all windows (which channels are worst) |
| `channel_active_frac` | `(C,)` | fraction of windows each channel was active |
| `provenance` | `(W, 3)` | `(run, subrun, event)` of each window |
| `is_anomaly` | `(W,)` | present only when `inference.threshold` is set |

It also writes score-distribution, score-over-time, and per-channel-error PNGs
next to the output, and logs the top-10 worst channels.

---

## Evaluation

All evaluation runs on the saved scores, so you can re-aggregate and re-threshold
without re-running the model.

### Good-vs-bad separation

```bash
python -m sbn_anomaly.infer.window_score \
    --scores scores_good.npz --compare scores_bad.npz --labels good bad \
    --aggregator group_max_mean --channel-map configs/SBNDTPCChannelMap_v2_with_positions.csv \
    --plot goodvsbad.png
```

Prints per-set score stats, a threshold-free **separation AUC** (Mann-Whitney,
tie-correct), the **confusion matrix + precision / recall / F1 / FPR** at the
operating threshold (good-set p99 unless `--threshold`), and an overlaid
histogram with the threshold line.

### Real-time detection (the operational metric)

DQM anomalies are usually *intermittent* at the window level, so window-level
recall looks low even when runs are clearly separable. Judge the model by
**streaming detection latency** instead:

```bash
python -m sbn_anomaly.infer.window_score \
    --scores scores_good.npz --compare scores_bad.npz --labels good bad \
    --aggregator group_max_mean --channel-map configs/SBNDTPCChannelMap_v2_with_positions.csv \
    --stream --persist-n 3 --persist-m 2
```

A window fires when `M` of the last `N` windows are over threshold (or a single
window exceeds `--instant-threshold`). It reports **false-alarm rate on good
runs** and **detection rate + latency (in windows) on bad runs** — i.e. how few
windows until a problem is flagged, with no full-run wait.

Aggregators (swap with `--aggregator`): `mean`, `max`, `topk_mean`,
`group_max_mean` (default). Requires `provenance` in the scores npz (inference
saves it). `--per-run` gives a whole-run rollup if you also want that.

---

## Node features

Node features are per-channel aggregates, computed per temporal bin
(`F = n_temporal_bins × len(node_features)`). Set them in `data.node_features`:

| Feature | Meaning |
|---------|---------|
| `sum`, `min`, `max`, `mean`, `stdev`, `count` | moments of the hit integral |
| `occupancy` | fraction of events in the bin with ≥1 hit (dead/hot channels) |
| `hit_rate` | average hits per event |
| `time_mean`, `time_spread` | mean / spread of hit time (needs a time branch) |

### `log_features` (heavy-tail transform)

Integral-magnitude features (`sum`/`min`/`max`/`mean`) are heavy-tailed even
after z-scoring, which makes the VAE reconstruct them as a flat mean. List them
in `data.log_features` to apply a **sign-preserving `log1p`** (`sign(x)·log1p(|x|)`,
so negative induction-plane integrals are handled) **before** standardization, so
the model can actually track them. Must be a subset of `node_features`, and must
be the **same for training and inference**.

Features are then **z-scored** on good-run statistics (`standardize: true`), so
reconstruction error means "deviation from nominal in sigmas."

---

## Graph construction

`data.edge_mode` selects how channels are connected (built from
`data.channel_map`, the SBND channel map with electronics + geometry):

- `electronics` (recommended) — ASIC cliques (16-channel) + per-FEMB readout-chain
  adjacency. This is where coherent noise and board/ASIC failures correlate; it
  deliberately crosses wire planes because a FEMB spans planes.
- `wire` — adjacent wires within each (plane, TPC, side).
- `both` — union of the two.
- `sequential` — connect channels within `±adjacency_radius` of each other in
  offline-channel id (no channel map needed).

---

## Configuration reference (`configs/graph_vae.yaml`)

**`data:`**
- `events_path` / `windows_path` — cached input (events npz preferred).
- `window_size`, `n_temporal_bins`, `stride` — a window = `window_size` events in
  `n_temporal_bins` bins; a new window every `stride` events. Sets decision
  granularity / latency.
- `n_channels` — detector channel count (SBND ≈ 11276).
- `channel_map`, `edge_mode` — graph construction (above).
- `node_features`, `log_features` — features (above).
- `standardize` — z-score features on good-run stats.
- `prune_inactive` — drop channels with no hits from each window's graph.

**`model:`**
- `latent_dim` — per-channel bottleneck size (smaller = tighter, harder to
  memorize anomalies).
- `encoder_hidden_dims` — variable-width graph encoder dimensions. The number of
  message-passing layers is `len(encoder_hidden_dims)`. For example,
  `[128, 64, 32]` builds `enc_in -> 128 -> 64 -> 32 -> mu/logvar`.
- `decoder_hidden_dims` — variable-width decoder MLP dimensions. The number of
  decoder hidden layers is `len(decoder_hidden_dims)`. For example, `[32, 64]`
  builds `latent -> 32 -> 64 -> reconstructed features`; use `[]` for a direct
  `latent -> reconstructed features` decoder.
- `dropout`, `mask_ratio` — regularization; `mask_ratio` is the denoising fraction.
- `use_channel_idx` — feed channel id as conditioning (learn per-channel baselines).
- `conv` — `sage` (self-preserving, recommended) or `gcn`.

Example variable-layer model block:

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

> (Important): Keep training and inference on the same YAML architecture, 
               or checkpoint loading will fail with size mismatches.

**`training:`**
- `lr`, `weight_decay`, `batch_size`, `max_epochs`, `validation_split`.
- `beta` — KL weight (β-VAE); `beta_warmup_epochs` — linear ramp 0→β.
- `use_amp`, `save_best_only`, `checkpoint_dir`, `output_path`.

**`inference:`**
- `checkpoint_path`, `input_path`, `batch_size`.
- `window_aggregator`, `group_level`, `topk`, `threshold`, `plot`.

---

## Tuning notes

Lessons from tuning this model on SBND runs:

- **Measure every change with the `--compare` AUC (and `--stream` latency)** — not
  the histogram shape. Change **one thing at a time**.
- **Tune layer widths with `encoder_hidden_dims` and `decoder_hidden_dims`.** The
  encoder list controls both width and graph depth; deeper/wider encoders can
  learn richer board/channel context but may over-smooth or memorize. The decoder
  list controls reconstruction capacity; too much decoder capacity can reduce
  anomaly separation by reconstructing bad windows too well.
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
# or point at the pixi python
.pixi/envs/default/bin/python -m pytest tests/
```

Relevant suites: `test_node_features.py` (occupancy/timing/log features),
`test_window_score.py` (aggregators, AUC, confusion), `test_channel_graph.py`
(electronics graph), plus the data/model/train tests.
