# GNN Workflow: Quick Reference & Key Questions

## 1. How Input Data Flows Into the Network

### Step-by-Step

```
Step 1: ROOT FILES
  └─ High-energy physics detector data
     Each event contains hits: (channel_id, integral_value)

Step 2: MATERIALIZATION (materialize_windows.py)
  ├─ Stream events from ROOT files
  ├─ Group hits by channel (aggregate per channel per event)
  ├─ Create rolling windows: N consecutive events together
  │  └─ window_size=20 events per window
  │     (each event ~1-100 hits from different channels)
  ├─ Within each window, create temporal bins
  │  └─ n_temporal_bins=4 (finer time resolution within the window)
  ├─ For each channel in each temporal bin, compute statistics
  │  └─ features: [sum, min, max, mean, stdev, count]
  │     (6 statistics total per channel per bin)
  ├─ Discover max channel ID from ROOT data → number of channels C
  │  └─ For your detector: C = 11,276 channels
  └─ Save to windows.npz: shape (N_windows, C, 4, 6)

Step 3: DATASET PREPARATION (GraphWindowDatasetPyG)
  ├─ Load windows.npz with shape (N_windows, C, 4, 6)
  ├─ Create training samples by:
  │  ├─ Take history=4 consecutive windows (frame t-4 to t-1)
  │  ├─ These form the input (context of past behavior)
  │  └─ Window at t is the target (what to predict)
  ├─ Flatten temporal dimensions
  │  ├─ 4 windows × (4 bins × 6 features) = 4 × 24 = 96 values per channel
  │  ├─ Prepend normalized channel index (1 value)
  │  └─ Result: (C channels, 97 values) = x  [where C = 11,276 for your data]
  ├─ Prune inactive channels (optional optimization)
  │  └─ If a channel has near-zero activity, remove it temporarily
  ├─ Create sparse graph with adjacency
  │  └─ Each channel connects to neighbors within radius=4
  ├─ Build PyG Data object
  │  └─ x: node features
  │     y: target (next window)
  │     edge_index: sparse connectivity
  └─ DataLoader batches these objects

Step 4: FORWARD PASS (GNNForecasterPyG)
  ├─ Input: PyG Batch with x(total_nodes, 97), y(total_nodes, 24), edge_index
  ├─ Decompress temporal:
  │  ├─ Split x into: channel_idx(1) + temporal(96)
  │  ├─ Reshape temporal from (96,) to (4 timesteps, 24 features)
  │  └─ Now in explicit temporal form: [t_0, t_1, t_2, t_3]
  ├─ Per-frame spatial encoding (loop 4 times):
  │  ├─ Combine: [channel_idx, temporal[t]]  (25 features)
  │  ├─ Pass through 3 stacked GCN layers
  │  │  └─ GCN: node features + edge connectivity → aggregation from neighbors
  │  │     Each layer: apply message-passing (includes ReLU + Dropout)
  │  └─ Output: h_t (128 features per node)
  ├─ Stack all 4 frames: enc_seq (4 timesteps, num_nodes, 128)
  ├─ Temporal modeling with GRU:
  │  ├─ GRU processes sequence of frame encodings
  │  ├─ Output: out_seq (4 timesteps, num_nodes, 256)
  │  └─ Take last timestep: (num_nodes, 256)
  ├─ Decoder:
  │  ├─ Linear projection: 256 → 24 features
  │  └─ pred: (num_nodes, 24) = prediction for next window
  └─ Return prediction

Step 5: ANOMALY SCORE CALCULATION
  ├─ Compare prediction vs. actual next window
  ├─ Per-node reconstruction error:
  │  └─ MSE = mean((pred - y)^2)  for each channel
  ├─ Aggregate to per-window:
  │  ├─ window_mean = average MSE across all channels
  │  └─ window_max  = worst (highest) MSE channel
  └─ Interpretation:
      Low score (< 0.25) = prediction matched reality = normal
      High score (> 0.25) = prediction failed = anomalous
```

---

## 2. How Network Pieces Work Together

### Information Flow Diagram

```
┌─────────────────────────────────────────────────────┐
│ GCN (Graph Convolutional Network)                   │
│                                                     │
│ Purpose: Aggregate spatial information              │
│ Input: Per-node features + graph structure          │
│ Output: Aggregated features incorporating neighbors │
│                                                     │
│ Operation:                                          │
│   For each node:                                    │
│   ├─ Collect features from neighboring nodes       │
│   ├─ Weight them by learned parameters             │
│   ├─ Sum aggregations                              │
│   └─ Apply ReLU activation                         │
│                                                     │
│ Applied per timestep (4 times, once per frame)     │
│ Stacked 3 layers deep for multi-hop aggregation    │
└─────────────────────────────────────────────────────┘
             ↑      (learns spatial patterns)
             │
┌─────────────────────────────────────────────────────┐
│ GRU (Gated Recurrent Unit)                          │
│                                                     │
│ Purpose: Model temporal dynamics & make prediction  │
│ Input: Sequence of spatially-encoded frames (4)     │
│ Output: Hidden state capturing temporal context    │
│                                                     │
│ Operation:                                          │
│   Process frame 1: hidden_1 = GRU(frame_1, None)   │
│   Process frame 2: hidden_2 = GRU(frame_2, hidden_1)
│   Process frame 3: hidden_3 = GRU(frame_3, hidden_2)
│   Process frame 4: hidden_4 = GRU(frame_4, hidden_3)
│   ↓                                                 │
│   hidden_4 contains all temporal knowledge          │
│                                                     │
│ Key idea: GRU gates control what info to keep/     │
│           forget, learning task-relevant dynamics  │
└─────────────────────────────────────────────────────┘
             ↑      (learns temporal patterns)
             │
┌─────────────────────────────────────────────────────┐
│ MSE Loss (Mean Squared Error)                       │
│                                                     │
│ Purpose: Train the model to predict next window    │
│ Loss = mean((pred - actual)^2)                      │
│                                                     │
│ During training:                                    │
│   ├─ Backprop through all layers                   │
│   ├─ GCN learns: which neighbor patterns matter    │
│   └─ GRU learns: how patterns evolve over time     │
│                                                     │
│ During inference:                                   │
│   └─ MSE becomes anomaly score (no gradients)      │
│                                                     │
│ Interpretation:                                     │
│   The model is trained to FORECAST normal behavior │
│   Large prediction error → behavior is abnormal    │
└─────────────────────────────────────────────────────┘
```

### Cooperation Example

**Scenario: Detecting dead/stuck electronics**

```
Normal detector:
  t-4: scattered hits across channels (~50-100 hits)
  t-3: scattered hits (similar pattern)
  t-2: scattered hits (similar pattern)
  t-1: scattered hits (similar pattern)
  t: predict scattered hits ← model confident
  Actual t: scattered hits (as predicted)
  MSE: LOW ✓ Normal

Stuck electronics (one channel always max):
  t-4: stuck_ch=5000, others scattered
  t-3: stuck_ch=5000, others scattered  ← MODEL LEARNS PATTERN
  t-2: stuck_ch=5000, others scattered  ← GRU captures: "ch 5000 persistent"
  t-1: stuck_ch=5000, others scattered  ← GCN propagates via neighbors
  t: predict stuck_ch=5000 OR scattered ← MODEL CONFUSED
      (it learned stuck pattern, but real physics says should be normal)
  Actual t: normal distribution (electronics fixed)
  MSE: HIGH ✗ Anomalous

How it catches it:
  ├─ GCN: "channel 5000 and neighbors are correlated"
  ├─ GRU: "this correlation is persistent across frames"
  ├─ Decoder: "expects channel 5000 to be high"
  └─ But actual is normal → prediction error
```

---

## 3. Data Compression/Decompression

### Compression (Training Data Preparation)

```
Original window representation:
  Window: (256 channels, 4 temporal_bins, 6 features)
  Size: 256 × 4 × 6 = 6,144 values per window

Compression approach (online flattening):
  Time → spatial: (4, 6) → (24,)  [4 bins × 6 features]
  Combine history: 4 windows × 24 = 96 values per channel
  Add channel metadata: +1 normalized channel index
  Result: (256, 97) = 24,832 values per sample
  
  OR if pruned (M active channels):
  Result: (M, 97) where M ≤ 256
  
  Savings from pruning: ~30-50% depending on event sparsity
```

### Decompression (Model Forward Pass)

```
Compressed input: (num_nodes, 1 + 96)

In model.forward():
  
  1. Extract components:
     channel_idx = x[:, 0:1]      (num_nodes, 1)
     temporal = x[:, 1:]           (num_nodes, 96)
  
  2. Shape temporal back to explicit frames:
     temporal = temporal.view(num_nodes, T=4, F=24)
     Now: (num_nodes, 4, 24)
     
  3. Loop per frame:
     for t in range(4):
       x_t = temporal[:, t, :]     (num_nodes, 24)
       ↓
       x_t_full = concat([channel_idx, x_t])  (num_nodes, 25)
       ↓
       h_t = gcn_layers(x_t_full)  (num_nodes, 128)
       ↓
     Save h_t for each t
  
  4. Stack all frames:
     enc_seq = stack(h_0, h_1, h_2, h_3)  (4, num_nodes, 128)
  
  5. Temporal modeling:
     out_seq = gru(enc_seq)  (4, num_nodes, 256)
     
  6. Final prediction:
     pred = decoder(out_seq[-1])  (num_nodes, 24)
```

**Key insight**: Decompression happens on GPU during forward pass—no CPU memory overhead. The (96,) feature vector is only flattened; actual tensors maintain structure via views and reshapes.

---

## 4. Anomaly Score Calculation: Final Details

### Per-Node MSE

```python
per_node_mse = ((pred - target) ** 2).mean(dim=-1)
#              ^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^
#              element-wise squared        average over
#              error                       24 features

Shape: (num_nodes,)
Meaning: reconstruction error at each channel

Example:
  Channel 0: pred=[0.5, 1.0, 2.1, ...]  target=[0.4, 1.1, 2.0, ...]
             squared_errors=[0.01, 0.01, 0.01, ...]
             mse = 0.0067 ← low score (normal)
  
  Channel 100: pred=[0.0, 0.0, 0.0, ...] target=[1.0, 2.0, 3.0, ...]
               squared_errors=[1.0, 4.0, 9.0, ...]
               mse = 4.67 ← high score (anomalous)
```

### Per-Window Aggregation

```python
# For a window with M active nodes:

window_mean = per_node_mse.mean()
# Average error across all channels
# Detects: distributed anomalies (bad everywhere)

window_max = per_node_mse.max()
# Worst channel in the window
# Detects: localized anomalies (bad in one place)

both_returned = [window_mean, window_max]
# Downstream decision: alert if either > threshold
```

### Alert Decision

```python
threshold = 0.25  # tunable based on False Alarm Rate

# Single threshold:
if window_max > threshold:
    ALERT()  # conservative (catches any outlier)

# Dual threshold (typical):
if (window_max > threshold) OR (window_mean > 0.5 * threshold):
    ALERT()  # catches both localized and distributed issues

# Multi-branch (your system):
if (tpc_max > 0.25) OR (pmt_max > 0.25) OR (window_max > 0.25):
    ALERT()  # fires if ANY detector branch anomalous
```

---

## 5. Key Hyperparameters & Their Effect

| Hyperparameter | Value | Effect |
|---|---|---|
| `history` | 4 | Use 4 past windows to predict next; larger = more context |
| `gnn_hidden` | 128 | GCN output size; larger = more capacity but slower |
| `gnn_layers` | 3 | Number of GCN layers; more = deeper aggregation, risk of over-smoothing |
| `gru_hidden` | 256 | GRU hidden state size; larger = more temporal memory |
| `gru_layers` | 1 | Number of GRU layers; 1 is typical for small history |
| `dropout` | 0.1 | Regularization; higher = simpler model, lower = more capacity |
| `adjacency_radius` | 4 | Connect each channel to ±4 neighbors; larger = denser graph |
| `window_size` | 20 | Events per materialized window; larger = coarser temporal binning |
| `n_temporal_bins` | 4 | Fine-structure bins within a window; larger = finer temporal resolution |
| `anomaly_threshold` | 0.25 | MSE threshold for alert; tuned to balance sensitivity vs false alarms |

---

## 6. Common Failure Modes & How to Debug

| Symptom | Likely Cause | Debug Steps |
|---|---|---|
| All windows score ~0 | Model not training well or threshold too high | Check loss curve; verify train/test split; lower threshold |
| All windows score > threshold | Overfit to training data; or threshold too low | Increase regularization; verify data split; raise threshold |
| Missing anomalies | Model underfitting; or threshold too high | Add capacity (more layers); lower threshold; check data quality |
| False alarms on good runs | Model overfitting; or threshold too low | Early stopping; increase dropout; adjust threshold based on FAR |
| Per-channel NaN scores | Node pruning is too aggressive | Disable `prune_inactive` or adjust pruning threshold |
| Batch size effects | Batch normalization sensitivity | Use LayerNorm instead; or increase batch size |

---

## 7. Testing the Workflow Offline

```python
# Quick test in a notebook:

import torch
from sbn_anomaly.models.gnn_forecaster_pyg import GNNForecasterPyG
from sbn_anomaly.data.graph_window_dataset_pyg import GraphWindowDatasetPyG
from torch_geometric.loader import DataLoader

# Load materialized windows
windows = np.load("windows_train.npz")['windows']  # (N, C, 4, 6)

# Create dataset
dataset = GraphWindowDatasetPyG(
    windows=windows,
    history=4,
    radius=4,
)

# Create model
model = GNNForecasterPyG(
    frame_feat_dim=6,
    target_dim=24,
    history=4,
    gnn_hidden=128,
    gnn_layers=3,
)

# Single forward pass
batch = DataLoader(dataset, batch_size=32).__iter__().__next__()
pred = model(batch.to('cpu'))

print(f"Prediction shape: {pred.shape}")
print(f"Target shape: {batch.y.shape}")

# Compute MSE
mse = ((pred - batch.y) ** 2).mean(dim=-1)
print(f"Per-node MSE: min={mse.min():.4f}, max={mse.max():.4f}, mean={mse.mean():.4f}")
```

---

## Summary Table: Data Shapes at Each Stage

| Stage | Shape | Meaning |
|---|---|---|
| ROOT input | variable | hits(channel, integral) per event |
| Materialized | (N_w, C, 4, 6) | windows × channels × bins × features (C=11,276) |
| Flattened history | (C, 96) | per-window features flattened (4 windows × 24) |
| With channel idx | (C, 97) | + 1 normalized channel position |
| After pruning | (M, 97) | M active channels (M ≤ C) |
| PyG batch | x: (N, 97), y: (N, 24) | batched samples |
| After GCN per frame | (N, D_gnn) | spatial encoding (D_gnn=128) |
| After frame stacking | (T, N, D_gnn) | temporal sequence (T=4) |
| After GRU | (T, N, D_gru) | temporal context (D_gru=256) |
| Prediction | (N, D_target) | predicted next window (D_target=24) |
| MSE score | (N,) | per-node anomaly score |
| Window avg | (batch,) | per-window mean score |

---

## Visualization Quick Links

- **Full workflow**: [GNN_WORKFLOW.md](GNN_WORKFLOW.md)
- **Architecture details**: [GNN_ARCHITECTURE.md](GNN_ARCHITECTURE.md)
- **Model code**: [sbn_anomaly/models/gnn_forecaster_pyg.py](sbn_anomaly/models/gnn_forecaster_pyg.py)
- **Dataset code**: [sbn_anomaly/data/graph_window_dataset_pyg.py](sbn_anomaly/data/graph_window_dataset_pyg.py)
- **Trainer code**: [sbn_anomaly/train/gnn_trainer.py](sbn_anomaly/train/gnn_trainer.py)
- **Inference code**: [sbn_anomaly/infer/inferrer.py](sbn_anomaly/infer/inferrer.py)

