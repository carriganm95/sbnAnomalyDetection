# GNN Network Workflow: From Input Data to Anomaly Scores

## Overview

The GNN network is a forecasting-based anomaly detector that predicts the next time window in a sequence and flags anomalies when the prediction error is large. The workflow has three stages: **data materialization**, **model training**, and **inference/scoring**.

---

## Stage 1: Data Materialization (Input → Windows)

### Source Data
Input comes from ROOT files containing calorimeter hit data:
- **Hit branches**: `hits0.h.integral`, `hits0.h.channel` (and other hit planes)
- **Metadata branches**: `meta.run`, `meta.subrun`, `meta.evt`

### Materialization Process

```
ROOT events (streaming)
    ↓
aggregate hits per channel per event
    ↓
rolling window across events (window_size=20 events, stride=20)
    ↓
temporal binning (split each window into n_temporal_bins=4 bins)
    ↓
per-channel statistics per bin (sum, min, max, mean, stdev, count for each channel)
    ↓
(N_windows, N_channels, n_bins, n_features) numpy array
```

**Key parameters** (from `configs/gnn.yaml`):
- `window_size: 20` — number of events per window
- `n_temporal_bins: 4` — temporal resolution within each window
- `node_features: [sum, min, max, mean, stdev, count]` — 6 statistics per channel per bin

**Output shape**:
```
windows: (N_windows, C, B, F)
  N_windows ≈ total_events / window_size
  C = number of channels (discovered from data; 11,276 for your detector)
  B = 4 (temporal bins per window)
  F = 6 (node features per bin)
```

**File**: [sbn_anomaly/data/materialize_windows.py](sbn_anomaly/data/materialize_windows.py)

---

## Stage 2: Data Preparation → Graph Structure

### Dataset Construction

The [GraphWindowDatasetPyG](sbn_anomaly/data/graph_window_dataset_pyg.py) converts materialized windows into PyG Data objects:

#### Data Flattening (Compression)

Each window sample is prepared as:
```
Window: (N_channels, 4 temporal bins, 6 features) → (N_channels, 24)
History: T=4 past windows → (N_channels, 4×24 = 96) temporal features

Per-node feature vector:
  x = [channel_idx (normalized), temporal_features (flattened)]
  x: (N_channels, 1 + 96) = (N_channels, 97)

Target:
  y = next window (N_channels, 24) — the frame to predict
```

**Shape summary during preparation**:
| Stage | Shape | Description |
|-------|-------|-------------|
| Original window | (N, 4, 6) | N channels, 4 bins, 6 features |
| Temporal flattening | (N, 24) | 4 bins × 6 features |
| History (T=4 windows) | (N, 96) | 4 windows × 24 features |
| With channel index | (N, 97) | 1 + 96 |
| Target (next window) | (N, 24) | Same shape as one window |

#### Node Pruning (Optional Compression)

To reduce memory, **inactive channels are removed** per sample:
- Inactive = total absolute activity across all timesteps < 1e-6
- Only active nodes are kept in the graph
- A remap array tracks original → pruned channel indices
- Sparse edges are recomputed for the pruned subgraph

This reduces the effective graph size, especially for events with sparse hits.

#### Graph Construction

**Sparse adjacency** based on channel proximity:
```python
def build_sparse_edge_index(num_nodes: int, radius: int = 4) -> torch.LongTensor:
    # For each node i, connect to nodes [i-radius, ..., i+radius]
    # Respects boundaries (no self-loops unless part of radius)
```

**From config**: `adjacency_radius: 4` → each channel connects to ±4 neighbors

**Edge index shape**: (2, num_edges)
- For C channels with radius=4: ~2×radius×C edges (sparse!)
  - Example: 11,276 channels with radius=4 → ~90,208 edges
  - Theoretical max: 11,276² ≈ 127M edges → only 0.07% dense!
- After pruning to M active nodes: edges scale with M

#### PyG Data Object

```python
Data(
    x=(M, 97),           # M = number of active nodes
    y=(M, 24),           # target = next window
    edge_index=(2, E),   # sparse edges
    batch=None,          # filled by DataLoader
)
```

**File**: [sbn_anomaly/data/graph_window_dataset_pyg.py](sbn_anomaly/data/graph_window_dataset_pyg.py)

---

## Stage 3: Model Architecture & Forward Pass

### Model: GNNForecasterPyG

[sbn_anomaly/models/gnn_forecaster_pyg.py](sbn_anomaly/models/gnn_forecaster_pyg.py)

**Architecture**:
```
Input: x (total_nodes_in_batch, 1+T*F)  where T=history, F=features_per_bin
    ↓
Extract & Normalize
  channel_idx: (total_nodes, 1)
  temporal:    (total_nodes, T*F) → reshape to (total_nodes, T, F)
    ↓
PER-FRAME SPATIAL ENCODING (GCN layers)
  For each time step t in 1..T:
    x_t = [channel_idx, temporal[:, t, :]]  (total_nodes, 1+F)
    ↓
    stacked GCN layers (default: 3 layers)
      h_t = GCN(x_t, edge_index)  → ReLU → Dropout
      h_t: (total_nodes, gnn_hidden=128)
    ↓
  enc_seq: stack all T encoded frames → (T, total_nodes, 128)
    ↓
TEMPORAL MODELING (GRU)
  out_seq = GRU(enc_seq)  → (T, total_nodes, gru_hidden=256)
    ↓
DECODER (Linear layer)
  pred = Linear(gru_hidden, target_dim)
  pred: (total_nodes, 24)  ← prediction for next window
```

### GCN Layer Details

Standard `GCNConv` performs:
```
h' = σ(D^(-1/2) * A_hat * D^(-1/2) * X * W)

where:
  A_hat = adjacency + self-loops
  X = node features
  W = learnable weight matrix
  σ = ReLU activation
```

**Stochastic training**:
- Applies dropout between GCN layers: `dropout=0.1`
- GRU dropout applied between GRU layers if `gru_layers > 1`

### Configuration

From [configs/gnn.yaml](configs/gnn.yaml):
```yaml
model:
  history: 4                  # T=4 past windows
  gnn_hidden: 128             # hidden dimension of GCN
  gnn_layers: 3               # number of stacked GCN layers
  gru_hidden: 256             # hidden dimension of GRU
  gru_layers: 1               # number of GRU layers
  dropout: 0.1                # dropout probability
```

---

## Decompression During Forward Pass

As data flows through the model:

```
Input flattening (compressed):
  temporal: (total_nodes, 96)  ← 4 windows × 24 features

Decompression in forward():
  Reshape: (total_nodes, 96) → (total_nodes, T=4, F=24)
           ↓
  Per-frame extraction:
    for t in 0..3:
      x_t = temporal[:, t, :]  → (total_nodes, 24)
      This extracts one temporal bin from the flattened sequence
    ↓
  Prepend channel index for each frame:
    x_t = [channel_idx, x_t]  → (total_nodes, 1+24)
```

This decompression happens **efficiently on GPU** during inference—no materialization step needed.

---

## Stage 4: Loss & Training

**Training objective**: Next-frame prediction via reconstruction

```python
class GNNTrainerPyG(BaseTrainer):
    def compute_loss(self, batch):
        data = batch.to(device)
        pred = model(data)  # shape: (total_nodes, 24)
        y = data.y.float()  # shape: (total_nodes, 24)
        loss = MSELoss(pred, y)
        return loss
```

**Per-node loss**: MSE between predicted and actual next-window features

Training minimizes this loss so the model learns to **forecast normal behavior**.

**File**: [sbn_anomaly/train/gnn_trainer.py](sbn_anomaly/train/gnn_trainer.py)

---

## Stage 5: Inference & Anomaly Score Calculation

### Inference Workflow

```
Test windows (no labels)
    ↓
Build Data objects (same as training)
    ↓
Forward pass through trained model
    ↓
pred: (total_nodes, 24) — predicted next window per node
y:    (total_nodes, 24) — actual next window per node
    ↓
Compute per-node MSE:
  per_node_mse = (pred - y)^2 → mean over feature dimension
  per_node_mse: (total_nodes,) ← error at each channel
    ↓
Aggregate to window level:
  window_mean = mean(per_node_mse) over active channels
  window_max  = max(per_node_mse) over active channels
    ↓
Both scores indicate anomaly severity (higher = more anomalous)
```

### GNNScorer Implementation

[sbn_anomaly/infer/inferrer.py](sbn_anomaly/infer/inferrer.py) - `GNNScorer` class:

```python
@torch.no_grad()
def score_loader(self, loader):
    """Score all windows, returning per-channel and per-window anomaly scores."""
    
    for batch in loader:
        pred = model(data)  # forward pass
        per_node_mse = ((pred - data.y) ** 2).mean(dim=-1)
        
        # Map back to full channel grid
        # (pruned channels → full detector channels)
        for graph_idx in data.batch:
            g_mse = per_node_mse[graph_idx]
            g_channels = data.active_mask[graph_idx]  # original indices
            
            # Scatter into full-size array
            full_scores = np.full(num_channels, np.nan)
            full_scores[g_channels] = g_mse.numpy()
            node_scores.append(full_scores)
        
        window_scores_mean.append(g_mse.mean())
        window_scores_max.append(g_mse.max())
    
    return node_scores, window_scores_mean, window_scores_max
```

### Output

**Three score arrays per inference pass**:

1. **node_scores**: (N_windows, N_channels)
   - Per-channel MSE for each window
   - NaN for inactive channels (distinguishes "not hit" from "normal")

2. **window_scores_mean**: (N_windows,)
   - Mean prediction error across all active channels
   - Single scalar per window

3. **window_scores_max**: (N_windows,)
   - Maximum prediction error in the window
   - Sensitive to individual outlier channels

### Anomaly Decision

```python
threshold = 0.25  # tunable

if window_scores_mean > threshold:
    flag_as_anomaly()  # OR window_scores_max > threshold
```

Higher scores = **worse prediction → more anomalous**

---

## Multi-Branch Integration

The GNN is one of three independent branches:
- **TPC branch**: hits-based anomaly (GNN or dense model)
- **PMT branch**: photon-detector anomaly
- **Window branch**: waveform-based anomaly

[sbn_anomaly/infer/multi_branch.py](sbn_anomaly/infer/multi_branch.py) combines them:

```python
class MultiBranchScorer:
    def score(self, tpc_features, pmt_features, window_features):
        tpc_scores = tpc_scorer.score(tpc_features)   # (N,)
        pmt_scores = pmt_scorer.score(pmt_features)   # (N,)
        window_scores = window_scorer.score(window_features)  # (N,)
        
        max_scores = np.maximum(tpc_scores, np.maximum(pmt_scores, window_scores))
        
        # Alert if ANY branch exceeds threshold
        alert_flags = (
            (tpc_scores > threshold) |
            (pmt_scores > threshold) |
            (window_scores > threshold)
        )
        
        return {
            'tpc_scores': tpc_scores,
            'pmt_scores': pmt_scores,
            'window_scores': window_scores,
            'max_scores': max_scores,
            'alert_flags': alert_flags,
        }
```

---

## Summary: Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. MATERIALIZATION                                          │
│ ROOT hits → rolling windows → temporal bins → features      │
│ Output: (N_windows, N_channels, 4, 6)                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. DATA PREPARATION (GraphWindowDatasetPyG)                 │
│ Flatten temporal dims (1+96) → Prune inactive              │
│ Build sparse graphs (radius=4) → PyG Data objects          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. MODEL FORWARD PASS (GNNForecasterPyG)                   │
│ Per-frame GCN encoding → GRU temporal modeling             │
│ → Decoder → prediction (total_nodes, 24)                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. ANOMALY SCORING (GNNScorer)                              │
│ MSE(prediction, target) → per-node, per-window scores      │
│ Map back to channel grid → (N_windows, N_channels)          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. ALERT DECISION (MultiBranchScorer)                       │
│ Combine 3 branches → max score → flag if threshold exceeded │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Insights

1. **Forecasting + Reconstruction Error**: The GNN learns normal behavior through prediction. Anomalies are detected when predictions fail (high error).

2. **Sparse Graph**: Only connects adjacent channels (radius=4), reducing edges from O(N²) to O(N), making the model scalable.

3. **Temporal Compression**: Raw features (4 bins × 6 stats) are flattened into (96,) vectors for efficient batching and GPU processing.

4. **Per-Node + Per-Window Scores**: Returns both per-channel (for localization) and aggregated (for decision-making) anomaly scores.

5. **GCN for Spatial**: Graph Convolutional Networks aggregate information from neighboring channels, capturing local correlations.

6. **GRU for Temporal**: Recurrent layer learns temporal dynamics, predicting how the next window should evolve from the past 4 windows.

7. **Multi-Branch Fusion**: Combines independent TPC, PMT, and Window models via OR logic to maximize detection sensitivity while maintaining specificity.

