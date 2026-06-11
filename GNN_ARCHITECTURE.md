# GNN Architecture Visualization

## Variable Definitions

To avoid confusion with specific numbers, we use these variables throughout:

| Variable | Value | Definition |
|----------|-------|------------|
| `T` | 4 | History length (number of past windows) |
| `B` | 4 | Temporal bins per window |
| `F` | 6 | Node features per bin (sum, min, max, mean, stdev, count) |
| `F_window` | T × B × F = 96 | Total flattened features per channel per sample |
| `C` | Data-dependent (11,276 for your data) | Total number of channels; discovered during materialization from max channel ID in ROOT files |
| `D_gnn` | 128 | GCN hidden dimension |
| `D_gru` | 256 | GRU hidden dimension |
| `D_target` | B × F = 24 | Output dimension (features per window to predict) |
| `N` | Variable (≤ C) | Number of nodes in a batched sample; depends on pruning of inactive channels |

---

## Model Architecture Diagram

```
Input Batch from DataLoader (PyG)
├── x: (N, 1+F_window)            [channel_idx + temporal_features]
├── y: (N, D_target)              [target = next window]
└── edge_index: (2, E)            [sparse adjacency]

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  FORWARD PASS                                                       │
│                                                                     │
│  1. CHANNEL INDEX EXTRACTION                                        │
│     channel_idx = x[:, :1]     → (N, 1)                            │
│     temporal    = x[:, 1:]     → (N, F_window=96)                  │
│                                                                     │
│  2. TEMPORAL DECOMPRESSION                                          │
│     Reshape: (N, F_window) → (N, T=4, B×F=24)                      │
│     ↓ Loop over T timesteps                                        │
│                                                                     │
│     For t=0,1,2,...,T-1:                                            │
│     ├─ Extract frame: x_t = temporal[:, t, :]  (N, B×F)            │
│     ├─ Prepend channel_idx: [channel_idx, x_t]  (N, 1+B×F)         │
│     │                                                               │
│     └─ SPATIAL ENCODING (GCN layers)                               │
│        GCN Layer 1:                                                │
│        ├─ Input norm (BatchNorm or LayerNorm on 1+B×F dims)       │
│        ├─ GCNConv: (N, 1+B×F) → (N, D_gnn)                        │
│        ├─ ReLU activation                                          │
│        └─ Dropout(p=0.1)                                           │
│           ↓                                                        │
│        GCN Layer 2:                                                │
│        ├─ GCNConv: (N, D_gnn) → (N, D_gnn)                        │
│        ├─ ReLU + Dropout                                           │
│           ↓                                                        │
│        GCN Layer 3:                                                │
│        ├─ GCNConv: (N, D_gnn) → (N, D_gnn)                        │
│        ├─ ReLU + Dropout                                           │
│           ↓                                                        │
│        h_t: (N, D_gnn)  [encoded frame]                            │
│                                                                     │
│  3. TEMPORAL SEQUENCE STACKING                                      │
│     enc_seq = stack([h_0, h_1, ..., h_{T-1}])                      │
│     enc_seq: (T, N, D_gnn)                                         │
│                                                                     │
│  4. TEMPORAL MODELING (GRU)                                         │
│     GRU(input_size=D_gnn, hidden_size=D_gru, num_layers=1)        │
│     out_seq, hidden = GRU(enc_seq)                                 │
│     out_seq: (T, N, D_gru)                                         │
│     ↓                                                              │
│     Take final timestep: out_seq[-1]  (N, D_gru)                   │
│                                                                     │
│  5. DECODER (Linear Projection)                                     │
│     pred = Linear(D_gru, D_target)                                 │
│     pred: (N, D_target)  ← PREDICTION FOR NEXT WINDOW             │
│                                                                     │
│  6. COMPUTE ANOMALY SCORES                                          │
│     per_node_mse = mean((pred - y)^2, dim=-1)                      │
│     per_node_mse: (N,)                                             │
│     ↓                                                              │
│     Aggregate to window (accounting for graph membership):          │
│     ├─ window_mean = mean(per_node_mse)                            │
│     └─ window_max  = max(per_node_mse)                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Output:
├── pred: (N, D_target)                           [raw model output]
├── node_scores: (num_windows, C)                 [per-channel MSE]
├── window_mean: (num_windows,)                   [aggregated per-window mean]
└── window_max: (num_windows,)                    [aggregated per-window max]
```

## Sparse Graph Connectivity (Spatial)

```
Channel Space (C channels arranged in 1D)

Node indices: 0  1  2  3  4  5  6  7  8  9 ... C-1

radius = 4 adjacency:

Node 2 connects to neighbors within radius 4:
    [A]--[A]--[X]--[A]--[A]
     -2   -1    0   +1   +2   +3   +4
     ↓    ↓    ↓    ↓    ↓    ↓    ↓
     0    1   [2]   3    4    5    6

This creates a band-diagonal adjacency matrix:

    0  1  2  3  4  5  6
  0[X][X][X][X][X][ ][ ]
  1[X][X][X][X][X][X][ ]
  2[X][X][X][X][X][X][X]
  3[ ][X][X][X][X][X][X]
  4[ ][ ][X][X][X][X][X]
  5[ ][ ][ ][X][X][X][X]
  6[ ][ ][ ][ ][X][X][X]

Edges ≈ 2 × (C - 1) × radius ≈ 2 × radius × C for large C
(sparse: only ~(2×radius)/C percentage of theoretical maximum C² edges)

For your data with C=11,276 and radius=4:
  Edges ≈ 2 × 11,276 × 4 ≈ 90,208
  (vs theoretical max of 11,276² ≈ 127M, so only 0.07% dense)
```

## Data Shape Evolution

```
MATERIALIZATION (input preparation):
┌────────────────────────────────────────────────────┐
│ ROOT hit data from calorimeter                      │
│ Format: hits per channel per event                  │
└────────────────────────────────────────────────────┘
                    ↓
              Rolling windows
              (window_size events)
                    ↓
┌────────────────────────────────────────────────────┐
│ windows.npz: (N_windows, C, B, F)                  │
│ - N_windows: varies (depends on total events)      │
│ - C: total channels discovered from data           │
│       (11,276 for your detector)                   │
│ - B: temporal bins per window (4)                   │
│ - F: features per bin (6: sum, min, max, mean,     │
│       stdev, count)                                 │
└────────────────────────────────────────────────────┘

DATASET PREPARATION (GraphWindowDatasetPyG):
┌────────────────────────────────────────────────────┐
│ For each training sample:                           │
│ - Take T consecutive windows (T=4)                  │
│ - Flatten temporal: (C, T×B, F) → (C, T×B×F)       │
│ - Add channel index: (C, 1+T×B×F)                   │
│ - Prune if needed: (M, 1+T×B×F) where M ≤ C       │
│ - Next window target: (C, B×F) or (M, B×F)        │
└────────────────────────────────────────────────────┘
        ↓
    DataLoader batches
        ↓
┌────────────────────────────────────────────────────┐
│ PyG Batch object:                                   │
│ - x: (N, 1+F_window) where F_window=T×B×F         │
│ - y: (N, D_target) where D_target=B×F             │
│ - edge_index: (2, E)                              │
│ - batch: (N,) [graph assignment]                   │
└────────────────────────────────────────────────────┘

MODEL FORWARD:
┌────────────────────────────────────────────────────┐
│ Decompress temporal: (1+F_window,) → (1+T×B×F,)   │
│   - channel_idx: (1,)                              │
│   - temporal (F_window,) → reshape to (T, B×F)    │
│                                                    │
│ Per-frame encoding:                                 │
│   Frame t: concat([channel_idx, temporal[t]])      │
│   → (1+B×F,) → GCN layers → (D_gnn,)              │
│                                                    │
│ After T frames: (T, D_gnn)                         │
│ After GRU: (T, D_gru) → take last → (D_gru,)      │
│ After Decoder: (D_target,) ← prediction            │
└────────────────────────────────────────────────────┘

BATCHED SHAPES:
┌────────────────────────────────────────────────────┐
│ Input batch: (N, 1+F_window)                       │
│ After frame encoding: (T, N, D_gnn)                │
│ After GRU: (T, N, D_gru)                           │
│ Final pred: (N, D_target)                          │
│ Target: (N, D_target)                              │
│ MSE scores: (N,)                                   │
│ Reduced to per-window: (batch_size,) [mean/max]    │
└────────────────────────────────────────────────────┘
```

## Parameter Count & Computational Complexity

```
MODEL PARAMETERS:

GCN Layers (num_gnn_layers):
├─ Layer 1: (1+B×F) → D_gnn                   ≈ (1+B×F)×D_gnn params
├─ Layer 2: D_gnn → D_gnn                     ≈ D_gnn×D_gnn params
└─ Layer L: D_gnn → D_gnn                     ≈ D_gnn×D_gnn params
   Total GCN: ~(num_gnn_layers × D_gnn²) params

GRU:
├─ input_size: D_gnn
├─ hidden_size: D_gru
└─ Parameters: 3 × (D_gru × (D_gnn + D_gru + 1))  ≈ 3×D_gru×(D_gnn+D_gru) params

Decoder:
└─ (D_gru) → (D_target): D_target × D_gru      ≈ D_target×D_gru params

Total: ~(num_gnn_layers×D_gnn² + 3×D_gru×(D_gnn+D_gru) + D_target×D_gru) parameters

With T=4, B=4, F=6, D_gnn=128, D_gru=256:
  D_target = B×F = 24
  F_window = T×B×F = 96
  Total ≈ 340K parameters (lightweight for spatio-temporal modeling)

COMPUTATIONAL COMPLEXITY (per batch):

Forward pass:
├─ GCN: O(|E| × D_gnn) ≈ O(2×radius×N × D_gnn) per frame × T frames
├─ GRU: O(N × D_gru × (D_gnn + D_gru))
└─ Decoder: O(N × D_gru × D_target)

Speed: varies by batch node count and hardware, but typically 50-100 batches/sec on GPU
```

## Training vs Inference

```
TRAINING Loop:
┌─ Load batch (shuffled, aug possible)
├─ Forward pass: pred = model(data)
├─ Loss: MSELoss(pred, data.y)
├─ Backward + Optimizer step
└─ Repeat until convergence
   → checkpoint saved when val loss improves

INFERENCE Loop:
┌─ Load batch (non-shuffled, in order)
├─ no_grad(): Forward pass
├─ Compute per-node MSE vs ground truth
├─ Pool scores to per-window level
└─ Return (window_mean, window_max)
   → sent to alert decision logic
```

## Anomaly Score Semantics

```
MSE-based anomaly scoring:

Low Error (MSE < 0.25):
├─ Prediction matches reality
├─ Normal behavior pattern recognized
└─ → LOW anomaly score → NO alert

High Error (MSE > 0.25):
├─ Prediction fails to match
├─ Unusual/exceptional activity pattern
└─ → HIGH anomaly score → ALERT

Special cases:
├─ Inactive channel: NaN score (not a hit, don't penalize)
├─ Single outlier: captured in window_max (worst channel)
└─ Distributed anomaly: captured in window_mean (global pattern)

Multi-branch decision:
    alert = (tpc_mse > 0.25) OR (pmt_mse > 0.25) OR (window_mse > 0.25)
    → fires if ANY branch detects an anomaly
    → sensitive to diverse failure modes
```

