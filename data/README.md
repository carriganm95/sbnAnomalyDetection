# Data

This directory holds file lists, cached event files, and notes on available data for training and testing.

## Available data

**Storage location:** `/pnfs/icarus/persistent/users/micarrig/DQM/`

| Type | Subdirectory |
|------|-------------|
| Reconstructed (reco) | `<run_dir>/reco` |
| Raw decode | `<run_dir>/raw_decode` |

**Run directories:**

| Status | Run dirs |
|--------|----------|
| Good | 19305, 19308, 19315, 829 |
| Known problems | 20614, 20615, 20620, 20621, 20173, 830 |

## File lists

Text files listing ROOT paths for training and testing are kept here. Format is one path per line; lines beginning with `#` are treated as comments.

```
# Example: data/train_files.txt
/pnfs/icarus/persistent/users/micarrig/DQM/19305/reco/run19305_evt0.root
/pnfs/icarus/persistent/users/micarrig/DQM/19305/reco/run19305_evt1.root
...
```

Pass these to the training CLI:

```bash
sbn-train --config configs/gnn.yaml --root-file-list data/train_files.txt
```

## Cached events files

After a training run with `training.save_events_path` configured, a compact `.npz` events file is written here. This stores pre-aggregated sparse hit data (see [sparse event format](#sparse-event-format)) and loads much faster than re-streaming ROOT files.

```bash
# Use a cached events file for subsequent runs (set in configs/gnn.yaml):
#   data:
#     events_path: data/events_cache.npz
sbn-train --config configs/gnn.yaml
```

## Sparse event format

The events NPZ contains four arrays in CSR format:

| Key | Dtype | Description |
|-----|-------|-------------|
| `channels_flat` | int64 | Concatenated channel (wire) indices across all events |
| `integrals_flat` | float32 | Corresponding hit integrals |
| `offsets` | int64 | Row pointers — event `i` has hits `channels_flat[offsets[i]:offsets[i+1]]` |
| `n_channels` | int64 | Total number of channels in the detector |

This format is compact because only channels with hits are stored. For a typical run with ~380 hits/event across 11,276 channels, this is roughly 1,000× smaller than a dense (events × channels) array.

### Reading the events file manually

```python
import numpy as np

data = np.load("data/events_cache.npz")
n_events = len(data["offsets"]) - 1
print(f"{n_events} events, {data['n_channels']} channels")

# Hits for event 42:
i = 42
ch = data["channels_flat"][data["offsets"][i] : data["offsets"][i+1]]
val = data["integrals_flat"][data["offsets"][i] : data["offsets"][i+1]]
```

### Loading as a training dataset

```python
from sbn_anomaly.data.sparse_window_dataset import SparseWindowDatasetPyG

dataset = SparseWindowDatasetPyG.from_npz(
    "data/events_cache.npz",
    history=4,        # past frames used as input
    window_size=20,   # events per frame
    n_bins=4,         # temporal bins within each frame
    stride=5,         # step between consecutive windows
    radius=4,         # channel adjacency radius for graph edges
    node_features=["sum", "min", "max", "mean", "stdev", "count"],
)
print(f"{len(dataset)} training windows")
```

## Window structure

Each sample produced by `SparseWindowDatasetPyG` covers `(history + 1) * window_size` consecutive events:

```
events: [──── history frames ────][─ target frame ─]
             history * window_size     window_size

Each frame:
  window_size events  →  split into n_bins temporal bins
  each bin:  hits aggregated per channel  →  node features [sum, min, max, ...]
```

The model sees the `history` past frames as input and predicts the `target` frame. Anomaly score is the per-channel MSE between the prediction and the actual target frame.
