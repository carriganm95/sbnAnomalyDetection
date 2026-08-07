# Data

File lists, cached event files, and notes on the runs used to train and test the
graph VAE. See the top-level [README](../README.md) for the full workflow.

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

Good runs are used for training (the VAE learns the nominal); good + bad runs are
used for evaluation (good-vs-bad separation and streaming detection latency).

## File lists

Text manifests of ROOT paths, one per line; lines beginning with `#` are
comments. Recognized extensions: `.txt`, `.lst`, `.list`, `.filelist`, `.csv`.

```
# data/good_train.txt
/pnfs/icarus/persistent/users/micarrig/DQM/19305/reco/run19305_evt0.root
/pnfs/icarus/persistent/users/micarrig/DQM/19305/reco/run19305_evt1.root
...
```

## Building events npz files

Materialize a compact sparse-events `.npz` from a file list (default output of
`materialize_windows` — no `--windows` flag). Use the graph-VAE config so the hit
branches, window definition, and channel count match training:

```bash
python -m sbn_anomaly.data.materialize_windows --config ../configs/graph_vae.yaml \
    --root-file-list data/good_train.txt --output data/good_events_train.npz
```

Then train/score point at these npz files (`data.events_path` or
`sbn-infer --input`). Merge several correctly (offsets must be shifted — a plain
`np.concatenate` corrupts the CSR structure):

```bash
python -m sbn_anomaly.data.merge_events --output data/all_good.npz --glob 'data/good_*.npz'
```

## Sparse event format

The events npz stores hits in CSR form (only channels with hits are kept, ~1000×
smaller than a dense events × channels array):

| Key | Dtype | Description |
|-----|-------|-------------|
| `channels_flat` | int64 | channel ids of all hits, concatenated across events |
| `integrals_flat` | float32 | corresponding hit integrals |
| `times_flat` | float32 | hit times (enables `time_mean` / `time_spread` features) |
| `widths_flat` | float32 | hit widths (enables `width_*` features; only meaningfully populated on the collection plane in SBND ntuples) |
| `sumadcs_flat` | float32 | raw ADC sum under the hit (enables `sumadc_*` features) |
| `mults_flat` | float32 | hit multiplicity — overlapping/split-fit hits (enables `mult_*` features) |
| `hassps_flat` | float32 | whether the hit has a matched 3-D space point, 0/1 (enables `sp_fraction`) |
| `planes_flat` / `wires_flat` / `tpcs_flat` | int32 | hit geometry — used for per-plane grouping and diagnostics, not as node features directly |
| `offsets` | int64 | event `i` has hits `channels_flat[offsets[i]:offsets[i+1]]` |
| `n_channels` | int64 | total detector channel count |
| `evt_run` / `evt_subrun` / `evt_num` | int32 | per-event provenance |

`widths_flat`/`sumadcs_flat`/`mults_flat`/`hassps_flat`/`planes_flat`/`wires_flat`/
`tpcs_flat` are all optional — `SparseWindowDatasetPyG.from_root()` pulls them when
present in the source tree (checked per file; zero-filled and logged if a
production is missing one) and `save_events()`/`from_npz()` round-trip whichever
are present. Requesting a `node_features` entry that needs one of these without
having materialized it raises a clear `ValueError` telling you which array is
missing.

See [`scripts/README.md`](../scripts/README.md#compare_events_distributionspy)
for a tool that compares these values (raw and windowed) between two events
npz files before you trust one as a `node_features` entry.

Per-channel window features are computed from this on the fly when the dataset is
loaded, so a single events npz supports any `node_features` / `window_size`
choice without re-materializing.
