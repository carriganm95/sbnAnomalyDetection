# Scripts

Standalone tools that sit outside the `sbn_anomaly` package and `sbn-train`/
`sbn-infer` CLIs — data-prep bridges, one-off diagnostics, and training
shortcuts. See the top-level [README](../README.md) for the model/pipeline
itself.

## compare_events_distributions.py

Compares variable distributions between two sparse **events** `.npz` files
(the format `SparseWindowDatasetPyG.save_events()` / `materialize_windows.py`
writes — see [`data/README.md`](../data/README.md#sparse-event-format)),
e.g. a good-run file vs. a bad-run file, and writes the result as ROOT
histograms rather than per-channel PDFs (a full detector has thousands of
channels — one ROOT file with directories is the only way to keep that
browsable).

This is a **diagnostic that doesn't touch the model at all** — it only needs
`numpy`/`pandas`/`uproot`, no `torch`/`torch_geometric`. Run it before adding
or trusting a `node_features` entry: if a variable/statistic shows no
separation between good and bad here, adding it to `configs/graph_vae.yaml`
is unlikely to help either, since the GNN only ever sees an aggregate of the
same underlying per-hit numbers.

### Two comparison modes

Both can be produced in the same run; each is independently toggleable.

| Mode | What it histograms | Answers |
|------|---------------------|---------|
| **Raw per-hit** (default, `--no-raw-hit` to disable) | Every individual hit's value | "Does this quantity differ hit-by-hit?" |
| **Windowed** (`--windowed`) | One summary statistic (mean/median/stdev/...) per `(window, channel)` group, using the same event-windowing the model trains on | "Do the actual per-window numbers the model sees as node features differ?" — also supports statistics (e.g. `median`) the model doesn't currently compute |

### Input

Two events `.npz` files, each built via:
```bash
python -m sbn_anomaly.data.materialize_windows --config configs/graph_vae.yaml \
    --root-file-list data/good_train.txt --output data/events_good.npz
```
Only `channels_flat`/`offsets`/`n_channels` are required; everything else
(`integrals_flat`, `times_flat`, `widths_flat`, `sumadcs_flat`, `mults_flat`,
`hassps_flat`, `planes_flat`) is used opportunistically — a variable missing
from one file is skipped there with a warning instead of failing the run, so
you can compare an older and a newer materialization even if they don't carry
the same set of optional branches.

### Output layout

```
all_detector/<variable>_<label>
per_plane/plane<p>/<variable>_<label>
per_channel/ch<channel>/<variable>_<label>

windowed/<variable>/<stat>/all_detector_<label>
windowed/<variable>/<stat>/per_plane/plane<p>/<label>
windowed/<variable>/<stat>/per_channel/ch<channel>/<label>   # opt-in, see --windowed-per-channel
```
Both files' histograms for the same variable/plane/channel share one binning
(see `--range-mode`), so a channel's `good` and `bad` histograms drawn on the
same axes are directly comparable.

Browse with ROOT:
```bash
root -l compare.root
# in ROOT: TBrowser b
```
or with uproot:
```python
import uproot
f = uproot.open("compare.root")
f["per_channel/ch04200/width_good"].to_hist()
f["windowed/width/median/per_channel/ch04200/good"].to_hist()
```

### Option reference

| Option | Default | Description |
|---|---|---|
| `--file-a` | *(required)* | First events `.npz` (e.g. good runs) |
| `--file-b` | *(required)* | Second events `.npz` (e.g. bad runs) |
| `--label-a` | `good` | Name used in output keys/logging for file A |
| `--label-b` | `bad` | Name used in output keys/logging for file B |
| `--output` | *(required)* | Output ROOT file path |
| `--variables` | all present in either file | Subset of `integral`, `time`, `width`, `sumadc`, `mult`, `hasSP` |
| `--bins` | `60` | Histogram bin count (applies to raw and windowed) |
| `--range-mode` | `percentile` | `percentile` (clip outliers) or `minmax` (full range) for the shared histogram range per variable |
| `--range-percentiles` | `0.1 99.9` | Low/high percentile bounds when `--range-mode percentile` |
| `--channel-map` | *(none)* | CSV with `offlchan`/`plane` columns (e.g. `configs/SBNDTPCChannelMap_v2_with_positions.csv`), used to derive plane from channel id when a file lacks `planes_flat` |
| `--channel-range LO HI` | full detector | Restrict per-channel histograms (raw **and** windowed) to channels in `[LO, HI)` — e.g. `4000 5500` for the collection-plane region |
| `--no-raw-hit` | off | Skip the raw per-hit section entirely (useful with `--windowed` if you only want windowed comparisons) |
| `--no-per-channel` | off | Skip the (large, slow) raw per-channel section |
| `--no-per-plane` | off | Skip the raw per-plane section |
| `--no-all-detector` | off | Skip the raw all-detector section |
| `--progress-every` | `1000` | Log a progress line every N channels/windows processed |
| `--windowed` | off | Enable the windowed comparison mode |
| `--window-size` | `100` | Events per window — match `data.window_size` in your training config |
| `--window-stride` | `--window-size` (non-overlapping) | Window stride in events |
| `--window-stats` | `mean median stdev` | One or more of `mean`, `median`, `stdev`, `sum`, `min`, `max`, `count` |
| `--window-variables` | same as `--variables` | Subset of variables to window-aggregate |
| `--windowed-per-channel` | off | Also write the per-channel windowed breakdown (large — combine with `--channel-range`); `all_detector`/`per_plane` windowed comparisons are always written when `--windowed` is set, regardless of this flag |
| `--max-windows` | unlimited | Cap the number of windows processed per file — fast first look on a huge input |

### Examples

Full detector, raw per-hit only, every variable present in either file:
```bash
python scripts/compare_events_distributions.py \
    --file-a data/events_good.npz --label-a good \
    --file-b data/events_bad.npz  --label-b bad \
    --output compare.root
```

Add windowed mean/median/stdev (`window_size=100` to match `configs/graph_vae.yaml`)
at the all-detector and per-plane level — cheap, sensible first pass:
```bash
python scripts/compare_events_distributions.py \
    --file-a data/events_good.npz --file-b data/events_bad.npz \
    --windowed --window-size 100 --output compare.root
```

Add the heavier per-channel windowed breakdown, restricted to the
collection-plane region flagged in the DQM plot:
```bash
python scripts/compare_events_distributions.py \
    --file-a data/events_good.npz --file-b data/events_bad.npz \
    --windowed --window-size 100 --windowed-per-channel \
    --channel-range 4000 5500 --output compare_collection.root
```

Skip the raw per-hit section and look only at windowed values:
```bash
python scripts/compare_events_distributions.py \
    --file-a data/events_good.npz --file-b data/events_bad.npz \
    --no-raw-hit --windowed --output compare_windowed_only.root
```

Fast sanity check on a huge input before committing to a full run:
```bash
python scripts/compare_events_distributions.py \
    --file-a data/events_good.npz --file-b data/events_bad.npz \
    --windowed --max-windows 20 --no-per-channel --output compare_quicklook.root
```

### Cost / sizing notes

- `all_detector` and `per_plane` sections are cheap regardless of mode —
  always safe to leave on.
- Raw `per_channel` scales with `variables × channels × 2 files`; for the
  full ~11k-channel SBND detector across all six variables this is on the
  order of `10^5` histograms. Use `--channel-range` to scope it down, or
  `--no-per-channel` for a first look.
- Windowed `per_channel` (`--windowed-per-channel`) additionally scales with
  `stats`, so it's opt-in for a reason — pair it with `--channel-range`.
- `--max-windows` bounds windowed-mode runtime independent of the above, for
  a quick look before running the full window count.

## check_standardization_per_plane.py

Checks whether `graph_vae`'s feature standardization — the `(x - mean) / std`
applied per node feature when `data.standardize: true` — is masking a
per-plane bias. `SparseWindowDatasetPyG._fit_standardization()` fits **one**
mean/std per feature column by pooling every active channel from every plane
together; if a feature's true scale differs by plane (collection is
unipolar, induction is bipolar — plausible), every "normal" window on the
underrepresented plane gets z-scored to a nonzero offset purely from being
that plane, not from being anomalous.

This script re-fits the same statistic split by plane (reusing
`SparseWindowDatasetPyG`'s own `_compute_frame`/`_fit_standardization`, same
RNG seed and frame sampling, cross-checked to match exactly) and reports each
plane's offset from the pooled mean in units of the pooled std, per feature
column. Unlike `compare_events_distributions.py`, this **does** require
torch/torch_geometric (it's calling the model's own dataset class directly),
so run it in the real training environment.

```bash
python scripts/check_standardization_per_plane.py \
    --events data/good_events_train.npz --config configs/graph_vae.yaml
```
Reads `data.window_size`/`n_temporal_bins`/`stride`/`node_features`/
`channel_map` from the config (same values `sbn-train` would use), with
`--window-size`/`--n-bins`/`--stride`/`--node-features`/`--channel-map` as
overrides. Plane comes from `planes_flat` in the events npz if present,
otherwise `--channel-map`. `--output-csv` dumps the full per-column/per-plane
table (pooled and per-plane mean/std, offset, std ratio, sample count) for
further analysis; columns whose offset exceeds `--flag-threshold` (default
`0.3` pooled-sigma) are also flagged directly in the printed report.

`--max-frames`/`--seed` must stay at the defaults (`500`/`0`) to reproduce
training's exact fit; the script cross-checks its own replicated pooled
mean/std against `ds._fit_standardization()`'s real output and warns if they
don't match bit-for-bit.

**If this flags a real bias:** set `data.standardize_by: plane` in
`configs/graph_vae.yaml` (see the [top-level README](../README.md) config
table) and retrain. That fits a separate mean/std per plane instead of one
pooled fit — planes with too few samples (`data.min_plane_samples`, default
20) fall back to the pooled stats automatically. Re-run this script
afterwards (or just re-check good/bad separation) to confirm the offsets
have shrunk.

## check_reconstruction_by_plane.py

Checks whether the model is actually *fitting* one plane worse than another
on good data — a different question from standardization. `GraphVAETrainer`'s
loss is `((x_hat - y) ** 2).mean()`, an unweighted mean over every active
node/feature in a batch. If one plane has fewer active channels per window
than another (plausible — SBND collection is a minority of total channels
vs. the two induction planes), its share of the total gradient signal is
smaller purely from channel count, independent of anything wrong with its
data, and the model can end up under-fitting that plane.

This loads a trained checkpoint and reuses `GraphVAETrainer.collect_channel_mse`
(no reimplemented forward pass) to get per-channel average reconstruction MSE
over a good-run events npz, then groups it by plane:

```bash
python scripts/check_reconstruction_by_plane.py \
    --config configs/graph_vae.yaml \
    --checkpoint checkpoints/graph_vae/v7/graph_vae_final.pt \
    --events data/good_events_val.npz
```

Prefer a held-out validation events npz if you have one. It reloads the
`standardization.npz` saved next to the checkpoint (same one training used —
important given `standardize_by` may be `plane`) so the measured MSE reflects
what the model actually trained against, rather than refitting from scratch.
Plane comes from `planes_flat` in the events npz if present, otherwise
`--channel-map` (default: `data.channel_map` from `--config`). Also requires
torch/torch_geometric (it loads and runs the real model).

The report includes a worst/best plane MSE ratio with a rule-of-thumb verdict
(`> 1.5x` flagged as a likely real fit imbalance). If it comes back flat
across planes, the loss isn't the bottleneck and `check_standardization_per_plane.py`
combined with a good/bad separation re-check is a better next step than
touching the loss function. If it does show a persistent gap, that points at
a plane-balanced loss (mean of per-plane means instead of one global mean)
rather than more features or a bigger model.

## Other scripts

| Script | Purpose |
|---|---|
| `visualize_channel_graph.py` | Plots the electronics-aware channel graph (ASIC cliques, FEMB readout chains) for one or more channels — sanity-check the adjacency `sbn_anomaly.data.channel_graph` builds. |
| `plot_raw_waveforms.py` | Plots preprocessed waveforms from a `raw_waveforms_*.npz` (the raw-ADC VAE input), by row index, channel, event, or random sample. |
| `dump_rawdigits.C` | Gallery/ROOT macro: flattens `raw::RawDigit` ADC waveforms from art/LArSoft files into a flat ntuple uproot can read directly, with no art dictionaries needed downstream. |
| `dump_rawdigits_pyroot.py` | Same bridge as `dump_rawdigits.C` but via PyROOT instead of gallery. |
| `rawdigits_to_npz_gallery.py` | One-step art `RawDigit` files → preprocessed waveform `.npz`, via gallery — no intermediate flat ROOT file. Run on a LArSoft node. |
| `inspect_art_rawdigits.py`, `inspect_rawdigit_obj.py`, `probe_pyroot_access.py` | One-off probes for figuring out whether/how `raw::RawDigit` branches can be read from a given art file (uproot vs PyROOT vs gallery). Paste output back when debugging a new file format. |
| `run_materialize_windows.sh` | Batch-runs `materialize_windows` over multiple `CI_build_lar_ci_<N>` directories' `reco/DQM*.root` files. |
| `run_inference.sh` | Runs `sbn-infer` for a given config against a features file. |
| `train_tpc.sh` / `train_pmt.sh` / `train_fusion.sh` / `train_window.sh` | Thin wrappers around `sbn-train --config configs/<model>.yaml` for the legacy (non-graph_vae) model types. |
