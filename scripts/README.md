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

## check_plane_information_content.py

Checks a fifth hypothesis, distinct from the standardization/loss-weighting
ones above: collection is often the best-calibrated, lowest-noise plane in
SBND, but that can come with **fewer hits per channel per window** than
induction. Per-plane standardization corrects for a plane's absolute
mean/std, but not for *why* that std is what it is — if a channel only sees
a handful of hits per window, its window-level statistics (count, occupancy,
mean-of-hit-integral) are themselves noisy estimates from small sample size
(shot noise), and standardization just divides by that inflated noise floor.
The same absolute anomaly deviation then buys fewer standard deviations on
that plane, purely from statistics, not from any bug.

This tests both halves directly from a good-run events npz, no
model/checkpoint/torch required:

- **Premise** — does this plane actually see fewer hits/window? (median
  hits/window, occupancy, per plane)
- **Mechanism** — is a channel's window-to-window hit-count variance close to
  pure Poisson shot noise at its own rate, or is there real structure above
  that floor? For hits arriving at window-level rate `lambda`, Poisson
  statistics alone predict `CV_theory = 1/sqrt(lambda)`; the ratio
  `CV_observed / CV_theory` near `1.0` means the variance standardization
  divides by is mostly irreducible counting noise, not genuine signal.

```bash
python scripts/check_plane_information_content.py \
    --events data/good_events_val.npz \
    --window-size 100 --stride 100 \
    --channel-map configs/SBNDTPCChannelMap_v2_with_positions.csv
```
Or pass `--config configs/graph_vae.yaml` to pick up `window_size`/`stride`/
`channel_map` from its `data` section instead of specifying them by hand.
Plane comes from `planes_flat` in the events npz if present, otherwise
`--channel-map`. `--output-csv` dumps the full per-channel table
(`mean_count`, `occupancy`, `cv_count_observed`, `cv_count_theory`, `ratio`,
plus the same for window-summed integral as context).

**If a plane has both the fewest hits/window and a near-1.0 ratio:** that
supports the hypothesis. The fix isn't more features or architecture — try a
larger `data.window_size` (more hits to average per window, at the cost of
time resolution) and/or a per-plane anomaly threshold instead of comparing
that plane's raw score against a single global one, since a real degradation
there may only ever produce a modest absolute rise even once everything else
is working correctly.

## compare_model_features.py

The model-accurate companion to `compare_events_distributions.py`. That
script's windowed mode applies a generic stat menu (mean/median/stdev/...) to
a raw hit variable — fast and torch-free, but it can drift from what
`graph_vae` actually computes (it has no way to reproduce `occupancy`, a
hits-with-a-hit / events-in-bin ratio rather than a per-hit reduction, or
`sp_fraction`), and it can't show standardized values at all.

This script instead builds the real `SparseWindowDatasetPyG` from `--config`'s
`data` section and reads `data.y` straight out of `__getitem__` for every
window — the identical code path training/inference use, so there's no
second implementation of the feature math to fall out of sync. For every
`data.node_features` entry and temporal bin it recovers **both**:

- **raw** — physical-unit value, recovered by inverting the model's own
  `(x - mean) / std` with that same mean/std (not recomputed from scratch)
- **standardized** — `data.y` itself, literally what the loss function and
  reconstruction target look like

File A's fitted standardization (or an externally supplied checkpoint's) is
reused for File B too — never refit per file, matching real inference, where
good-run statistics are applied to whatever data comes in.

```bash
python scripts/compare_model_features.py \
    --config configs/graph_vae.yaml \
    --file-a data/good_events_val.npz --file-b data/bad_events_val.npz \
    --output model_features_compare.root
```

Reuse a checkpoint's fitted standardization instead of refitting from
`--file-a` (recommended when you have one — matches exactly what it trained
against):
```bash
python scripts/compare_model_features.py \
    --config configs/graph_vae.yaml \
    --file-a data/good_events_val.npz --file-b data/bad_events_val.npz \
    --standardization checkpoints/graph_vae/v7/standardization.npz \
    --output model_features_compare.root
```

### Output

Same `all_detector`/`per_plane`/`per_channel` (opt-in via `--per-channel`)
ROOT histogram layout as `compare_events_distributions.py`, split by feature,
temporal bin, and raw vs. standardized:
```
model_features/<feature>/bin<b>/raw/all_detector_<label>
model_features/<feature>/bin<b>/raw/per_plane/plane<p>/<label>
model_features/<feature>/bin<b>/standardized/all_detector_<label>
model_features/<feature>/bin<b>/standardized/per_plane/plane<p>/<label>
```
**Plus** overlay+ratio canvases (good/bad histograms overlaid via
`TRatioPlot`, ratio = bad/good in the pad below) at the `all_detector` and
`per_plane` level by default:
```
canvases/<raw|standardized>/<feature>/bin<b>/all_detector
canvases/<raw|standardized>/<feature>/bin<b>/per_plane/plane<p>
```
`--no-canvases` skips that section if PyROOT isn't available on a given node.

Each histogram gets Poisson (`sqrt(N)`) bin errors (drawn as error bars on
both the overlay and the propagated ratio), and each canvas is annotated with
a `#chi^{2}/ndf` (and p-value) from ROOT's own two-sample Poisson chi2 test
(`TH1::Chi2TestX`, option `"UU"` — appropriate since both histograms are raw
unweighted counts). `ndf` can come back `0` for a bin range where too few
bins have entries in both histograms (e.g. a small proof-of-concept sample);
the canvas falls back to a "too few populated bins" label rather than
dividing by zero. The chi2/ndf for every canvas is also logged to stdout as
it's written, so you can grep for the largest ones without opening ROOT.

### Requires

**torch/torch_geometric** (builds the real dataset) **and PyROOT** (draws/
writes the `TCanvas`/`TRatioPlot` objects — `uproot` alone can't create
those). Run this in a full LArSoft/analysis environment, not the torch-free
environment `compare_events_distributions.py` is designed for.

The extraction/histogram-writing logic (raw-value recovery via inverting the
standardization, per-plane grouping, channel indexing) is unit-tested with
synthetic duck-typed datasets covering both global and per-plane
standardization shapes. **The PyROOT canvas-drawing section could not be
exercised in this development environment (no ROOT/PyROOT install available
here)** — smoke-test `write_canvases` on a small file before trusting it on
a full comparison; the `TRatioPlot` two-histogram convention (`ratio = h1/h2`)
should be double-checked against your ROOT version.

### Option reference

| Option | Default | Description |
|---|---|---|
| `--config` | *(required)* | Training config, e.g. `configs/graph_vae.yaml` — supplies `window_size`/`n_temporal_bins`/`stride`/`node_features`/`standardize_by`/`channel_map` |
| `--file-a` / `--file-b` | *(required)* | Reference/good and comparison/bad events npz |
| `--label-a` / `--label-b` | `good` / `bad` | |
| `--output` | *(required)* | Output ROOT file (histograms + canvases) |
| `--standardization` | *(none)* | Optional `standardization.npz` (e.g. from a checkpoint dir); if omitted, fits from `--file-a` |
| `--channel-map` | `data.channel_map` in `--config` | |
| `--channel-range LO HI` | full detector | Restrict per-channel output (with `--per-channel`) |
| `--per-channel` | off | Also write per-channel histograms (large — combine with `--channel-range`) |
| `--no-canvases` | off | Skip the PyROOT canvas section |
| `--bins` / `--range-mode` / `--range-percentiles` | `60` / `percentile` / `0.1 99.9` | Same semantics as `compare_events_distributions.py` |
| `--max-windows` | unlimited | Cap windows processed per file — fast first look |

## xrootd_mirror_ci_data.py

Mirrors `reco/` and `decode/` ROOT files from the `CI_build_lar_ci_*` DQM
dCache area to a local directory via `xrdcp`, preserving the source
directory structure (`<dest>/CI_build_lar_ci_<N>/reco/...`,
`<dest>/CI_build_lar_ci_<N>/decode/...`).

```bash
python scripts/xrootd_mirror_ci_data.py \
    --source-glob '/pnfs/sbnd/scratch/ci_validation/dqm/v09_93_01_02/CI_build_lar_ci*' \
    --dest /exp/sbnd/data/users/<you>/DQM/ci_mirror \
    --xrootd-door root://fndca1.fnal.gov:1094 \
    --dry-run
```
Drop `--dry-run` once the file list/commands it prints look right.

**You must supply `--xrootd-door` yourself** — the correct xrootd
redirector/door for your dCache instance isn't something that could be
confirmed from this environment; verify it (e.g. copy one small file first)
before pointing this at the full dataset. `--subdirs` defaults to
`reco decode`; pass `--subdirs raw_decode` etc. if your area uses a
different name than `decode` (this project's `data/README.md` documents
`raw_decode` elsewhere — check `ls` on your actual area first).

Listing/globbing under `/pnfs` is normal POSIX filesystem access (dCache's
NFS4 namespace mount); only the actual file copy goes through `xrdcp`, which
is the correct way to move real data off dCache rather than a plain `cp`.
Re-running the same command is safe and resumes automatically — files
already present at the destination are skipped unless `--overwrite` is
given, so a partially-failed run can just be re-launched. Failed transfers
are logged individually and summarized at the end.

File discovery, path-mirroring, URL construction, skip-existing behavior,
and dry-run orchestration are unit-tested against a synthetic directory
tree. The actual `xrdcp` subprocess call against a real dCache instance
could not be exercised in this development environment — do a `--dry-run`
first, then a small real transfer, before mirroring the full dataset.

| Option | Default | Description |
|---|---|---|
| `--source-glob` | *(required)* | Glob matching per-build top-level directories |
| `--dest` | *(required)* | Local destination root |
| `--copy-method` | `xrdcp` | `xrdcp` (recommended for real dCache transfers) or `cp` — a plain filesystem copy (`shutil.copy2`) that bypasses xrootd entirely, for quick local testing or when `/pnfs` is directly POSIX-readable |
| `--xrootd-door` | *(required unless `--copy-method cp`)* | e.g. `root://fndca1.fnal.gov:1094` — verify for your site |
| `--subdirs` | `reco decode` | Subdirectory name(s) to mirror under each matched build dir |
| `--exclude-dirs` | `log` | Directory name(s) to skip anywhere in the path under a subdir (e.g. `reco/log/foo.root` or a nested `.../some_job/log/foo.root` are both excluded) |
| `--overwrite` | off | Re-copy files that already exist at the destination |
| `--dry-run` | off | Print what would be copied / the exact `xrdcp` commands, copy nothing |
| `--max-workers` | `4` | Parallel `xrdcp` transfers |

## compare_hits_to_waveform.py

Overlays hit-finder Gaussian fits on raw ADC waveforms, per channel, for one
event — a direct visual check of how well hit-finding is performing. Takes
two files describing the *same* event from two stages of the pipeline,
matched by run/subrun/event:

- `--raw-file` — the flat `rawdigits` ntuple from `dump_rawdigits.C`
  (read via `sbn_anomaly.data.raw_digit_reader.RawDigitReader`)
- `--hits-file` — the reco/caloskim ROOT file with hit-finder output (same
  `hits0.h`/`hits1.h`/`hits2.h.*` branches `SparseWindowDatasetPyG.from_root`
  reads — see [`data/README.md`](../data/README.md#sparse-event-format))

```bash
python scripts/compare_hits_to_waveform.py \
    --raw-file good_raw_poc.root \
    --hits-file /pnfs/sbnd/.../reco/run19305_evt0.root \
    --run 19305 --subrun 1 --event 42 \
    --output hit_check_run19305_evt42.root
```

For each active channel (has a hit, or a raw deviation `--activity-threshold`
sigma above its own robust noise floor — `--all-channels` forces every
channel), draws the pedestal-subtracted waveform with one Gaussian per hit,
parameterized directly from the hit's own fields (mean = hit time, sigma =
hit width; amplitude from a `.amplitude` branch if present, otherwise
derived from `integral = amplitude * width * sqrt(2*pi)`). Output is one
`TCanvas` per channel, grouped by plane:
```
event_run<r>_subrun<s>_evt<e>/plane<p>/ch<channel>
```

**Read this caveat before concluding hit-finding looks "bad":** the hit
finder almost always fits the *deconvolved* wire signal, not raw ADC
directly. Raw induction-plane waveforms are bipolar; the deconvolved signal
the fit was performed on is approximately unipolar/Gaussian. A poor-looking
overlay on an **induction** channel doesn't by itself mean hit-finding is
malfunctioning — it may just be comparing two different signal
representations. The comparison is most directly meaningful on the
**collection** plane, where raw and deconvolved shapes are closer. This
script does not deconvolve (it doesn't have the field/electronics response
on hand) — it draws exactly what's in `--raw-file`, pedestal-subtracted and
optionally coherent-noise-subtracted (`--remove-coherent`).

### Requires

**PyROOT** (drawing/writing `TCanvas`/`TF1`) **and uproot+awkward** (reading
the hits tree). The amplitude-derivation math, activity detection (median-
absolute-deviation noise floor), pedestal/coherent-noise preprocessing (via
`raw_preprocess.py`), and run/subrun/event matching are unit-tested,
including a round-trip check that the derived Gaussian amplitude reproduces
the hit's stored integral exactly. **The PyROOT drawing section could not be
exercised in this development environment** — smoke-test on one event
before a larger run.

| Option | Default | Description |
|---|---|---|
| `--raw-file` / `--hits-file` | *(required)* | See above |
| `--run` / `--subrun` / `--event` | *(required unless `--event-index`)* | Event to match across both files |
| `--event-index` | *(none)* | Alternative: match by entry order instead (only valid if both files share the same event ordering) |
| `--output` | *(required)* | Output ROOT file |
| `--tree-name` | `caloskim/TrackCaloSkim` | |
| `--hit-branches` | matches `configs/graph_vae.yaml` | Used only to discover the `hits0.h`/etc. prefixes |
| `--remove-coherent` | off | Also subtract common-mode noise per electronics group |
| `--coherent-group-size` | `64` | |
| `--activity-threshold` | `5.0` | Sigma above a channel's own noise floor to count as active when it has no hits |
| `--all-channels` | off | Draw every channel, including inactive ones |
| `--channel-range LO HI` | full detector | Restrict to a channel range |

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
