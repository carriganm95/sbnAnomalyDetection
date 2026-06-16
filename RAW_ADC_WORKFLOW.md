# Raw-ADC Anomaly Detection Workflow

This is the raw-waveform counterpart to the hit-level GNN forecaster. Instead of
reconstructed hit aggregates, it works directly on the raw ADC samples in the
`raw::RawDigit` products, compresses each channel's waveform with a
**variational autoencoder (VAE)**, and feeds the per-channel latents into the
**same** `GNNForecasterPyG` + GRU forecasting stack used by `configs/gnn.yaml`.

Everything here is additive — the existing `tpc`, `pmt`, `fusion`, `window`, and
`gnn` model types are untouched.

```
art RawDigit files                    (per channel: Channel(), Samples(),
   │   scripts/dump_rawdigits.C        GetPedestal(), ADC(itick))
   ▼
flat raw-ADC ntuple  (TTree "rawdigits": run, subrun, event, channel[],
   │                   pedestal[], adc[nchan*nticks])
   ▼   RawDigitReader + raw_preprocess (pedestal sub, coherent-noise removal, scale)
per-channel waveforms ──► TPCWaveformVAE (1-D conv VAE) ──► latent z_c per channel
   │                                                    │  (+ recon/KL anomaly score)
   ▼   build_raw_latents                                ▼
windows array (num_events, n_channels, latent_dim) ──► GraphWindowDatasetPyG
   ▼
GNNForecasterPyG (GCN spatial + GRU temporal) ──► per-channel forecast MSE = anomaly score
```

## Why this shape

* **Compression first.** ~10k channels × thousands of ticks per event is far too
  much to graph directly. The VAE turns each channel waveform into a small latent
  (default 24 dims), a >100× reduction, while keeping amplitude/shape information.
* **Variational, not plain AE.** The KL term gives a smooth, regularised latent
  space that behaves better for density-based anomaly scoring, and the VAE's own
  reconstruction error + KL is already a standalone per-channel anomaly signal.
* **Reuse the GNN.** Per-channel latents are exactly the node-feature contract of
  `GraphWindowDatasetPyG`, so the spatial+temporal forecasting code is reused
  unchanged. One **event** is one temporal frame; the GRU forecasts the next
  event's latents and flags channels whose prediction error is large.

## Step 0 — Dump raw digits to a flat ntuple

`raw::RawDigit` lives in art files and may be Huffman-compressed, so we let
gallery/LArSoft decode it and write a flat tree uproot can read. From a LArSoft
environment with gallery set up:

```bash
root -l -b -q 'scripts/dump_rawdigits.C("data_..._tpcdecode.root", "raw_run10376.root", "daq", 0)'
# args: input art file, output ntuple, input tag ("daq"), nevents (0 = all)
```

This writes a TTree `rawdigits` with one entry per event and jagged per-channel
arrays. Loop a file list in a shell to convert many runs.

## Step 1 — Train the waveform VAE

```bash
sbn-train --config configs/raw_vae.yaml --root-files raw_run*.root
```

Key knobs in `configs/raw_vae.yaml`:

* `model.input_length` — waveform length the VAE expects (must be divisible by
  `2**model.depth`; waveforms are padded/truncated to it). SBND ≈ 3415 ticks,
  ICARUS ≈ 4096.
* `model.latent_dim` — per-channel compressed size = GNN node-feature dim.
* `data.preprocess` — pedestal subtraction (uses `GetPedestal()`), coherent-noise
  removal (`coherent_group_size` = channels per FEMB/motherboard group), and a
  fixed ADC `scale`.
* `data.channels_per_event` — random channel subsample per event to bound memory.
* `training.beta` / `beta_warmup_epochs` — KL weight and warmup to avoid early
  posterior collapse.

**Optional — cache the training input as an npz** (avoids re-streaming ROOT each
epoch):

```bash
python -m sbn_anomaly.data.build_raw_waveforms \
    --config configs/raw_vae.yaml --root-files raw_run*.root \
    --output data/raw_waveforms_train.npz --max-waveforms 2000000
```

Then set `data.waveforms_path: data/raw_waveforms_train.npz` in the config and
run `sbn-train --config configs/raw_vae.yaml` **without** `--root-files`.

Score channels directly with the VAE (reconstruction MSE + β·KL per channel):

```bash
sbn-infer --config configs/raw_vae.yaml --root-files raw_run.root --output raw_vae_scores.npz
# -> node_scores (N_events, n_channels), scores, scores_max, provenance
```

## Step 2 — Materialise per-channel latents

```bash
python -m sbn_anomaly.data.build_raw_latents \
    --config configs/raw_gnn.yaml \
    --vae-checkpoint checkpoints/raw_vae/v1/vae_final.pt \
    --root-files raw_run*.root \
    --output data/raw_latents_train.npz
```

Produces `windows` of shape `(num_events, n_channels, latent_dim)` (missing
channels left as zeros) plus `(run, subrun, event)` provenance. The VAE params in
`configs/raw_gnn.yaml` (`input_length`, `latent_dim`, `depth`, ...) **must match**
the trained checkpoint.

## Step 3 — Train / run the raw GNN forecaster

```bash
sbn-train --config configs/raw_gnn.yaml          # reads data.windows_path
sbn-infer --config configs/raw_gnn.yaml --output raw_gnn_scores.npz
```

Output mirrors the existing GNN inference: per-window, per-channel forecast MSE
(`node_scores`), plus per-window mean/max scores and provenance.

## Validation suggestion

You likely have no labelled anomalies. Validate by **injecting synthetic
pathologies** into good runs before Step 1 — dead/saturated channels, stuck ADC
codes, noise bursts, missing FEMBs — and confirm the VAE score and the GNN
forecast residual localise them to the right channels. This is the single most
important step before trusting a real-time DQM deployment.

## Files added

| File | Role |
|------|------|
| `scripts/dump_rawdigits.C` | gallery macro: art RawDigits → flat ntuple |
| `sbn_anomaly/data/raw_digit_reader.py` | uproot reader for the flat ntuple |
| `sbn_anomaly/data/raw_preprocess.py` | pedestal sub, coherent-noise removal, scaling (NumPy) |
| `sbn_anomaly/data/raw_waveform_dataset.py` | per-channel waveform datasets (stream + array) |
| `sbn_anomaly/models/tpc_waveform_vae.py` | `TPCWaveformVAE` 1-D conv VAE |
| `sbn_anomaly/train/vae_trainer.py` | `VAETrainer` (β-VAE ELBO) |
| `sbn_anomaly/data/build_raw_latents.py` | encode waveforms → latent windows for the GNN |
| `configs/raw_vae.yaml`, `configs/raw_gnn.yaml` | stage-1 / stage-2 configs |
| `tests/test_raw_preprocess.py`, `tests/test_raw_vae.py`, `tests/test_raw_latents.py` | tests |
```
