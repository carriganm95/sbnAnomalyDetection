#!/usr/bin/env python3
"""Compare the EXACT graph_vae node_features between two events npz files --
both raw (physical units) and standardized (the actual tensor values the
model trains/scores on) -- with overlay+ratio PyROOT canvases.

This is the model-accurate companion to compare_events_distributions.py.
That script compares a generic menu of window stats (mean/median/stdev/...)
on whichever raw hit variable you pick; it's a fast, torch-free first look,
but it can drift from what the model actually computes -- e.g. it has no way
to reproduce `occupancy` (hits-with-a-hit / events-in-bin, not a per-hit
reduction) or `sp_fraction`, and it can't show standardized values at all.

This script instead builds the REAL `SparseWindowDatasetPyG` from --config's
`data` section and reads `data.y` directly out of `__getitem__` for every
window -- the identical code path training/inference use -- so there is no
second implementation of the feature math to drift out of sync. For each
`data.node_features` entry and each temporal bin, it recovers both:

  - raw:          the physical-unit value _compute_frame produced, before
                  standardization (recovered by inverting the model's own
                  (x - mean) / std with that same mean/std -- not
                  recomputed from scratch)
  - standardized: `data.y` itself -- literally what the loss function and
                  the model's reconstruction target look like

File A's fitted (or an externally supplied checkpoint's) standardization is
reused for File B too -- never refit per file -- matching real inference,
where good-run statistics are applied to whatever data comes in.

Output: the same all_detector / per_plane / per_channel(opt-in) ROOT
histogram layout compare_events_distributions.py uses (one output file,
browsable the same way), PLUS overlay+ratio PyROOT canvases (good vs. bad
histograms overlaid, ratio = bad/good in a pad below, via TRatioPlot) at the
all_detector and per_plane level by default.

Requires: torch/torch_geometric (real dataset construction) AND PyROOT (for
the canvases -- not just uproot, since uproot can't draw/write TCanvas
objects). Run this in your full LArSoft/analysis environment, not the
torch-free environment compare_events_distributions.py is designed for.

Usage
-----
    python scripts/compare_model_features.py \\
        --config configs/graph_vae.yaml \\
        --file-a data/good_events_val.npz --label-a good \\
        --file-b data/bad_events_val.npz  --label-b bad \\
        --output model_features_compare.root

Reuse a checkpoint's fitted standardization instead of refitting from
--file-a (recommended if you have a trained model already -- matches
exactly what it trained against):
    python scripts/compare_model_features.py \\
        --config configs/graph_vae.yaml \\
        --file-a data/good_events_val.npz --file-b data/bad_events_val.npz \\
        --standardization checkpoints/graph_vae/v7/standardization.npz \\
        --output model_features_compare.root

Restrict to the flagged collection-plane region and skip canvases (e.g. no
PyROOT available on this node):
    python scripts/compare_model_features.py \\
        --config configs/graph_vae.yaml \\
        --file-a data/good_events_val.npz --file-b data/bad_events_val.npz \\
        --channel-range 4000 5500 --no-canvases --output collection_only.root

Output layout
-------------
    model_features/<feature>/bin<b>/raw/all_detector_<label>
    model_features/<feature>/bin<b>/raw/per_plane/plane<p>/<label>
    model_features/<feature>/bin<b>/raw/per_channel/ch<channel>/<label>          (opt-in)
    model_features/<feature>/bin<b>/standardized/all_detector_<label>
    model_features/<feature>/bin<b>/standardized/per_plane/plane<p>/<label>
    model_features/<feature>/bin<b>/standardized/per_channel/ch<channel>/<label>  (opt-in)

    canvases/<raw|standardized>/<feature>/bin<b>/all_detector
    canvases/<raw|standardized>/<feature>/bin<b>/per_plane/plane<p>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Pure-numpy helpers shared with compare_events_distributions.py -- same
# grouping/writing/range logic, same output conventions.
from compare_events_distributions import (  # noqa: E402
    _channel_plane_lookup,
    _channel_to_plane_lut,
    _group_indices,
    _hist_range,
    _write_all_detector,
    _write_per_channel,
    _write_per_plane,
)

logger = logging.getLogger("compare_model_features")


# ---------------------------------------------------------------------------
# Dataset construction and per-window extraction
# ---------------------------------------------------------------------------

def _load_config_data_section(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("data", {})


def _common_dataset_kwargs(data_cfg: dict, channel_map: Optional[str], n_channels: Optional[int]) -> dict:
    return dict(
        history=1,  # unused in reconstruction mode
        window_size=int(data_cfg.get("window_size", 100)),
        n_bins=int(data_cfg.get("n_temporal_bins", 4)),
        stride=int(data_cfg.get("stride", 100)),
        radius=int(data_cfg.get("adjacency_radius", 4)),
        node_features=data_cfg.get("node_features") or None,
        prune_inactive=bool(data_cfg.get("prune_inactive", True)),
        channel_map=channel_map or data_cfg.get("channel_map"),
        edge_mode=str(data_cfg.get("edge_mode", "sequential")),
        reconstruction=True,
        standardize=True,
        standardize_by=str(data_cfg.get("standardize_by", "global")),
        min_plane_samples=int(data_cfg.get("min_plane_samples", 20)),
        log_features=data_cfg.get("log_features") or None,
        n_channels=n_channels or data_cfg.get("n_channels"),
    )


def build_datasets(
    file_a: str,
    file_b: str,
    config_path: str,
    standardization_path: Optional[str],
    channel_map: Optional[str],
    n_channels: Optional[int],
):
    """Build File A's dataset (fits/loads standardization) and File B's
    dataset (reusing File A's exact feature_mean/feature_std) -- never refit
    per file, matching real inference behavior.
    """
    from sbn_anomaly.data.sparse_window_dataset import SparseWindowDatasetPyG

    data_cfg = _load_config_data_section(config_path)
    kwargs = _common_dataset_kwargs(data_cfg, channel_map, n_channels)

    feat_mean = feat_std = None
    if standardization_path:
        s = np.load(standardization_path)
        feat_mean, feat_std = s["feature_mean"], s["feature_std"]
        logger.info("Loaded standardization from %s", standardization_path)

    logger.info("Building dataset A (%s) ...", file_a)
    ds_a = SparseWindowDatasetPyG.from_npz(file_a, feature_mean=feat_mean, feature_std=feat_std, **kwargs)
    logger.info("Building dataset B (%s), reusing A's standardization ...", file_b)
    ds_b = SparseWindowDatasetPyG.from_npz(
        file_b, feature_mean=ds_a.feature_mean, feature_std=ds_a.feature_std, **kwargs)
    return ds_a, ds_b


def extract_feature_values(ds, max_windows: Optional[int] = None, progress_every: int = 50) -> dict:
    """Walk every window of `ds` via its real __getitem__ and recover, per
    (feature_name, bin_index): parallel (channel, standardized, raw) arrays
    across all (window, active-channel) pairs.

    `raw` is recovered by inverting the model's own standardization
    ((y * std) + mean) with `ds.feature_mean`/`ds.feature_std` -- the exact
    same affine map __getitem__ applied, just run backwards -- rather than
    calling _compute_frame a second time. Handles both global ((F,)) and
    per-plane ((num_nodes, F)) standardization shapes.

    `ds` just needs to behave like SparseWindowDatasetPyG: len(), __getitem__
    returning an object with .y (array-like with .numpy() or already an
    ndarray) and .active_mask, plus .n_bins/.n_node_features/.node_features/
    .feature_mean/.feature_std attributes. Kept duck-typed so this can be
    unit-tested without torch/torch_geometric.
    """
    n = len(ds) if max_windows is None else min(len(ds), max_windows)
    out = {(f, b): {"channel": [], "std": [], "raw": []}
           for b in range(ds.n_bins) for f in ds.node_features}

    std_is_global = np.asarray(ds.feature_std).ndim == 1

    for i in range(n):
        data = ds[i]
        y = data.y.numpy() if hasattr(data.y, "numpy") else np.asarray(data.y)
        active = data.active_mask.numpy() if hasattr(data.active_mask, "numpy") else np.asarray(data.active_mask)

        if std_is_global:
            std_for_rows = ds.feature_std
            mean_for_rows = ds.feature_mean
        else:
            std_for_rows = ds.feature_std[active]
            mean_for_rows = ds.feature_mean[active]
        raw_flat = y * std_for_rows + mean_for_rows

        y_r = y.reshape(y.shape[0], ds.n_bins, ds.n_node_features)
        raw_r = raw_flat.reshape(raw_flat.shape[0], ds.n_bins, ds.n_node_features)

        for fi, fname in enumerate(ds.node_features):
            for b in range(ds.n_bins):
                key = (fname, b)
                out[key]["channel"].append(active)
                out[key]["std"].append(y_r[:, b, fi])
                out[key]["raw"].append(raw_r[:, b, fi])

        if (i + 1) % progress_every == 0:
            logger.info("  extracted %d/%d windows", i + 1, n)

    for key, d in out.items():
        d["channel"] = np.concatenate(d["channel"]) if d["channel"] else np.empty(0, dtype=np.int64)
        d["std"] = np.concatenate(d["std"]).astype(np.float32) if d["std"] else np.empty(0, dtype=np.float32)
        d["raw"] = np.concatenate(d["raw"]).astype(np.float32) if d["raw"] else np.empty(0, dtype=np.float32)
    return out


# ---------------------------------------------------------------------------
# ROOT histogram writing (reuses compare_events_distributions.py's helpers)
# ---------------------------------------------------------------------------

def write_histograms(
    f,
    vals_a: dict,
    vals_b: dict,
    label_a: str,
    label_b: str,
    plane_lut: np.ndarray,
    n_channels: int,
    bins: int,
    range_mode: str,
    range_percentiles,
    channel_range: Optional[tuple],
    do_per_channel: bool,
    progress_every: int,
) -> int:
    n_written = 0
    keys = sorted(set(vals_a) | set(vals_b))
    chan_lo, chan_hi = channel_range if channel_range else (0, n_channels)

    for (fname, b) in keys:
        a = vals_a.get((fname, b), {"channel": np.empty(0), "std": np.empty(0), "raw": np.empty(0)})
        bb = vals_b.get((fname, b), {"channel": np.empty(0), "std": np.empty(0), "raw": np.empty(0)})
        for kind in ("raw", "standardized"):
            src_key = "raw" if kind == "raw" else "std"
            a_val, b_val = a[src_key], bb[src_key]
            rng = _hist_range([a_val, b_val], range_mode, *range_percentiles)
            base = f"model_features/{fname}/bin{b}/{kind}"

            n_written += _write_all_detector(f, f"{base}/all_detector_{label_a}", a_val, bins, rng)
            n_written += _write_all_detector(f, f"{base}/all_detector_{label_b}", b_val, bins, rng)

            if a["channel"].size:
                n_written += _write_per_plane(
                    f, f"{base}/per_plane", a["channel"], a_val,
                    plane_lut[np.clip(a["channel"], 0, n_channels - 1)], bins, rng, label_a)
            if bb["channel"].size:
                n_written += _write_per_plane(
                    f, f"{base}/per_plane", bb["channel"], b_val,
                    plane_lut[np.clip(bb["channel"], 0, n_channels - 1)], bins, rng, label_b)

            if do_per_channel:
                if a["channel"].size:
                    n_written += _write_per_channel(
                        f, f"{base}/per_channel", a["channel"], a_val,
                        (chan_lo, chan_hi), bins, rng, label_a, progress_every, f"{label_a}/{fname}/bin{b}/{kind}")
                if bb["channel"].size:
                    n_written += _write_per_channel(
                        f, f"{base}/per_channel", bb["channel"], b_val,
                        (chan_lo, chan_hi), bins, rng, label_b, progress_every, f"{label_b}/{fname}/bin{b}/{kind}")
    return n_written


# ---------------------------------------------------------------------------
# PyROOT overlay + ratio canvases
#
# NOT exercised in this environment (no ROOT/PyROOT available) -- the
# extraction/writing logic above is unit-tested independently. Smoke-test
# this section on a small file before trusting it on a full dataset.
# ---------------------------------------------------------------------------

def _root_safe_name(value: str) -> str:
    """Return a ROOT-safe object name with no path separators or spaces."""
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value))


def _normalize_histogram_counts(counts: np.ndarray):
    """Return unit-area bin fractions and propagated Poisson errors."""
    raw = np.asarray(counts, dtype=np.float64)
    total = float(np.sum(raw))
    if total <= 0.0:
        return raw.copy(), np.zeros_like(raw)
    return raw / total, np.sqrt(np.clip(raw, 0.0, None)) / total


def _make_th1(
    ROOT,
    name: str,
    title: str,
    values: np.ndarray,
    errors: np.ndarray,
    lo: float,
    hi: float,
):
    """Build a detached TH1D from supplied bin values and errors."""
    values = np.asarray(values, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    if values.shape != errors.shape:
        raise ValueError(f"Histogram values/errors shape mismatch for {name}")

    safe_name = _root_safe_name(name)
    h = ROOT.TH1D(safe_name, title, len(values), float(lo), float(hi))
    h.SetDirectory(0)
    h.Sumw2()

    for i, (value, error) in enumerate(zip(values, errors), start=1):
        h.SetBinContent(i, float(value))
        h.SetBinError(i, float(error))

    ROOT.SetOwnership(h, False)
    return h


def _chi2_ndf(ROOT, name: str, counts_1: np.ndarray, counts_2: np.ndarray, lo: float, hi: float):
    """Run ROOT's unweighted two-sample chi-square test on raw counts."""
    import ctypes

    counts_1 = np.asarray(counts_1, dtype=np.float64)
    counts_2 = np.asarray(counts_2, dtype=np.float64)
    if counts_1.sum() <= 0.0 or counts_2.sum() <= 0.0:
        return 0.0, 0, 0.0

    errors_1 = np.sqrt(np.clip(counts_1, 0.0, None))
    errors_2 = np.sqrt(np.clip(counts_2, 0.0, None))
    h1 = _make_th1(ROOT, f"chi2_{name}_1", "", counts_1, errors_1, lo, hi)
    h2 = _make_th1(ROOT, f"chi2_{name}_2", "", counts_2, errors_2, lo, hi)

    chi2 = ctypes.c_double(0.0)
    ndf = ctypes.c_int(0)
    igood = ctypes.c_int(0)
    pvalue = h1.Chi2TestX(h2, chi2, ndf, igood, "UU")
    return chi2.value, ndf.value, float(pvalue)


def _get_or_make_dir(tdirectory, path: str):
    d = tdirectory
    if not path:
        return d

    for part in path.split("/"):
        sub = d.GetDirectory(part)
        if not sub:
            sub = d.mkdir(part)
        if not sub:
            raise RuntimeError(f"Could not create ROOT directory component: {part}")
        d = sub
    return d


def _draw_ratio_canvas(
    ROOT,
    name: str,
    x_title: str,
    counts_good: np.ndarray,
    counts_bad: np.ndarray,
    lo: float,
    hi: float,
    label_a: str,
    label_b: str,
    logger_=None,
):
    """Create a unit-area overlay and bad/good ratio canvas."""
    raw_good = np.asarray(counts_good, dtype=np.float64)
    raw_bad = np.asarray(counts_bad, dtype=np.float64)

    if raw_good.ndim != 1 or raw_bad.ndim != 1:
        raise ValueError("Histogram count arrays must be one-dimensional")
    if raw_good.size != raw_bad.size:
        raise ValueError(
            f"Histogram bin mismatch for {name}: {raw_good.size} versus {raw_bad.size}"
        )
    if raw_good.size == 0:
        return None
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        if logger_ is not None:
            logger_.warning("Skipping %s: invalid histogram range [%s, %s]", name, lo, hi)
        return None

    good_total = float(raw_good.sum())
    bad_total = float(raw_bad.sum())
    if good_total <= 0.0 or bad_total <= 0.0:
        if logger_ is not None:
            logger_.warning(
                "Skipping canvas %-45s: empty histogram (%s=%g, %s=%g)",
                name, label_a, good_total, label_b, bad_total,
            )
        return None

    plot_good, error_good = _normalize_histogram_counts(raw_good)
    plot_bad, error_bad = _normalize_histogram_counts(raw_bad)

    safe_name = _root_safe_name(name)
    h_good = _make_th1(
        ROOT,
        f"h_{safe_name}_{_root_safe_name(label_a)}",
        f"{name};{x_title};fraction of entries",
        plot_good,
        error_good,
        lo,
        hi,
    )
    h_bad = _make_th1(
        ROOT,
        f"h_{safe_name}_{_root_safe_name(label_b)}",
        f"{name};{x_title};fraction of entries",
        plot_bad,
        error_bad,
        lo,
        hi,
    )

    h_good.SetLineColor(ROOT.kBlue + 1)
    h_good.SetLineWidth(2)
    h_good.SetMarkerStyle(20)
    h_good.SetMarkerColor(ROOT.kBlue + 1)
    h_bad.SetLineColor(ROOT.kRed + 1)
    h_bad.SetLineWidth(2)
    h_bad.SetMarkerStyle(21)
    h_bad.SetMarkerColor(ROOT.kRed + 1)

    chi2, ndf, pvalue = _chi2_ndf(ROOT, safe_name, raw_bad, raw_good, lo, hi)
    if logger_ is not None:
        if ndf > 0:
            logger_.info(
                "  %-45s chi2/ndf = %.2f/%d = %.3f (p=%.4f)",
                name, chi2, ndf, chi2 / ndf, pvalue,
            )
        else:
            logger_.info("  %-45s chi2/ndf unavailable", name)

    canvas = ROOT.TCanvas(f"c_{safe_name}", name, 800, 700)
    ROOT.SetOwnership(canvas, False)
    canvas.cd()

    ratio_plot = ROOT.TRatioPlot(h_bad, h_good)
    ROOT.SetOwnership(ratio_plot, False)
    ratio_plot.SetH1DrawOpt("E")
    ratio_plot.SetH2DrawOpt("E")
    ratio_plot.Draw()

    ratio_plot.GetLowerRefYaxis().SetTitle(f"{label_b}/{label_a}")
    ratio_plot.GetUpperRefYaxis().SetTitle("fraction of entries")

    ratio_plot.GetUpperPad().cd()
    legend = ROOT.TLegend(0.65, 0.72, 0.88, 0.88)
    ROOT.SetOwnership(legend, False)
    legend.AddEntry(h_good, label_a, "lep")
    legend.AddEntry(h_bad, label_b, "lep")
    legend.Draw()

    chi2_text = ROOT.TLatex()
    ROOT.SetOwnership(chi2_text, False)
    chi2_text.SetNDC(True)
    chi2_text.SetTextSize(0.045)
    if ndf > 0:
        chi2_text.DrawLatex(
            0.15,
            0.83,
            f"#chi^{{2}}/ndf = {chi2:.2f}/{ndf} = {chi2 / ndf:.2f} (p = {pvalue:.3f})",
        )
    else:
        chi2_text.DrawLatex(0.15, 0.83, "#chi^{2}/ndf unavailable")

    canvas.Modified()
    canvas.Update()
    canvas._keepalive = (h_good, h_bad, ratio_plot, legend, chi2_text)
    return canvas

def _write_and_close_canvas(canvas, key_name: str) -> bool:
    """Write one canvas, then close it without letting PyROOT double-delete it."""
    if canvas is None:
        return False

    canvas.Write(key_name, 2)  # TObject::kOverwrite
    canvas.Close()

    # ROOT owns/deletes the drawing objects. Their Python wrappers must not.
    if hasattr(canvas, "_keepalive"):
        del canvas._keepalive
    return True


def write_canvases(
    output_path: str,
    vals_a: dict,
    vals_b: dict,
    label_a: str,
    label_b: str,
    plane_lut: np.ndarray,
    n_channels: int,
    bins: int,
    range_mode: str,
    range_percentiles,
) -> int:
    import ROOT

    ROOT.gROOT.SetBatch(True)
    ROOT.TH1.AddDirectory(False)

    outfile = ROOT.TFile.Open(output_path, "UPDATE")
    if not outfile or outfile.IsZombie():
        raise RuntimeError(f"Could not open ROOT output for canvas writing: {output_path}")

    n_canvases = 0
    keys = sorted(set(vals_a) | set(vals_b))

    try:
        for fname, b in keys:
            empty = {
                "channel": np.empty(0, dtype=np.int64),
                "std": np.empty(0, dtype=np.float32),
                "raw": np.empty(0, dtype=np.float32),
            }
            a = vals_a.get((fname, b), empty)
            bb = vals_b.get((fname, b), empty)

            for kind in ("raw", "standardized"):
                src_key = "raw" if kind == "raw" else "std"
                a_val = np.asarray(a[src_key])
                b_val = np.asarray(bb[src_key])

                # A ratio requires entries on both sides. Histograms are still
                # written by uproot even when only one side has data.
                if a_val.size == 0 or b_val.size == 0:
                    logger.warning(
                        "Skipping %s/%s/bin%d canvases: %s values=%d, %s values=%d",
                        kind,
                        fname,
                        b,
                        label_a,
                        a_val.size,
                        label_b,
                        b_val.size,
                    )
                    continue

                rng = _hist_range([a_val, b_val], range_mode, *range_percentiles)
                lo, hi = float(rng[0]), float(rng[1])
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    logger.warning(
                        "Skipping %s/%s/bin%d canvases: invalid range [%s, %s]",
                        kind,
                        fname,
                        b,
                        lo,
                        hi,
                    )
                    continue

                counts_a, _ = np.histogram(a_val, bins=bins, range=(lo, hi))
                counts_b, _ = np.histogram(b_val, bins=bins, range=(lo, hi))

                if counts_a.sum() <= 0 or counts_b.sum() <= 0:
                    logger.warning(
                        "Skipping %s/%s/bin%d: one histogram is empty after binning",
                        kind, fname, b,
                    )
                    continue

                base_dir = f"canvases/{kind}/{fname}/bin{b}"
                directory = _get_or_make_dir(outfile, base_dir)
                directory.cd()

                unique_name = f"{kind}_{fname}_bin{b}_all_detector"
                canvas = _draw_ratio_canvas(
                    ROOT,
                    unique_name,
                    f"{fname} bin {b} ({kind})",
                    counts_a,
                    counts_b,
                    lo,
                    hi,
                    label_a,
                    label_b,
                    logger_=logger,
                )
                if _write_and_close_canvas(canvas, "all_detector"):
                    n_canvases += 1

                channels = []
                if np.asarray(a["channel"]).size:
                    channels.append(np.asarray(a["channel"], dtype=np.int64))
                if np.asarray(bb["channel"]).size:
                    channels.append(np.asarray(bb["channel"], dtype=np.int64))
                if not channels:
                    continue

                all_channels = np.clip(np.concatenate(channels), 0, n_channels - 1)
                planes = sorted({int(p) for p in plane_lut[all_channels] if p >= 0})

                for plane in planes:
                    if np.asarray(a["channel"]).size:
                        a_channels = np.asarray(a["channel"], dtype=np.int64)
                        a_plane = plane_lut[np.clip(a_channels, 0, n_channels - 1)]
                        pa = a_val[a_plane == plane]
                    else:
                        pa = np.empty(0, dtype=np.float32)

                    if np.asarray(bb["channel"]).size:
                        b_channels = np.asarray(bb["channel"], dtype=np.int64)
                        b_plane = plane_lut[np.clip(b_channels, 0, n_channels - 1)]
                        pb = b_val[b_plane == plane]
                    else:
                        pb = np.empty(0, dtype=np.float32)

                    if pa.size == 0 or pb.size == 0:
                        logger.warning(
                            "Skipping %s/%s/bin%d/plane%d: %s values=%d, %s values=%d",
                            kind,
                            fname,
                            b,
                            plane,
                            label_a,
                            pa.size,
                            label_b,
                            pb.size,
                        )
                        continue

                    counts_pa, _ = np.histogram(pa, bins=bins, range=(lo, hi))
                    counts_pb, _ = np.histogram(pb, bins=bins, range=(lo, hi))

                    plane_dir = _get_or_make_dir(outfile, f"{base_dir}/per_plane")
                    plane_dir.cd()

                    unique_plane_name = f"{kind}_{fname}_bin{b}_plane{plane}"
                    plane_canvas = _draw_ratio_canvas(
                        ROOT,
                        unique_plane_name,
                        f"{fname} bin {b} plane {plane} ({kind})",
                        counts_pa,
                        counts_pb,
                        lo,
                        hi,
                        label_a,
                        label_b,
                        logger_=logger,
                    )
                    if _write_and_close_canvas(plane_canvas, f"plane{plane}"):
                        n_canvases += 1

        outfile.cd()
        outfile.Write("", 2)
    finally:
        # Prevent gPad from retaining a pointer to a canvas that was closed.
        try:
            ROOT.gROOT.SetSelectedPad(0)
        except Exception:
            pass
        outfile.Close()

    return n_canvases


# ---------------------------------------------------------------------------

def compare(
    file_a: str,
    file_b: str,
    config: str,
    output: str,
    label_a: str = "good",
    label_b: str = "bad",
    standardization: Optional[str] = None,
    channel_map: Optional[str] = None,
    n_channels: Optional[int] = None,
    channel_range: Optional[tuple] = None,
    do_per_channel: bool = False,
    do_canvases: bool = True,
    bins: int = 60,
    range_mode: str = "percentile",
    range_percentiles=(0.1, 99.9),
    max_windows: Optional[int] = None,
    progress_every: int = 1000,
    uproot_module=None,
) -> int:
    uproot = uproot_module
    if uproot is None:
        import uproot as uproot  # noqa: PLW0127

    ds_a, ds_b = build_datasets(file_a, file_b, config, standardization, channel_map, n_channels)
    n_channels = n_channels or ds_a.num_nodes

    if len(ds_a) == 0:
        logger.warning(
            "Dataset A produces zero windows. Its histograms will be empty and "
            "all ratio canvases will be skipped. Check that the file contains "
            "at least data.window_size events."
        )
    if len(ds_b) == 0:
        logger.warning(
            "Dataset B produces zero windows. Its histograms will be empty and "
            "all ratio canvases will be skipped. Check that the file contains "
            "at least data.window_size events."
        )

    if ds_a._planes_flat is not None:
        plane_lut = _channel_to_plane_lut(ds_a._channels_flat, ds_a._planes_flat, n_channels)
    elif channel_map or ds_a.channel_map:
        plane_lut = _channel_plane_lookup(channel_map or ds_a.channel_map, n_channels)
    else:
        raise ValueError("No plane info: no planes_flat in the events npz and no --channel-map given.")

    logger.info("Extracting feature values from dataset A (%d windows) ...", len(ds_a))
    vals_a = extract_feature_values(ds_a, max_windows=max_windows, progress_every=progress_every)
    logger.info("Extracting feature values from dataset B (%d windows) ...", len(ds_b))
    vals_b = extract_feature_values(ds_b, max_windows=max_windows, progress_every=progress_every)

    n_written = 0
    with uproot.recreate(output) as f:
        n_written = write_histograms(
            f, vals_a, vals_b, label_a, label_b, plane_lut, n_channels, bins,
            range_mode, range_percentiles, channel_range, do_per_channel, progress_every,
        )
    logger.info("Wrote %d histograms to %s", n_written, output)

    if do_canvases:
        n_canvases = write_canvases(
            output, vals_a, vals_b, label_a, label_b, plane_lut, n_channels,
            bins, range_mode, range_percentiles,
        )
        logger.info("Wrote %d canvases to %s", n_canvases, output)

    return n_written


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="e.g. configs/graph_vae.yaml")
    ap.add_argument("--file-a", required=True, help="Reference/good events npz -- also fits standardization unless --standardization is given")
    ap.add_argument("--file-b", required=True, help="Comparison/bad events npz -- standardized with FILE A's stats, never refit")
    ap.add_argument("--label-a", default="good")
    ap.add_argument("--label-b", default="bad")
    ap.add_argument("--output", required=True)
    ap.add_argument("--standardization", default=None,
                     help="Optional standardization.npz (e.g. from a checkpoint dir) -- if given, "
                          "used instead of fitting from --file-a")
    ap.add_argument("--channel-map", default=None, help="Default: data.channel_map in --config")
    ap.add_argument("--n-channels", type=int, default=None)
    ap.add_argument("--channel-range", nargs=2, type=int, default=None, metavar=("LO", "HI"))
    ap.add_argument("--per-channel", action="store_true",
                     help="Also write per-channel histograms (off by default -- large; combine with --channel-range)")
    ap.add_argument("--no-canvases", action="store_true", help="Skip PyROOT overlay+ratio canvases (requires PyROOT)")
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--range-mode", choices=["percentile", "minmax"], default="percentile")
    ap.add_argument("--range-percentiles", nargs=2, type=float, default=[0.1, 99.9])
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--progress-every", type=int, default=1000)
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    compare(
        file_a=args.file_a, file_b=args.file_b, config=args.config, output=args.output,
        label_a=args.label_a, label_b=args.label_b, standardization=args.standardization,
        channel_map=args.channel_map, n_channels=args.n_channels,
        channel_range=tuple(args.channel_range) if args.channel_range else None,
        do_per_channel=args.per_channel, do_canvases=not args.no_canvases,
        bins=args.bins, range_mode=args.range_mode, range_percentiles=tuple(args.range_percentiles),
        max_windows=args.max_windows, progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
