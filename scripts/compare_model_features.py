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

def _make_th1(ROOT, name: str, title: str, counts: np.ndarray, lo: float, hi: float):
    """Build a TH1D from precomputed bin counts with Poisson (sqrt-N)
    statistical errors.

    counts are raw entry counts (np.histogram output), not weighted fills, so
    the standard counting-statistics error is sqrt(N) per bin. Sumw2() alone
    is fill-order-dependent (it back-fills fSumw2[bin] = content[bin] only if
    called after SetBinContent, assuming unit weights) -- set bin errors
    explicitly as well so this doesn't silently depend on that behavior.
    """
    h = ROOT.TH1D(name, title, len(counts), lo, hi)
    for i, c in enumerate(counts):
        h.SetBinContent(i + 1, float(c))
    h.Sumw2()
    for i, c in enumerate(counts):
        h.SetBinError(i + 1, float(np.sqrt(max(float(c), 0.0))))
    h.SetDirectory(0)
    return h


def _chi2_ndf(ROOT, h1, h2):
    """Poisson two-sample chi2 test between h1 and h2 via ROOT's own
    TH1::Chi2TestX (not reimplemented here) -- "UU" (unweighted-unweighted)
    is the correct option since both histograms are raw entry counts with
    sqrt(N) errors, not weighted fills.

    Returns (chi2, ndf, pvalue). ndf can come back 0 if too few bins have
    entries in both histograms (e.g. a very small proof-of-concept sample) --
    callers must guard against dividing by it.

    Uses ctypes for the Double_t&/Int_t& out-parameters (the modern PyROOT
    convention, >= 6.22-ish). If your PyROOT is old enough that this doesn't
    bind correctly, the legacy fallback is ROOT.Long()/ROOT.Double() proxy
    objects in place of the ctypes ones below.
    """
    import ctypes
    chi2 = ctypes.c_double(0.0)
    ndf = ctypes.c_int(0)
    igood = ctypes.c_int(0)
    pvalue = h1.Chi2TestX(h2, chi2, ndf, igood, "UU")
    return chi2.value, ndf.value, pvalue


def _get_or_make_dir(tdirectory, path: str):
    d = tdirectory
    if not path:
        return d
    for part in path.split("/"):
        sub = d.GetDirectory(part)
        if not sub:
            sub = d.mkdir(part)
        d = sub
    return d


def _draw_ratio_canvas(ROOT, name: str, x_title: str, counts_good: np.ndarray, counts_bad: np.ndarray,
                        lo: float, hi: float, label_a: str, label_b: str, logger_=None):
    """One overlay canvas: good vs bad histograms (with Poisson error bars)
    on top, ratio (bad/good, errors propagated from both) below, via
    TRatioPlot, plus a chi2/ndf annotation from ROOT's own two-sample Poisson
    chi2 test. Both histograms share (bins, lo, hi) so bin-by-bin comparison
    (ratio and chi2 alike) is meaningful.
    """
    h_good = _make_th1(ROOT, f"{name}_good", f"{name};{x_title};entries", counts_good, lo, hi)
    h_bad = _make_th1(ROOT, f"{name}_bad", name, counts_bad, lo, hi)
    h_good.SetLineColor(ROOT.kBlue + 1)
    h_good.SetLineWidth(2)
    h_good.SetMarkerStyle(20)
    h_good.SetMarkerColor(ROOT.kBlue + 1)
    h_bad.SetLineColor(ROOT.kRed + 1)
    h_bad.SetLineWidth(2)
    h_bad.SetMarkerStyle(21)
    h_bad.SetMarkerColor(ROOT.kRed + 1)

    chi2, ndf, pvalue = _chi2_ndf(ROOT, h_bad, h_good)
    if logger_ is not None:
        if ndf > 0:
            logger_.info("  %-40s chi2/ndf = %.2f/%d = %.3f  (p=%.4f)", name, chi2, ndf, chi2 / ndf, pvalue)
        else:
            logger_.info("  %-40s chi2/ndf: ndf=0 (too few bins with entries in both histograms)", name)

    c = ROOT.TCanvas(f"c_{name}", name, 800, 700)
    c.cd()
    # TRatioPlot(h1, h2) plots ratio = h1 / h2 -- put bad first so the ratio
    # panel reads bad/good (matches the ratio convention used elsewhere in
    # these diagnostics: deviation from 1 = bad differs from good). Draw
    # option "E" on both so the overlay shows Poisson error bars, not just
    # outlines.
    rp = ROOT.TRatioPlot(h_bad, h_good)
    rp.SetH1DrawOpt("E")
    rp.SetH2DrawOpt("E")
    rp.Draw()
    rp.GetLowerRefYaxis().SetTitle(f"{label_b}/{label_a}")
    rp.GetUpperRefYaxis().SetTitle("entries")
    rp.GetUpperPad().cd()
    legend = ROOT.TLegend(0.65, 0.72, 0.88, 0.88)
    legend.AddEntry(h_good, label_a, "lep")
    legend.AddEntry(h_bad, label_b, "lep")
    legend.Draw()

    chi2_text = ROOT.TLatex()
    chi2_text.SetNDC(True)
    chi2_text.SetTextSize(0.045)
    if ndf > 0:
        chi2_text.DrawLatex(
            0.15, 0.83, f"#chi^{{2}}/ndf = {chi2:.2f}/{ndf} = {chi2 / ndf:.2f}  (p = {pvalue:.3f})")
    else:
        chi2_text.DrawLatex(0.15, 0.83, "#chi^{2}/ndf: too few populated bins")

    c.Update()
    # Keep references alive on the canvas object so they aren't garbage
    # collected before Write() -- ROOT/PyROOT ownership footgun.
    c._keepalive = (h_good, h_bad, rp, legend, chi2_text)
    return c


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

    outfile = ROOT.TFile(output_path, "UPDATE")
    n_canvases = 0
    keys = sorted(set(vals_a) | set(vals_b))
    for (fname, b) in keys:
        a = vals_a.get((fname, b), {"channel": np.empty(0), "std": np.empty(0), "raw": np.empty(0)})
        bb = vals_b.get((fname, b), {"channel": np.empty(0), "std": np.empty(0), "raw": np.empty(0)})
        for kind in ("raw", "standardized"):
            src_key = "raw" if kind == "raw" else "std"
            a_val, b_val = a[src_key], bb[src_key]
            if a_val.size == 0 and b_val.size == 0:
                continue
            rng = _hist_range([a_val, b_val], range_mode, *range_percentiles)
            lo, hi = rng
            base_dir = f"canvases/{kind}/{fname}/bin{b}"

            counts_a, _ = np.histogram(a_val, bins=bins, range=rng) if a_val.size else (np.zeros(bins), None)
            counts_b, _ = np.histogram(b_val, bins=bins, range=rng) if b_val.size else (np.zeros(bins), None)
            d = _get_or_make_dir(outfile, f"{base_dir}")
            d.cd()
            c = _draw_ratio_canvas(ROOT, "all_detector", f"{fname} bin{b} ({kind})",
                                    counts_a, counts_b, lo, hi, label_a, label_b, logger_=logger)
            c.Write("all_detector")
            n_canvases += 1

            if a["channel"].size or bb["channel"].size:
                planes = sorted(set(
                    int(p) for p in plane_lut[np.clip(
                        np.concatenate([a["channel"], bb["channel"]]).astype(np.int64), 0, n_channels - 1)]
                    if p >= 0
                ))
                for p in planes:
                    pa = a_val[plane_lut[np.clip(a["channel"], 0, n_channels - 1)] == p] if a["channel"].size else np.empty(0)
                    pb = b_val[plane_lut[np.clip(bb["channel"], 0, n_channels - 1)] == p] if bb["channel"].size else np.empty(0)
                    if pa.size == 0 and pb.size == 0:
                        continue
                    counts_pa, _ = np.histogram(pa, bins=bins, range=rng) if pa.size else (np.zeros(bins), None)
                    counts_pb, _ = np.histogram(pb, bins=bins, range=rng) if pb.size else (np.zeros(bins), None)
                    pd = _get_or_make_dir(outfile, f"{base_dir}/per_plane")
                    pd.cd()
                    cp = _draw_ratio_canvas(ROOT, f"plane{p}", f"{fname} bin{b} plane{p} ({kind})",
                                             counts_pa, counts_pb, lo, hi, label_a, label_b, logger_=logger)
                    cp.Write(f"plane{p}")
                    n_canvases += 1

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
