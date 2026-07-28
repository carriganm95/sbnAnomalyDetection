#!/usr/bin/env python3
"""Overlay hit-finder Gaussian fits on raw ADC waveforms, per channel, for
one event -- a direct visual check of how well hit-finding is performing.

Takes two files describing the SAME event from two different stages of the
processing chain (matched by run/subrun/event, using the structures already
used elsewhere in this repo):

  --raw-file   flat "rawdigits" ntuple from scripts/dump_rawdigits.C
               (sbn_anomaly.data.raw_digit_reader.RawDigitReader: one
               RawEvent per entry with run/subrun/event/channel/pedestal/adc)
  --hits-file  the reco/caloskim ROOT file with hit-finder output
               (same hits0.h/hits1.h/hits2.h.{integral,channel,time,plane,
               width,...} branches SparseWindowDatasetPyG.from_root reads --
               see data/README.md's sparse event format table)

For each channel active in that event (has a hit, or a raw-waveform
deviation clearly above its own noise floor -- see --activity-threshold),
draws the pedestal-subtracted raw waveform with one Gaussian curve per hit
found on that channel, parameterized directly from the hit's own fields:
mean = hit time, sigma = hit width. Amplitude uses a `.amplitude` branch if
your ntuple has one (checked automatically); otherwise it's derived from
the Gaussian-area identity integral = amplitude * width * sqrt(2*pi), i.e.
amplitude = integral / (width * sqrt(2*pi)) -- this assumes `width` is a
Gaussian sigma, which is GausHitFinder's convention but worth checking
against your production's actual hit-finder module.

IMPORTANT CAVEAT -- read before concluding hit-finding looks "bad": the hit
finder almost always fits the DECONVOLVED wire signal (recob::Wire), not
raw ADC counts directly. Raw induction-plane waveforms are bipolar (the
induced-current shape), while the deconvolved signal the fit was performed
on is approximately unipolar/Gaussian -- so a visibly poor-looking overlay
on an INDUCTION channel does not by itself mean the hit finder is
malfunctioning; it may just be an apples-to-oranges signal representation.
The comparison is most directly meaningful on the COLLECTION plane, where
raw and deconvolved shapes are closer to each other. This script draws
exactly what's in --raw-file (pedestal-subtracted, and optionally
coherent-noise-subtracted with --remove-coherent) -- it does not attempt to
deconvolve, since that requires the field/electronics response this repo
doesn't have on hand.

Usage
-----
    python scripts/compare_hits_to_waveform.py \\
        --raw-file good_raw_poc.root \\
        --hits-file /pnfs/sbnd/.../reco/run19305_evt0.root \\
        --run 19305 --subrun 1 --event 42 \\
        --output hit_check_run19305_evt42.root

Restrict to the flagged collection-plane channel range, force every channel
(including inactive ones) to be drawn:
    python scripts/compare_hits_to_waveform.py \\
        --raw-file good_raw_poc.root --hits-file .../run19305_evt0.root \\
        --run 19305 --subrun 1 --event 42 \\
        --channel-range 4000 5500 --all-channels \\
        --output hit_check_collection_evt42.root

Not sure which run/event to pick? List every (run, subrun, event) common to
both files (cheap -- only reads the small scalar run/subrun/event branches,
never the heavy waveform/hit arrays), then exit without drawing anything:
    python scripts/compare_hits_to_waveform.py \\
        --raw-file good_raw_poc.root --hits-file .../run19305_evt0.root \\
        --list-events

If none of --run/--subrun/--event/--event-index are given at all, the first
event common to both files is used automatically (logged, so you know which
one you got) -- handy for a quick first look:
    python scripts/compare_hits_to_waveform.py \\
        --raw-file good_raw_poc.root --hits-file .../run19305_evt0.root \\
        --output hit_check_first_event.root

Output layout: one TCanvas per channel, grouped by plane:
    event_run<r>_subrun<s>_evt<e>/plane<p>/ch<channel>

Requires: PyROOT (drawing/writing TCanvas/TF1) and uproot+awkward (reading
the hits tree via sbn_anomaly.data.materialize_windows._group_hit_prefixes,
same as SparseWindowDatasetPyG.from_root). Run in your full analysis
environment.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sbn_anomaly.data.raw_preprocess import pedestal_subtract, remove_coherent_noise  # noqa: E402

logger = logging.getLogger("compare_hits_to_waveform")


# ---------------------------------------------------------------------------
# Event lookup / hit extraction -- pure Python + numpy/awkward, no PyROOT.
# ---------------------------------------------------------------------------

def list_common_events(
    raw_file: str,
    hits_file: str,
    tree_name: str = "caloskim/TrackCaloSkim",
    raw_tree_name: str = "rawdigits",
    hits_meta_branches=("meta.run", "meta.subrun", "meta.evt"),
):
    """(run, subrun, event) triples present in BOTH files, sorted.

    Reads only the small per-event scalar run/subrun/event branches from
    each file (never the heavy jagged adc/hit arrays), so this is cheap even
    on a large production file -- safe to call just to print a menu.

    Returns (common_sorted, n_only_in_raw, n_only_in_hits).
    """
    import uproot

    with uproot.open(raw_file) as rf:
        if raw_tree_name not in rf:
            raise ValueError(f"Tree '{raw_tree_name}' not found in {raw_file}")
        raw_arrs = rf[raw_tree_name].arrays(["run", "subrun", "event"], library="np")
    raw_events = set(zip(raw_arrs["run"].tolist(), raw_arrs["subrun"].tolist(), raw_arrs["event"].tolist()))

    with uproot.open(hits_file) as hf:
        if tree_name not in hf:
            raise ValueError(f"Tree '{tree_name}' not found in {hits_file}")
        tree = hf[tree_name]
        # recursive=True: meta.run/meta.subrun/meta.evt are nested UNDER a
        # parent branch (e.g. "trk"), not top-level themselves -- a
        # non-recursive tree.keys() misses them entirely even though
        # tree.arrays([...]) can address them fine by their own branch name.
        available = set(tree.keys(recursive=True))
        missing = [b for b in hits_meta_branches if b not in available]
        if missing:
            raise ValueError(
                f"Missing meta branch(es) {missing} in {hits_file} -- run with --list-branches to see "
                f"what '{tree_name}' actually has, then pass the real names via --meta-branches.")
        hit_arrs = tree.arrays(list(hits_meta_branches), library="np")
    hits_events = set(zip(
        hit_arrs[hits_meta_branches[0]].tolist(),
        hit_arrs[hits_meta_branches[1]].tolist(),
        hit_arrs[hits_meta_branches[2]].tolist(),
    ))

    common = sorted(raw_events & hits_events)
    return common, len(raw_events - hits_events), len(hits_events - raw_events)


def list_hits_tree_branches(hits_file: str, tree_name: str) -> None:
    """Print the top-level keys in --hits-file, and every branch (recursive
    -- includes branches nested under a parent branch like "trk", which a
    shallow listing misses) in --tree-name if it's found there.
    """
    import uproot

    with uproot.open(hits_file) as f:
        top = sorted(f.keys(recursive=False))
        print(f"Top-level key(s) in {hits_file}:")
        for k in top:
            print(f"  {k}")

        if tree_name in f:
            names = sorted(f[tree_name].keys(recursive=True))
            print(f"\n{len(names)} branch(es) in tree '{tree_name}' (recursive):")
            for n in names:
                print(f"  {n}")
        else:
            print(f"\nTree '{tree_name}' not found directly under a top-level key. "
                  f"Every nested key (recursive) -- look for the real tree path:")
            for k in sorted(f.keys(recursive=True)):
                print(f"  {k}")


def find_raw_event(raw_file: str, run: Optional[int], subrun: Optional[int], event: Optional[int],
                    event_index: Optional[int] = None):
    """Return the matching RawEvent from --raw-file (by run/subrun/event, or
    by simple entry index if event_index is given instead).
    """
    from sbn_anomaly.data.raw_digit_reader import RawDigitReader

    reader = RawDigitReader(raw_file)
    for i, ev in enumerate(reader):
        if event_index is not None:
            if i == event_index:
                return ev
        elif ev.run == run and ev.subrun == subrun and ev.event == event:
            return ev
    where = f"event_index={event_index}" if event_index is not None else f"run={run} subrun={subrun} event={event}"
    raise ValueError(f"No matching event ({where}) found in {raw_file}")


def compute_hit_amplitude(integral: float, width: float, stored_amplitude: Optional[float]) -> float:
    """Prefer a directly-stored amplitude branch; otherwise derive it from
    the Gaussian area identity (integral = amplitude * width * sqrt(2*pi)),
    GausHitFinder's convention for `width` = sigma.
    """
    if stored_amplitude is not None and stored_amplitude > 0:
        return float(stored_amplitude)
    sigma = width if width > 0 else 1.0
    return float(integral) / (sigma * math.sqrt(2.0 * math.pi))


def load_hits_for_event(
    hits_file: str,
    tree_name: str,
    hit_branches: List[str],
    run: Optional[int],
    subrun: Optional[int],
    event: Optional[int],
    event_index: Optional[int] = None,
    meta_branches=("meta.run", "meta.subrun", "meta.evt"),
) -> Dict[int, list]:
    """Return {channel: [hit dict, ...]} for one entry of --hits-file.

    Each hit dict has time/width/integral/amplitude/plane. Reads the small
    per-event meta branches first (cheap, whole tree) to find the matching
    entry index, then reads only that single entry's (jagged) hit branches
    -- avoids loading the full tree into memory for a one-event lookup.
    """
    import awkward as ak
    import uproot

    from sbn_anomaly.data.materialize_windows import _group_hit_prefixes

    prefixes = _group_hit_prefixes(hit_branches)
    integral_b = [p + ".integral" for p in prefixes]
    channel_b = [p + ".channel" for p in prefixes]
    time_b = [p + ".time" for p in prefixes]
    plane_b = [p + ".plane" for p in prefixes]
    width_b = [p + ".width" for p in prefixes]
    amp_b = [p + ".amplitude" for p in prefixes]

    with uproot.open(hits_file) as f:
        if tree_name not in f:
            raise ValueError(f"Tree '{tree_name}' not found in {hits_file}")
        tree = f[tree_name]
        # recursive=True -- see list_common_events for why (branches nested
        # under a parent like "trk" are invisible to a shallow scan).
        available = set(tree.keys(recursive=True))

        if event_index is not None:
            idx = event_index
        else:
            missing_meta = [b for b in meta_branches if b not in available]
            if missing_meta:
                raise ValueError(
                    f"Missing meta branch(es) {missing_meta} in {hits_file} -- run with "
                    f"--list-branches to see what '{tree_name}' actually has, then pass the real "
                    f"names via --meta-branches (or use --event-index instead)")
            meta = tree.arrays(list(meta_branches), library="np")
            matches = np.where(
                (meta[meta_branches[0]] == run)
                & (meta[meta_branches[1]] == subrun)
                & (meta[meta_branches[2]] == event)
            )[0]
            if matches.size == 0:
                raise ValueError(f"run={run} subrun={subrun} event={event} not found in {hits_file}")
            idx = int(matches[0])

        have_amp = bool(amp_b) and all(b in available for b in amp_b)
        needed = list(set(integral_b + channel_b + time_b + plane_b + width_b + (amp_b if have_amp else [])))
        chunk = tree.arrays(needed, entry_start=idx, entry_stop=idx + 1, library="ak")

        hits_by_channel: Dict[int, list] = {}
        for pfx_idx, (ib, cb, tb, pb, wb) in enumerate(zip(integral_b, channel_b, time_b, plane_b, width_b)):
            try:
                integrals = ak.to_numpy(ak.flatten(chunk[ib][0], axis=None)).astype(np.float64)
                channels = ak.to_numpy(ak.flatten(chunk[cb][0], axis=None)).astype(np.int64)
                times = ak.to_numpy(ak.flatten(chunk[tb][0], axis=None)).astype(np.float64)
                planes = ak.to_numpy(ak.flatten(chunk[pb][0], axis=None)).astype(np.int64)
                widths = ak.to_numpy(ak.flatten(chunk[wb][0], axis=None)).astype(np.float64)
            except Exception:
                continue
            amps = None
            if have_amp:
                amps = ak.to_numpy(ak.flatten(chunk[amp_b[pfx_idx]][0], axis=None)).astype(np.float64)
            m = min(len(integrals), len(channels), len(times), len(planes), len(widths))
            for i in range(m):
                c = int(channels[i])
                if c < 0:
                    continue
                width = float(widths[i])
                stored_amp = float(amps[i]) if amps is not None and i < len(amps) else None
                hits_by_channel.setdefault(c, []).append(dict(
                    time=float(times[i]), width=width, integral=float(integrals[i]),
                    amplitude=compute_hit_amplitude(float(integrals[i]), width, stored_amp),
                    plane=int(planes[i]),
                ))
        return hits_by_channel


def robust_noise_std(waveform: np.ndarray) -> float:
    """Median-absolute-deviation noise estimate -- unlike plain std, isn't
    inflated by the pulses themselves, so it's a fair 'is this channel
    active' baseline even on a channel with a real signal in it.
    """
    med = np.median(waveform)
    mad = np.median(np.abs(waveform - med))
    return float(1.4826 * mad)


def is_active(waveform: np.ndarray, hits: list, nsigma: float) -> bool:
    if hits:
        return True
    noise = robust_noise_std(waveform)
    if noise <= 0:
        return bool(np.max(np.abs(waveform)) > 0)
    return bool(np.max(np.abs(waveform)) > nsigma * noise)


def preprocess_waveform(adc_row: np.ndarray, pedestal: float, remove_coherent: bool,
                         group_waveforms: Optional[np.ndarray], coherent_group_size: int) -> np.ndarray:
    """Pedestal-subtract one channel's waveform; coherent-noise removal (if
    requested) needs the OTHER channels in the same event, so it's done
    once for the whole event upstream and this just extracts one row --
    see build_channel_waveforms.
    """
    if group_waveforms is not None:
        return group_waveforms
    return pedestal_subtract(adc_row.reshape(1, -1), np.array([pedestal], dtype=np.float32))[0]


def build_all_waveforms(raw_event, remove_coherent: bool, coherent_group_size: int) -> np.ndarray:
    """Pedestal-subtract (and optionally coherent-noise-subtract) the WHOLE
    event's (n_channels, n_ticks) ADC matrix at once -- coherent-noise
    removal needs all channels in a group together, not one at a time.
    """
    wf = pedestal_subtract(raw_event.adc, raw_event.pedestal)
    if remove_coherent:
        wf = remove_coherent_noise(wf, group_size=coherent_group_size)
    return wf


# ---------------------------------------------------------------------------
# PyROOT drawing -- NOT exercised in this environment (no ROOT/PyROOT
# available here). The extraction/matching logic above is unit-tested
# independently with synthetic/duck-typed inputs.
# ---------------------------------------------------------------------------

def draw_channel_canvas(ROOT, channel: int, plane: int, waveform: np.ndarray, hits: list, run, subrun, event):
    n = len(waveform)
    title = f"Channel {channel} (plane {plane}) -- run {run} subrun {subrun} evt {event}"
    h = ROOT.TH1D(f"wf_ch{channel}", f"{title};tick;ADC (pedestal-subtracted)", n, 0, n)
    for i, v in enumerate(waveform):
        h.SetBinContent(i + 1, float(v))
    h.SetLineColor(ROOT.kBlack)
    h.SetLineWidth(1)

    c = ROOT.TCanvas(f"c_ch{channel}", f"ch{channel}", 1000, 500)
    h.Draw("HIST")

    colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen + 2, ROOT.kMagenta + 1, ROOT.kOrange + 7, ROOT.kCyan + 2]
    funcs = []
    legend = ROOT.TLegend(0.70, 0.60, 0.89, 0.89)
    legend.AddEntry(h, "raw waveform", "l")
    for i, hit in enumerate(sorted(hits, key=lambda x: x["time"])):
        lo = max(0.0, hit["time"] - 6 * max(hit["width"], 1e-3))
        hi = min(float(n), hit["time"] + 6 * max(hit["width"], 1e-3))
        fname = f"hit_{channel}_{i}"
        fn = ROOT.TF1(fname, "gaus", lo, hi)
        fn.SetParameters(hit["amplitude"], hit["time"], max(hit["width"], 1e-3))
        fn.SetLineColor(colors[i % len(colors)])
        fn.SetLineWidth(2)
        fn.Draw("SAME")
        funcs.append(fn)
        legend.AddEntry(fn, f"hit {i}: t={hit['time']:.1f} w={hit['width']:.2f} q={hit['integral']:.0f}", "l")
    legend.Draw()
    c.Update()
    c._keepalive = (h, funcs, legend)
    return c


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


# ---------------------------------------------------------------------------

def run(
    raw_file: str,
    hits_file: str,
    output: str,
    run_: Optional[int],
    subrun: Optional[int],
    event: Optional[int],
    event_index: Optional[int],
    tree_name: str,
    hit_branches: List[str],
    remove_coherent: bool,
    coherent_group_size: int,
    activity_nsigma: float,
    all_channels: bool,
    channel_range: Optional[tuple],
    meta_branches=("meta.run", "meta.subrun", "meta.evt"),
) -> None:
    import ROOT
    ROOT.gROOT.SetBatch(True)

    logger.info("Looking up event in %s ...", raw_file)
    raw_event = find_raw_event(raw_file, run_, subrun, event, event_index)
    logger.info("Found run=%d subrun=%d event=%d, %d channels, %d ticks",
                raw_event.run, raw_event.subrun, raw_event.event, raw_event.n_channels, raw_event.n_ticks)

    logger.info("Looking up matching hits in %s ...", hits_file)
    hits_by_channel = load_hits_for_event(
        hits_file, tree_name, hit_branches, raw_event.run, raw_event.subrun, raw_event.event, event_index,
        meta_branches=meta_branches)
    total_hits = sum(len(v) for v in hits_by_channel.values())
    logger.info("Found %d hits across %d channels for this event", total_hits, len(hits_by_channel))

    logger.info("Preprocessing waveforms (remove_coherent=%s) ...", remove_coherent)
    all_wf = build_all_waveforms(raw_event, remove_coherent, coherent_group_size)

    lo, hi = channel_range if channel_range else (-1, float("inf"))
    outfile = ROOT.TFile(output, "RECREATE")
    base_dir = f"event_run{raw_event.run}_subrun{raw_event.subrun}_evt{raw_event.event}"
    n_drawn = 0
    for ci, channel in enumerate(raw_event.channel):
        channel = int(channel)
        if not (lo <= channel < hi):
            continue
        waveform = all_wf[ci]
        hits = hits_by_channel.get(channel, [])
        if not all_channels and not is_active(waveform, hits, activity_nsigma):
            continue
        plane = hits[0]["plane"] if hits else -1
        d = _get_or_make_dir(outfile, f"{base_dir}/plane{plane}")
        d.cd()
        c = draw_channel_canvas(ROOT, channel, plane, waveform, hits,
                                 raw_event.run, raw_event.subrun, raw_event.event)
        c.Write(f"ch{channel:05d}")
        n_drawn += 1
        if n_drawn % 100 == 0:
            logger.info("  drawn %d channels so far", n_drawn)

    outfile.Close()
    logger.info("Wrote %d channel canvases to %s", n_drawn, output)
    if not all_channels:
        logger.info(
            "Only channels with a hit or raw activity > %.1f sigma above their own noise floor were "
            "drawn -- pass --all-channels to force every channel.", activity_nsigma)


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-file", required=True, help="Flat rawdigits.root (scripts/dump_rawdigits.C output)")
    ap.add_argument("--hits-file", required=True, help="reco/caloskim ROOT file with hit-finder output")
    ap.add_argument("--run", type=int, default=None)
    ap.add_argument("--subrun", type=int, default=None)
    ap.add_argument("--event", type=int, default=None)
    ap.add_argument("--event-index", type=int, default=None,
                     help="Alternative to --run/--subrun/--event: match by entry order in both files "
                          "instead (only valid if both files were produced with the same event ordering)")
    ap.add_argument("--list-events", action="store_true",
                     help="Print every (run, subrun, event) common to --raw-file and --hits-file, then exit "
                          "without drawing anything -- use this to pick --run/--subrun/--event")
    ap.add_argument("--list-branches", action="store_true",
                     help="Print every branch (recursive) in --tree-name of --hits-file, then exit -- "
                          "use this if --tree-name/--meta-branches don't match what's actually in the file")
    ap.add_argument("--meta-branches", nargs=3, default=["meta.run", "meta.subrun", "meta.evt"],
                     metavar=("RUN_BRANCH", "SUBRUN_BRANCH", "EVENT_BRANCH"),
                     help="Per-event provenance branch names in --hits-file, in run/subrun/event order "
                          "(default matches SBND caloskim: meta.run meta.subrun meta.evt)")
    ap.add_argument("--output", default=None, help="Required unless --list-events/--list-branches is given")
    ap.add_argument("--tree-name", default="caloskim/TrackCaloSkim")
    ap.add_argument("--hit-branches", nargs="+",
                     default=["hits0.h.integral", "hits0.h.channel",
                              "hits1.h.integral", "hits1.h.channel",
                              "hits2.h.integral", "hits2.h.channel"],
                     help="Used only to discover the hits0.h/hits1.h/hits2.h prefixes (default matches configs/graph_vae.yaml)")
    ap.add_argument("--remove-coherent", action="store_true",
                     help="Also subtract common-mode/coherent noise per electronics group (default: pedestal-subtract only)")
    ap.add_argument("--coherent-group-size", type=int, default=64)
    ap.add_argument("--activity-threshold", type=float, default=5.0, dest="activity_nsigma",
                     help="Draw a channel with no hits only if its raw waveform deviates more than this "
                          "many sigma above its own robust noise floor (default 5). Ignored with --all-channels")
    ap.add_argument("--all-channels", action="store_true", help="Draw every channel, including inactive ones")
    ap.add_argument("--channel-range", nargs=2, type=int, default=None, metavar=("LO", "HI"))
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.list_branches:
        list_hits_tree_branches(args.hits_file, args.tree_name)
        return

    meta_branches = tuple(args.meta_branches)

    if args.list_events:
        common, n_only_raw, n_only_hits = list_common_events(
            args.raw_file, args.hits_file, args.tree_name, hits_meta_branches=meta_branches)
        print(f"{len(common)} event(s) present in BOTH files "
              f"({n_only_raw} only in --raw-file, {n_only_hits} only in --hits-file):")
        for i, (r, s, e) in enumerate(common):
            print(f"  [{i}] run={r} subrun={s} event={e}   "
                  f"(--run {r} --subrun {s} --event {e}   or   --event-index {i})")
        return

    have_explicit = args.run is not None and args.subrun is not None and args.event is not None
    have_partial = any(v is not None for v in (args.run, args.subrun, args.event)) and not have_explicit
    if have_partial:
        raise SystemExit("Provide all of --run/--subrun/--event together (or none, to auto-pick the "
                          "first common event), or use --event-index instead")
    if args.event_index is not None and have_explicit:
        raise SystemExit("Provide --event-index OR --run/--subrun/--event, not both")

    run_, subrun, event, event_index = args.run, args.subrun, args.event, args.event_index
    if not have_explicit and event_index is None:
        logger.info("No --run/--subrun/--event or --event-index given -- looking up the first "
                    "event common to both files ...")
        common, _, _ = list_common_events(
            args.raw_file, args.hits_file, args.tree_name, hits_meta_branches=meta_branches)
        if not common:
            raise SystemExit(f"No events are common to {args.raw_file} and {args.hits_file} -- "
                              f"nothing to draw. Re-run with --list-events to inspect both files.")
        run_, subrun, event = common[0]
        logger.info(
            "Using the first common event: run=%d subrun=%d event=%d (%d common event(s) total -- "
            "pass --run/--subrun/--event, --event-index, or --list-events to pick a different one)",
            run_, subrun, event, len(common))

    if not args.output:
        raise SystemExit("--output is required (unless --list-events/--list-branches is given)")

    run(
        raw_file=args.raw_file, hits_file=args.hits_file, output=args.output,
        run_=run_, subrun=subrun, event=event, event_index=event_index,
        tree_name=args.tree_name, hit_branches=args.hit_branches,
        remove_coherent=args.remove_coherent, coherent_group_size=args.coherent_group_size,
        meta_branches=meta_branches,
        activity_nsigma=args.activity_nsigma, all_channels=args.all_channels,
        channel_range=tuple(args.channel_range) if args.channel_range else None,
    )


if __name__ == "__main__":
    main()
