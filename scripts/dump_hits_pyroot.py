#!/usr/bin/env python3
"""Dump a recob::Hit collection straight from an art Events tree into the
same sparse CSR .npz format SparseWindowDatasetPyG.save_events() writes
(see data/README.md's "Sparse event format" table) -- e.g. for a branch
like:

    *Br  375 :recob::Hits_fasthit__Reco1.obj : vector<recob::Hit>          *

Why this exists: caloskim/TrackCaloSkim (what compare_hits_to_waveform.py
and the rest of this repo's hit branches normally read) only stores hits
that got associated with a reconstructed track, and stores one tree entry
per TRACK rather than per event (see the fix in
scripts/compare_hits_to_waveform.py's load_hits_for_event). That silently
drops hits that never got attached to a track. Reading the hit collection
directly from Events gives every hit the hit-finder produced for that
event, independent of downstream tracking -- a fair "all hits" baseline to
compare against.

Like scripts/dump_rawdigits_pyroot.py, this uses PyROOT's EMULATED-CLASS
reading: art files embed their StreamerInfo, so ROOT can build an emulated
recob::Hit at runtime and expose its data members (fChannel, fPeakTime,
fRMS, fPeakAmplitude, fIntegral, fMultiplicity, fWireID, fHitSummedADC,
...) WITHOUT the real lardataobj/LArSoft dictionaries. Every accessor below
tries the real C++ method first (Channel(), PeakTime(), RMS(), ...) and
falls back to the raw data member if the method isn't callable in this
emulated environment -- same pattern as dump_rawdigits_pyroot.py.

Field mapping (recob::Hit -> npz key):
    channel                          -> channels_flat
    Integral()                       -> integrals_flat
    PeakTime()                       -> times_flat
    RMS()                            -> widths_flat       (Gaussian sigma,
                                         same "width" convention used
                                         elsewhere in this repo)
    SummedADC() (falls back to
      ROISummedADC() if unavailable) -> sumadcs_flat
    Multiplicity()                   -> mults_flat
    WireID().Wire / .Plane / .TPC    -> wires_flat / planes_flat / tpcs_flat

NOTE -- hassps_flat (hit-has-matched-3D-spacepoint) is NOT derivable from
the Hit collection alone; it requires the recob::Hit <-> recob::SpacePoint
art Assn, which this script does not read. It is zero-filled in the
output -- the same fallback SparseWindowDatasetPyG already uses for
productions/files missing this optional array (see data/README.md), so
any node_features entry that needs sp_fraction will just see 0 everywhere
rather than failing.

Requires only ROOT (no gallery/LArSoft needed). Run it in your pixi/ROOT
environment:

    python scripts/dump_hits_pyroot.py \
        /pnfs/.../reco/run19305_evt0.root \
        --output data/hits_run19305_evt0.npz

If the default --tag-contains doesn't match your file's branch name, find
the right one first:

    python scripts/dump_hits_pyroot.py in.root --output out.npz --list-branches

Dump multiple files and combine them the same way dump_rawdigits_pyroot.py
output files are combined -- one npz per input file here, then:

    python -m sbn_anomaly.data.merge_events --output combined.npz \
        --glob 'data/hits_*.npz'
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def _all_branch_names(branch_holder, _depth: int = 0, _max_depth: int = 6) -> list[str]:
    """Recursively collect every branch name reachable from `branch_holder`
    (a TTree or a TBranch), not just the top-level listing -- see the
    identical helper (and its docstring) in dump_rawdigits_pyroot.py for
    why a shallow GetListOfBranches() scan isn't enough for a split/nested
    object's sub-branches.
    """
    names: list[str] = []
    blist = branch_holder.GetListOfBranches()
    if not blist:
        return names
    for b in blist:
        n = b.GetName()
        names.append(n)
        if _depth < _max_depth:
            names.extend(_all_branch_names(b, _depth + 1, _max_depth))
    return names


def _find_hits_branch(tree, tag_contains: str = "recob::Hits_fasthit__Reco1",
                       list_all: bool = False) -> str | None:
    """Find the branch holding the recob::Hit vector, matched against
    `tag_contains`. Same recursive-scan + GetEntry(0)-first approach as
    dump_rawdigits_pyroot.py's _find_rawdigit_branch, and the same
    Assns/TimeStamp exclusion (an Assns branch derived from the same
    product tag would otherwise also match the substring).
    """
    if tree.GetEntries() > 0:
        tree.GetEntry(0)  # force full branch-structure resolution before scanning

    all_names = _all_branch_names(tree)
    if list_all:
        print(f"# {len(all_names)} branch name(s) found (recursive scan):")
        for n in sorted(set(all_names)):
            print(f"#   {n}")

    candidates = [n for n in all_names if tag_contains in n and "Assns" not in n and "TimeStamp" not in n]
    top_level = {b.GetName() for b in tree.GetListOfBranches()}
    top_level_candidates = [c for c in candidates if c in top_level]
    if top_level_candidates:
        candidates = top_level_candidates

    candidates = sorted(set(candidates))
    if not candidates:
        return None
    if len(candidates) > 1:
        print(f"# WARNING: {len(candidates)} branches matched --tag-contains={tag_contains!r}; "
              f"using the first. Pass a more specific --tag-contains to disambiguate:")
        for c in candidates:
            print(f"#   {c}")
    return candidates[0]


def _get_value(tree, base: str):
    """Return the std::vector<recob::Hit> for the current entry."""
    obj_name = base + "obj" if base.endswith(".") else base + ".obj"
    for name in (obj_name, base, base.rstrip(".")):
        try:
            v = getattr(tree, name)
            if v is not None:
                return v
        except Exception:
            continue
    return None


def _call_or_member(obj, *names):
    """Try each name in order: call it if it's a method, else read it as a
    data member. Returns None if nothing worked.
    """
    for name in names:
        if not hasattr(obj, name):
            continue
        try:
            v = getattr(obj, name)
            return v() if callable(v) else v
        except Exception:
            continue
    return None


def _hit_channel(h) -> int:
    v = _call_or_member(h, "Channel", "fChannel")
    return int(v) if v is not None else -1


def _hit_integral(h) -> float:
    v = _call_or_member(h, "Integral", "fIntegral")
    return float(v) if v is not None else 0.0


def _hit_peaktime(h) -> float:
    v = _call_or_member(h, "PeakTime", "fPeakTime")
    return float(v) if v is not None else 0.0


def _hit_rms(h) -> float:
    v = _call_or_member(h, "RMS", "fRMS")
    return float(v) if v is not None else 0.0


def _hit_amplitude(h) -> float:
    v = _call_or_member(h, "PeakAmplitude", "fPeakAmplitude")
    return float(v) if v is not None else 0.0


def _hit_summedadc(h) -> float:
    # SummedADC() ("hit summed ADC", raw ADC under the hit's own fit window)
    # is what data/README.md's sumadcs_flat means; ROISummedADC() (raw ADC
    # under the whole ROI, a superset) is used only as a last-resort
    # fallback if a production didn't store the narrower quantity.
    v = _call_or_member(h, "SummedADC", "fHitSummedADC")
    if v is None:
        v = _call_or_member(h, "ROISummedADC", "fROISummedADC")
    return float(v) if v is not None else 0.0


def _hit_multiplicity(h) -> float:
    v = _call_or_member(h, "Multiplicity", "fMultiplicity")
    return float(v) if v is not None else 0.0


def _hit_wireid(h) -> tuple[int, int, int]:
    """Return (tpc, plane, wire) from the hit's geo::WireID.

    geo::WireID (and its PlaneID/TPCID/CryostatID base classes) are simple
    ID structs with PUBLIC data members named Cryostat/TPC/Plane/Wire
    directly (no leading 'f', no accessor methods) in current LArSoft --
    unlike recob::Hit's own private+accessor fields. _call_or_member still
    covers both naming conventions defensively in case an older/emulated
    layout differs.
    """
    wid = _call_or_member(h, "WireID")
    if wid is None:
        wid = getattr(h, "fWireID", None)
    if wid is None:
        return 0, 0, 0
    tpc = _call_or_member(wid, "TPC", "fTPC")
    plane = _call_or_member(wid, "Plane", "fPlane")
    wire = _call_or_member(wid, "Wire", "fWire")
    return (
        int(tpc) if tpc is not None else 0,
        int(plane) if plane is not None else 0,
        int(wire) if wire is not None else 0,
    )


def _event_ids(tree, entry: int):
    """Best-effort (run, subrun, event); falls back to (0, 0, entry) --
    identical to dump_rawdigits_pyroot.py's helper.
    """
    try:
        aux = tree.EventAuxiliary
        return int(aux.run()), int(aux.subRun()), int(aux.event())
    except Exception:
        try:
            eid = tree.EventAuxiliary.id()
            return int(eid.run()), int(eid.subRun()), int(eid.event())
        except Exception:
            return 0, 0, int(entry)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="PyROOT recob::Hit -> sparse-events npz dumper")
    p.add_argument("input", help="art ROOT file (reco)")
    p.add_argument("--output", default=None,
                   help="output sparse-events .npz (data/README.md format). Required unless --list-branches")
    p.add_argument("--nevents", type=int, default=0, help="0 = all events")
    p.add_argument("--tree-name", default="Events")
    p.add_argument("--tag-contains", default="recob::Hits_fasthit__Reco1",
                   help="substring identifying the recob::Hit product branch")
    p.add_argument("--list-branches", action="store_true",
                   help="Print every branch name found (recursive scan) and exit -- use this to find "
                        "the right --tag-contains value if the expected branch isn't being found")
    args = p.parse_args(argv)

    import ROOT

    ROOT.gErrorIgnoreLevel = ROOT.kError  # silence emulated-class info spam

    f = ROOT.TFile.Open(args.input)
    if not f or f.IsZombie():
        print(f"Could not open {args.input}", file=sys.stderr)
        return 1
    tree = f.Get(args.tree_name)
    if not tree:
        print(f"No '{args.tree_name}' tree", file=sys.stderr)
        return 1

    base = _find_hits_branch(tree, args.tag_contains, list_all=args.list_branches)
    if args.list_branches:
        return 0
    if base is None:
        print(f"No branch matching --tag-contains={args.tag_contains!r} found "
              f"(excluding Assns/TimeStamp companions). Re-run with --list-branches "
              f"to see every branch name and pick the right substring.", file=sys.stderr)
        return 1
    if not args.output:
        print("--output is required (unless --list-branches is given)", file=sys.stderr)
        return 1
    print(f"# recob::Hit branch: {base}")

    tree.SetBranchStatus("*", 0)
    for keep in (base + "*", "EventAuxiliary*"):
        tree.SetBranchStatus(keep, 1)

    n_entries = tree.GetEntries()
    to_do = n_entries if args.nevents == 0 else min(args.nevents, n_entries)
    print(f"# entries in file: {n_entries}; processing {to_do}")

    channel_parts: list[np.ndarray] = []
    integral_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    width_parts: list[np.ndarray] = []
    sumadc_parts: list[np.ndarray] = []
    mult_parts: list[np.ndarray] = []
    plane_parts: list[np.ndarray] = []
    wire_parts: list[np.ndarray] = []
    tpc_parts: list[np.ndarray] = []
    sizes: list[int] = []
    evt_run: list[int] = []
    evt_subrun: list[int] = []
    evt_num: list[int] = []

    total_hits = 0
    for i in range(to_do):
        tree.GetEntry(i)
        r, s, e = _event_ids(tree, i)
        evt_run.append(r)
        evt_subrun.append(s)
        evt_num.append(e)

        hits = _get_value(tree, base)
        n = 0 if hits is None else (int(hits.size()) if hasattr(hits, "size") else len(hits))
        if n == 0:
            sizes.append(0)
            if (i + 1) % 50 == 0 or i + 1 == to_do:
                print(f"  ... {i + 1}/{to_do} events ({total_hits} hits so far)")
            continue

        ch = np.empty(n, dtype=np.int64)
        integ = np.empty(n, dtype=np.float32)
        t = np.empty(n, dtype=np.float32)
        w = np.empty(n, dtype=np.float32)
        sadc = np.empty(n, dtype=np.float32)
        mul = np.empty(n, dtype=np.float32)
        pl = np.empty(n, dtype=np.int32)
        wr = np.empty(n, dtype=np.int32)
        tp = np.empty(n, dtype=np.int32)

        for k in range(n):
            hit = hits[k]
            ch[k] = _hit_channel(hit)
            integ[k] = _hit_integral(hit)
            t[k] = _hit_peaktime(hit)
            w[k] = _hit_rms(hit)
            sadc[k] = _hit_summedadc(hit)
            mul[k] = _hit_multiplicity(hit)
            tpc_, plane_, wire_ = _hit_wireid(hit)
            tp[k] = tpc_
            pl[k] = plane_
            wr[k] = wire_

        valid = ch >= 0
        if not valid.all():
            ch, integ, t, w, sadc, mul, pl, wr, tp = (
                ch[valid], integ[valid], t[valid], w[valid], sadc[valid],
                mul[valid], pl[valid], wr[valid], tp[valid],
            )

        channel_parts.append(ch)
        integral_parts.append(integ)
        time_parts.append(t)
        width_parts.append(w)
        sumadc_parts.append(sadc)
        mult_parts.append(mul)
        plane_parts.append(pl)
        wire_parts.append(wr)
        tpc_parts.append(tp)
        sizes.append(int(ch.size))
        total_hits += int(ch.size)

        if i == 0:
            print(f"# first event: run={r} subrun={s} event={e} n_hits={n} "
                  f"first_channel={int(ch[0]) if ch.size else 'n/a'}")
        if (i + 1) % 50 == 0 or i + 1 == to_do:
            print(f"  ... {i + 1}/{to_do} events ({total_hits} hits so far)")

    channels_flat = np.concatenate(channel_parts) if channel_parts else np.empty(0, dtype=np.int64)
    integrals_flat = np.concatenate(integral_parts) if integral_parts else np.empty(0, dtype=np.float32)
    times_flat = np.concatenate(time_parts) if time_parts else np.empty(0, dtype=np.float32)
    widths_flat = np.concatenate(width_parts) if width_parts else np.empty(0, dtype=np.float32)
    sumadcs_flat = np.concatenate(sumadc_parts) if sumadc_parts else np.empty(0, dtype=np.float32)
    mults_flat = np.concatenate(mult_parts) if mult_parts else np.empty(0, dtype=np.float32)
    planes_flat = np.concatenate(plane_parts) if plane_parts else np.empty(0, dtype=np.int32)
    wires_flat = np.concatenate(wire_parts) if wire_parts else np.empty(0, dtype=np.int32)
    tpcs_flat = np.concatenate(tpc_parts) if tpc_parts else np.empty(0, dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    n_channels = int(channels_flat.max()) + 1 if channels_flat.size else 0
    # hasSP requires the Hit<->SpacePoint Assn, not read here -- zero-fill
    # (same convention SparseWindowDatasetPyG already uses when a
    # production/file is missing this optional array; see data/README.md).
    hassps_flat = np.zeros_like(integrals_flat, dtype=np.float32)

    np.savez_compressed(
        args.output,
        channels_flat=channels_flat,
        integrals_flat=integrals_flat,
        offsets=offsets,
        n_channels=np.array(n_channels, dtype=np.int64),
        times_flat=times_flat,
        wires_flat=wires_flat,
        planes_flat=planes_flat,
        tpcs_flat=tpcs_flat,
        widths_flat=widths_flat,
        sumadcs_flat=sumadcs_flat,
        mults_flat=mults_flat,
        hassps_flat=hassps_flat,
        evt_run=np.array(evt_run, dtype=np.int32),
        evt_subrun=np.array(evt_subrun, dtype=np.int32),
        evt_num=np.array(evt_num, dtype=np.int32),
        filenames=np.array([str(args.input)], dtype="U512"),
    )
    print(f"# Wrote {to_do} event(s), {total_hits} hit(s), n_channels={n_channels} to {args.output}")
    print("# NOTE: hassps_flat is zero-filled (hasSP needs the Hit<->SpacePoint Assn, not read by this script)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
