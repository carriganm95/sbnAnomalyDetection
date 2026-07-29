#!/usr/bin/env python3
"""Dump raw::RawDigit ADC waveforms to a flat ntuple using PyROOT (no gallery).

uproot cannot read ``std::vector<raw::RawDigit>`` (memberwise serialization is
unsupported), but ROOT can: art files embed their StreamerInfo, so ROOT builds
*emulated* RawDigit objects at runtime and exposes their data members
(``fADC``, ``fChannel``, ``fPedestal``, ``fCompression``) WITHOUT needing the
LArSoft/lardataobj libraries. This script uses that to write the same flat
``rawdigits`` TTree that ``scripts/dump_rawdigits.C`` would, so the rest of the
Python pipeline (RawDigitReader -> build_raw_waveforms / build_raw_latents) is
unchanged.

Requires only ROOT (already in your pixi env). Run it there:

    python scripts/dump_rawdigits_pyroot.py \
        ../data/raw_decoded_reco_094.root ../data/raw_run094.root --nevents 0

Output tree 'rawdigits' branches (match dump_rawdigits.C):
    run, subrun, event, nchan, nticks : int
    channel[nchan]   : int
    pedestal[nchan]  : float
    adc[nchan*nticks]: short   (channel-major)

IMPORTANT: this reads the *stored* fADC. If the digits are compressed
(fCompression != 0) the stored bytes are NOT the waveform and this script will
stop with a clear message -- in that case the gallery/LArSoft dump (which can
uncompress) is required. Decoded data (process 'DECODE') is normally
uncompressed, so this should work.
"""

from __future__ import annotations

import argparse
import array
import sys


def _all_branch_names(branch_holder, _depth: int = 0, _max_depth: int = 6) -> list[str]:
    """Recursively collect every branch name reachable from `branch_holder`
    (a TTree or a TBranch), not just the top-level listing.

    TTree.GetListOfBranches() only returns TOP-LEVEL branches. For a split
    object (or an art product with sub-branches materialized per data
    member), the members only show up via THAT branch's own
    GetListOfBranches() -- exactly what TTree::Print()/Show() walk
    recursively to produce their (longer) output. A single flat pass over
    the tree's top-level list misses anything nested, which is the likely
    reason a branch visible in a plain print was invisible here.
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


def _find_rawdigit_branch(tree, tag_contains: str = "RawDigits_daq", list_all: bool = False) -> str | None:
    """Find the branch identifying the RawDigit product, matched against
    `tag_contains` (the CLI --tag-contains value -- previously hardcoded and
    ignored here, now actually used).

    Branch discovery also forces the tree to resolve its full split-branch
    structure first (GetEntry(0)) and searches recursively (see
    _all_branch_names), not just the shallow top-level list -- either lazy
    branch materialization or nested sub-branches of a split object can
    otherwise hide the real branch from a naive GetListOfBranches() scan
    even though it's visible in tree.Print()/Show().
    """
    if tree.GetEntries() > 0:
        tree.GetEntry(0)  # force full branch-structure resolution before scanning

    all_names = _all_branch_names(tree)
    if list_all:
        print(f"# {len(all_names)} branch name(s) found (recursive scan):")
        for n in sorted(set(all_names)):
            print(f"#   {n}")

    candidates = [n for n in all_names if tag_contains in n and "Assns" not in n and "TimeStamp" not in n]
    # Prefer an actual top-level product-wrapper branch over any nested
    # sub-branch of the same product that happens to also match the substring.
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
    """Return the std::vector<raw::RawDigit> for the current entry."""
    obj_name = base + "obj" if base.endswith(".") else base + ".obj"
    for name in (obj_name, base, base.rstrip(".")):
        try:
            v = getattr(tree, name)
            if v is not None:
                return v
        except Exception:
            continue
    return None


def _digit_channel(rd):
    for getter in ("Channel",):
        if hasattr(rd, getter):
            try:
                return int(getattr(rd, getter)())
            except Exception:
                pass
    return int(getattr(rd, "fChannel"))


def _digit_pedestal(rd):
    if hasattr(rd, "GetPedestal"):
        try:
            return float(rd.GetPedestal())
        except Exception:
            pass
    return float(getattr(rd, "fPedestal"))


def _digit_compression(rd):
    try:
        return int(getattr(rd, "fCompression"))
    except Exception:
        try:
            return int(rd.Compression())
        except Exception:
            return 0


def _digit_adc(rd, np):
    """Return the ADC waveform as a numpy int16 array (stored, uncompressed)."""
    # Prefer the real accessor if a dictionary happens to be loaded.
    if hasattr(rd, "Samples") and hasattr(rd, "ADC"):
        try:
            n = int(rd.Samples())
            return np.fromiter((rd.ADC(i) for i in range(n)), dtype=np.int16, count=n)
        except Exception:
            pass
    # Emulated path: read the fADC data member vector directly.
    vec = getattr(rd, "fADC", None)
    if vec is None:
        vec = getattr(rd, "fADCs", None)
    if vec is None:
        raise RuntimeError("no fADC member on RawDigit")
    try:
        return np.asarray(vec, dtype=np.int16)
    except Exception:
        n = vec.size()
        return np.fromiter((vec[i] for i in range(n)), dtype=np.int16, count=n)


def _event_ids(tree, entry: int):
    """Best-effort (run, subrun, event); falls back to (0, 0, entry)."""
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
    p = argparse.ArgumentParser(description="PyROOT raw::RawDigit -> flat ntuple dumper")
    p.add_argument("input", help="art ROOT file (raw/decoded)")
    p.add_argument("output", help="output flat ntuple .root (tree 'rawdigits')")
    p.add_argument("--nevents", type=int, default=0, help="0 = all events")
    p.add_argument("--tag-contains", default="RawDigits_daq",
                   help="substring identifying the RawDigit product branch")
    p.add_argument("--list-branches", action="store_true",
                   help="Print every branch name found (recursive scan) before searching -- "
                        "use this to find the right --tag-contains value if the expected "
                        "branch isn't being found")
    args = p.parse_args(argv)

    import numpy as np
    import ROOT

    ROOT.gErrorIgnoreLevel = ROOT.kError  # silence emulated-class info spam

    f = ROOT.TFile.Open(args.input)
    if not f or f.IsZombie():
        print(f"Could not open {args.input}", file=sys.stderr)
        return 1
    tree = f.Get("Events")
    if not tree:
        print("No 'Events' tree", file=sys.stderr)
        return 1

    base = _find_rawdigit_branch(tree, args.tag_contains, list_all=args.list_branches)
    if base is None:
        print(f"No branch matching --tag-contains={args.tag_contains!r} found "
              f"(excluding Assns/TimeStamp companions). Re-run with --list-branches "
              f"to see every branch name and pick the right substring.", file=sys.stderr)
        return 1
    print(f"# RawDigit branch: {base}")

    # Only read what we need (faster).
    tree.SetBranchStatus("*", 0)
    for keep in (base + "*", "EventAuxiliary*"):
        tree.SetBranchStatus(keep, 1)

    # ---- output tree ----
    out = ROOT.TFile(args.output, "RECREATE")
    otree = ROOT.TTree("rawdigits", "Flat raw ADC dump (PyROOT)")
    run = array.array("i", [0]); subrun = array.array("i", [0]); event = array.array("i", [0])
    nchan = array.array("i", [0]); nticks = array.array("i", [0])
    v_chan = ROOT.std.vector("int")()
    v_ped = ROOT.std.vector("float")()
    v_adc = ROOT.std.vector("short")()
    otree.Branch("run", run, "run/I")
    otree.Branch("subrun", subrun, "subrun/I")
    otree.Branch("event", event, "event/I")
    otree.Branch("nchan", nchan, "nchan/I")
    otree.Branch("nticks", nticks, "nticks/I")
    otree.Branch("channel", v_chan)
    otree.Branch("pedestal", v_ped)
    otree.Branch("adc", v_adc)

    n_entries = tree.GetEntries()
    to_do = n_entries if args.nevents == 0 else min(args.nevents, n_entries)
    print(f"# entries in file: {n_entries}; processing {to_do}")

    checked_compression = False
    for i in range(to_do):
        tree.GetEntry(i)
        digits = _get_value(tree, base)
        if digits is None:
            print(f"  entry {i}: could not read RawDigit vector; skipping")
            continue
        ndig = int(digits.size()) if hasattr(digits, "size") else len(digits)
        if ndig == 0:
            continue

        # Fix nticks from the first digit; sanity/compression check once.
        first_adc = _digit_adc(digits[0], np)
        nt = int(first_adc.shape[0])
        if not checked_compression:
            comp0 = _digit_compression(digits[0])
            print(f"# first digit: channel={_digit_channel(digits[0])} "
                  f"nticks={nt} fCompression={comp0} "
                  f"first8={list(first_adc[:8])}")
            if comp0 != 0 and not (hasattr(digits[0], "ADC") and hasattr(digits[0], "Samples")):
                print("\nERROR: digits are compressed (fCompression != 0) and no "
                      "uncompress accessor is available via emulation.\n"
                      "       The stored fADC is NOT the waveform. Use the "
                      "gallery/LArSoft dump (dump_rawdigits.C) on a gpvm instead.",
                      file=sys.stderr)
                out.Close()
                return 2
            checked_compression = True

        r, s, e = _event_ids(tree, i)
        run[0], subrun[0], event[0] = r, s, e
        nchan[0], nticks[0] = ndig, nt

        v_chan.clear(); v_ped.clear(); v_adc.clear()
        v_chan.reserve(ndig); v_ped.reserve(ndig); v_adc.reserve(ndig * nt)
        for d in range(ndig):
            rd = digits[d]
            v_chan.push_back(_digit_channel(rd))
            v_ped.push_back(_digit_pedestal(rd))
            adc = _digit_adc(rd, np)
            # pad/truncate ragged channels to nt so the flat array stays rectangular
            if adc.shape[0] != nt:
                fixed = np.full(nt, int(round(_digit_pedestal(rd))), dtype=np.int16)
                m = min(nt, adc.shape[0])
                fixed[:m] = adc[:m]
                adc = fixed
            for val in adc.tolist():
                v_adc.push_back(int(val))
        otree.Fill()
        if (i + 1) % 10 == 0 or i + 1 == to_do:
            print(f"  ... {i + 1}/{to_do} events")

    out.cd()
    otree.Write()
    out.Close()
    print(f"# Wrote {to_do} event(s) to {args.output} (tree 'rawdigits')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
