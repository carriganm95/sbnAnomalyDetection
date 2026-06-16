#!/usr/bin/env python3
"""Probe an art ROOT file for raw::RawDigit branches readable by uproot.

Run this on a raw/decoded art file (the kind the gallery macro would read) to
discover whether the ADC waveforms can be read directly in Python -- i.e. without
the LArSoft/gallery C++ step. Paste the output back and a matching reader can be
built around the real branch names and types.

Usage:
    python scripts/inspect_art_rawdigits.py ../data/raw_decoded_reco_094.root
"""

from __future__ import annotations

import sys


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python scripts/inspect_art_rawdigits.py <file.root>")
        return 1
    path = argv[1]

    import uproot

    print(f"# uproot {uproot.__version__}")
    f = uproot.open(path)
    print("\n=== top-level keys (trees) ===")
    for k in f.keys(recursive=False):
        print("  ", k, "->", type(f[k]).__name__)

    # Pick the main event tree.
    tree_name = None
    for cand in ("Events", "Events;1"):
        if cand in f:
            tree_name = cand
            break
    if tree_name is None:
        # fall back to the first TTree
        for k in f.keys(recursive=False):
            try:
                if hasattr(f[k], "keys"):
                    tree_name = k
                    break
            except Exception:
                continue
    if tree_name is None:
        print("\nNo TTree found.")
        return 0
    tree = f[tree_name]
    print(f"\n=== using tree '{tree_name}' with {tree.num_entries} entries ===")

    # Find candidate RawDigit branches.
    all_keys = list(tree.keys())
    cands = [k for k in all_keys
             if "rawdigit" in k.lower() or k.lower().startswith("raw::")
             or "_daq_" in k.lower()]
    print(f"\n=== {len(cands)} candidate RawDigit branch name(s) ===")
    for k in cands[:60]:
        print("  ", k)
    if not cands:
        print("  (none matched 'RawDigit' / 'raw::' / '_daq_')")
        print("  --- showing all top-level branches for reference ---")
        for k in tree.keys(recursive=False)[:80]:
            print("    ", k)

    # Inspect the most promising top-level product branch and its members.
    top = sorted({k.split(".")[0].split("/")[0] for k in cands})
    print(f"\n=== top-level product branches: {top} ===")
    for prod in top:
        print(f"\n--- product '{prod}' ---")
        try:
            br = tree[prod]
            print("  typename:", getattr(br, "typename", "?"))
            print("  interpretation:", repr(br.interpretation)[:300])
        except Exception as exc:
            print("  (could not access product branch directly:", exc, ")")
        # member sub-branches (fADC, fChannel, fPedestal, fCompression, ...)
        members = [k for k in all_keys if k.startswith(prod)]
        interesting = [m for m in members if any(
            s in m for s in ("fADC", "fadc", "fChannel", "fPedestal",
                             "fSamples", "fCompression", "fSigma"))]
        print(f"  member branches of interest ({len(interesting)}):")
        for m in interesting[:30]:
            try:
                tn = tree[m].typename
            except Exception:
                tn = "?"
            print(f"    {m}   [{tn}]")

        # Try to actually read the ADC member for the first entry.
        adc_branches = [m for m in members if m.endswith("fADC") or m.endswith("fadc")
                        or "fADC" in m]
        for ab in adc_branches[:1]:
            print(f"  >>> attempting to read '{ab}' (entry 0) ...")
            try:
                arr = tree[ab].array(entry_stop=1, library="np")
                a0 = arr[0]
                import numpy as np
                a0 = np.asarray(a0, dtype=object) if a0 is None else a0
                print("      OK. type:", type(arr), "first-entry shape/len:",
                      getattr(a0, "shape", len(a0)))
                try:
                    flat = a0[0]
                    print("      first channel waveform len:", len(flat),
                          "dtype:", getattr(flat, "dtype", type(flat)))
                    print("      first 10 ADC samples:", list(flat[:10]))
                except Exception as exc:
                    print("      (could not index first channel:", exc, ")")
            except Exception as exc:
                print("      FAILED to read ADC member:", repr(exc))
                print("      -> ADCs are likely compressed or use a custom "
                      "streamer; the gallery/LArSoft dump is required.")

    print("\n# Done. Paste this whole output back to proceed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
