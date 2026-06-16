#!/usr/bin/env python3
"""Focused probe: can uproot decode raw::RawDigit ADC waveforms from an art file?

Targets the TPC raw-digit product (e.g. 'raw::RawDigits_daq__DECODE.') and its
'.obj' sub-branch (the std::vector<raw::RawDigit>), prints uproot's
interpretation, dumps the RawDigit streamer (member layout / fCompression), and
tries to actually read one event. Paste the output back.

Usage:
    python scripts/inspect_rawdigit_obj.py ../data/raw_decoded_reco_094.root
"""

from __future__ import annotations

import sys
import traceback


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python scripts/inspect_rawdigit_obj.py <file.root>")
        return 1
    path = argv[1]

    import uproot
    print(f"# uproot {uproot.__version__}")
    f = uproot.open(path)

    # ---- 1. show the RawDigit streamer (member layout, compression field) ----
    print("\n=== streamer for raw::RawDigit ===")
    try:
        f.show_streamers("raw::RawDigit")
    except Exception as exc:
        print("  show_streamers failed:", exc)
        try:
            for name in f.streamers:
                if "RawDigit" in name:
                    print("  streamer present:", name)
        except Exception as exc2:
            print("  (no streamer access:", exc2, ")")

    tree = f["Events"]

    # ---- 2. find the TPC raw-digit product branch (not Assns / TimeStamp) ----
    keys = list(tree.keys(recursive=False))
    cand = [k for k in keys
            if "RawDigits_daq" in k and "Assns" not in k and "TimeStamp" not in k]
    print("\n=== candidate raw-digit product branch(es) ===")
    for k in cand:
        print("  ", k)
    if not cand:
        print("  none found; aborting.")
        return 0

    prod = cand[0]
    print(f"\n=== inspecting product '{prod}' ===")
    branch = tree[prod]
    print("  sub-branches:")
    for sub in branch.keys():
        try:
            tn = branch[sub].typename
        except Exception:
            tn = "?"
        print(f"    {sub}   [{tn}]")

    # ---- 3. locate the .obj sub-branch and report its interpretation ----
    obj_key = None
    for sub in branch.keys():
        if sub.endswith(".obj") or sub.endswith("obj"):
            obj_key = sub
            break
    print(f"\n=== obj sub-branch: {obj_key} ===")
    if obj_key is None:
        print("  no .obj sub-branch; aborting.")
        return 0
    objb = branch[obj_key]
    try:
        print("  typename:", objb.typename)
    except Exception as exc:
        print("  typename failed:", exc)
    try:
        print("  interpretation:", repr(objb.interpretation)[:400])
    except Exception as exc:
        print("  interpretation failed:", exc)

    # ---- 4. the decisive test: actually read entry 0 ----
    print("\n=== attempting objb.array(entry_stop=1) ===")
    try:
        arr = objb.array(entry_stop=1)
        print("  SUCCESS. python type:", type(arr))
        try:
            import awkward as ak
            print("  awkward type:", ak.type(arr))
            rec0 = arr[0]
            print("  n_rawdigits in event 0:", len(rec0))
            print("  available fields:", getattr(rec0, "fields", "n/a"))
            # try to pull ADC + channel of the first RawDigit
            for fld in ("fADC", "fadc", "fChannel", "fPedestal",
                        "fSamples", "fCompression"):
                try:
                    val = rec0[fld]
                    sample = val[0] if hasattr(val, "__len__") else val
                    extra = ""
                    if hasattr(sample, "__len__"):
                        extra = f" (len {len(sample)}, first 8: {list(sample[:8])})"
                    print(f"    {fld}: ok{extra}")
                except Exception as exc:
                    print(f"    {fld}: not directly accessible ({exc})")
        except Exception as exc:
            print("  read ok but introspection failed:", exc)
        print("\n  >>> VERDICT: waveforms ARE readable in pure Python.")
    except Exception:
        print("  FAILED:")
        traceback.print_exc()
        print("\n  >>> VERDICT: uproot cannot decode these RawDigits "
              "(custom streamer / compression). The gallery/LArSoft dump on a "
              "gpvm or container is required.")

    print("\n# Done. Paste this whole output back.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
