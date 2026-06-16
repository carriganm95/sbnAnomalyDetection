#!/usr/bin/env python3
"""Find the working PyROOT access pattern for the raw::RawDigit vector branch.

Run:
    python scripts/probe_pyroot_access.py ../data/raw_decoded_reco_094.root
Paste the output back.
"""
from __future__ import annotations
import sys


def main(argv):
    if len(argv) < 2:
        print("usage: probe_pyroot_access.py <file.root>"); return 1
    path = argv[1]
    import ROOT
    ROOT.gErrorIgnoreLevel = ROOT.kError

    f = ROOT.TFile.Open(path)
    t = f.Get("Events")
    print("# entries:", t.GetEntries())

    print("\n=== branches mentioning RawDigit ===")
    for b in t.GetListOfBranches():
        n = b.GetName()
        if "RawDigit" in n:
            print(f"  branch: {n!r}  class={b.GetClassName()!r}")

    print("\n=== leaves mentioning RawDigit ===")
    for l in t.GetListOfLeaves():
        n = l.GetName()
        if "RawDigit" in n:
            print(f"  leaf: {n!r}  type={l.GetTypeName()!r}")

    base = "raw::RawDigits_daq__DECODE."
    t.GetEntry(0)

    print("\n=== getattr strategies (entry 0) ===")
    names = [
        base, base + "obj", base.rstrip("."), "raw::RawDigits_daq__DECODE",
        "raw::RawDigits_daq__DECODE.obj", "raw__RawDigits_daq__DECODE",
        "raw__RawDigits_daq__DECODE_", "RawDigits_daq__DECODE",
    ]
    for name in names:
        try:
            v = getattr(t, name)
            sz = v.size() if hasattr(v, "size") else ("len=%d" % len(v)
                                                       if hasattr(v, "__len__") else "n/a")
            print(f"  getattr({name!r}) -> {type(v).__name__}  size={sz}")
        except Exception as e:
            print(f"  getattr({name!r}) FAILED: {type(e).__name__}: {e}")

    print("\n=== wrapper.obj strategy ===")
    for wname in (base, base.rstrip("."), "raw::RawDigits_daq__DECODE"):
        try:
            w = getattr(t, wname)
            attrs = [a for a in dir(w) if not a.startswith("__")][:25]
            print(f"  wrapper {wname!r} type={type(w).__name__} attrs={attrs}")
            for member in ("obj", "product", "p"):
                if hasattr(w, member):
                    vec = getattr(w, member)
                    sz = vec.size() if hasattr(vec, "size") else "n/a"
                    print(f"    -> .{member} size={sz}")
                    if hasattr(vec, "size") and vec.size() > 0:
                        rd = vec[0]
                        rattrs = [a for a in dir(rd) if not a.startswith("__")][:30]
                        print(f"       RawDigit[0] attrs: {rattrs}")
            break
        except Exception as e:
            print(f"  wrapper {wname!r} FAILED: {type(e).__name__}: {e}")

    print("\n# Done. Paste this output back.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
