"""Correctly merge compact sparse-events .npz files (CSR format).

A naive per-key ``np.concatenate`` corrupts these files:
  - ``offsets`` is an index array into the flat per-hit arrays, so each file's
    offsets must be shifted by the running hit count and the duplicated leading
    0 dropped;
  - ``n_channels`` is a 0-d scalar -> take the max across files (and infer from
    the data if missing/zero);
  - ``evt_file_idx`` indexes ``filenames`` -> remap into the merged filename list.

Per-hit arrays (``channels_flat``/``integrals_flat``/``times_flat``/``wires_flat``/
``planes_flat``/``tpcs_flat``/``widths_flat``/``sumadcs_flat``/``mults_flat``/
``hassps_flat``) and per-event arrays (``evt_run``/``evt_subrun``/``evt_num``)
concatenate directly.

CLI:
    python -m sbn_anomaly.data.merge_events --output data/good_events_test.npz \
        data/good_events_00.npz data/good_events_01.npz ...
    python -m sbn_anomaly.data.merge_events --output out.npz --glob 'data/good_events_*.npz'
"""

from __future__ import annotations

import argparse
import glob as _glob
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

_PER_HIT = ["channels_flat", "integrals_flat", "times_flat",
            "wires_flat", "planes_flat", "tpcs_flat",
            "widths_flat", "sumadcs_flat", "mults_flat", "hassps_flat"]
_PER_EVENT = ["evt_run", "evt_subrun", "evt_num", "evt_file_idx"]


def merge_events_npz(files: Sequence[str], output: str) -> dict:
    """Merge sparse-events npz ``files`` into one CSR-correct npz at ``output``."""
    per_hit: dict[str, list] = {k: [] for k in _PER_HIT}
    per_hit_present = {k: True for k in _PER_HIT}
    per_event: dict[str, list] = {k: [] for k in _PER_EVENT}
    per_event_present = {k: True for k in _PER_EVENT}
    offsets_shifted: list[np.ndarray] = []
    filenames: list[str] = []

    cum_hits = 0        # running total length of channels_flat
    cum_files = 0       # running total number of filenames (for evt_file_idx remap)
    n_channels = 0
    n_events = 0

    for f in files:
        d = np.load(f, allow_pickle=False)
        if "channels_flat" not in d or "offsets" not in d:
            print(f"[skip] {f}: not a sparse-events npz")
            continue
        ch = d["channels_flat"]
        off = d["offsets"].astype(np.int64)

        if "n_channels" in d:
            n_channels = max(n_channels, int(d["n_channels"]))

        for k in _PER_HIT:
            if k in d:
                per_hit[k].append(d[k])
            else:
                per_hit_present[k] = False

        # Shift this file's offsets by the running hit count; drop leading 0.
        offsets_shifted.append(off[1:] + cum_hits)
        cum_hits += int(ch.size)
        n_events += int(off.size - 1)

        this_files = [str(x) for x in d["filenames"]] if "filenames" in d else []
        for k in _PER_EVENT:
            if k in d:
                arr = np.asarray(d[k])
                if k == "evt_file_idx":
                    arr = arr + cum_files  # remap into merged filename list
                per_event[k].append(arr)
            else:
                per_event_present[k] = False
        filenames += this_files
        cum_files += len(this_files)

    out: dict = {}
    for k in _PER_HIT:
        if per_hit_present[k] and per_hit[k]:
            out[k] = np.concatenate(per_hit[k])
    out["offsets"] = (
        np.concatenate([np.array([0], dtype=np.int64), *offsets_shifted])
        if offsets_shifted else np.array([0], dtype=np.int64)
    )

    if n_channels <= 0 and "channels_flat" in out and out["channels_flat"].size:
        n_channels = int(out["channels_flat"].max()) + 1
    out["n_channels"] = np.array(n_channels, dtype=np.int64)

    for k in _PER_EVENT:
        if per_event_present[k] and per_event[k]:
            out[k] = np.concatenate(per_event[k])
    if filenames:
        out["filenames"] = np.array(filenames, dtype="U512")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **out)
    total_hits = int(out.get("channels_flat", np.empty(0)).size)
    print(f"[merge] {len(files)} file(s) -> {output}: {n_events} events, "
          f"{total_hits} hits, n_channels={n_channels}")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Merge sparse-events npz files (CSR-correct).")
    p.add_argument("files", nargs="*", help="input sparse-events npz files")
    p.add_argument("--glob", default=None, help="glob pattern for input files")
    p.add_argument("--output", required=True, help="output merged npz")
    args = p.parse_args(argv)

    files = list(args.files)
    if args.glob:
        files += sorted(_glob.glob(args.glob))
    if not files:
        p.error("provide input files and/or --glob")
    merge_events_npz(files, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
