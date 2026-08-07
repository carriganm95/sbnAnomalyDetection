#!/usr/bin/env python3
"""
Filter train, good-test, and bad-test sparse NPZ files to a selected global
channel range, reindex the kept channels, and optionally keep selected runs.

Empty-event behavior
--------------------
Events with zero hits in the selected channel range are removed by default.
Use --keep-empty to preserve them.

Run filtering
-------------
Use --run-filter together with all three run-list options:

    --train-runs RUN [RUN ...]
    --good-runs RUN [RUN ...]
    --bad-runs RUN [RUN ...]

Each file is filtered independently using its evt_run array. When --run-filter
is enabled, every dataset must receive a non-empty run list.

By default, the listed runs are kept. Add --reverse-run-filter to instead remove
the listed runs and keep every other evt_run value.

Example
-------

    python script.py \
        --train events_train.npz \
        --good good_events_test.npz \
        --bad bad_events_test.npz \
        --start 8800 --end 9000 \
        --run-filter \
        --train-runs 19305 19308 19315 \
        --good-runs 20769 20782 \
        --bad-runs 20614 20615

Add --keep-empty to retain events that have no hits in [START, END). The
--force-ratio option is meaningful only when empty events are removed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# ============================================================
# Default user settings
# ============================================================

DEFAULT_TRAIN_NPZ_PATH = Path(
    "/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection/scratch/events_train.npz"
)

DEFAULT_GOOD_NPZ_PATH = Path(
    "/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection/scratch/good_events_test.npz"
)

DEFAULT_BAD_NPZ_PATH = Path(
    "/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection/scratch/bad_events_test.npz"
)

# Keep channels where:
#   START <= channels_flat < END
#
# The output file will reindex these global channels to local channels:
#   START     -> 0
#   START + 1 -> 1
#   ...
#   END - 1   -> END - START - 1
DEFAULT_START = 8800
DEFAULT_END = 9000

# Full detector channel count used for the original ratio.
DEFAULT_TOTAL_N_CHANNELS = 11276

# Deterministic random seed used only when --force-ratio selects a subset
# of zero-selected-hit events to preserve.
DEFAULT_RANDOM_SEED = 12345

# Run lists used only when --run-filter is enabled.
# Empty lists mean that no runs are kept for that dataset.
DEFAULT_TRAIN_RUNS_TO_KEEP: list[int] = [
    20184, 
    20186, 
    20188, 
    20190, 
    20199, 
    20201, 
    20202, 
    20205, 
    20207, 
    20208, 
    20209, 
    20211, 
    20218, 
    20219, 
    20221, 
    20223, 
    20224,
]
DEFAULT_GOOD_RUNS_TO_KEEP: list[int] = [
    20074, 
    20075, 
    20079, 
    20080, 
    20082, 
    20101, 
    20104, 
    20105, 
    20106, 
    20108, 
    20110, 
    20113, 
    20115, 
    20116, 
    20117, 
    20118, 
    20121, 
    20126, 
    20127, 
    20128, 
    20130, 
    20132, 
    20144, 
    20153,
]
DEFAULT_BAD_RUNS_TO_KEEP: list[int] = [
    19627,
    19946,
]


# ============================================================
# Ratio helpers
# ============================================================

def per_event_unique_channel_ratio(
    channels_flat: np.ndarray,
    offsets: np.ndarray,
    denominator_n_channels: int,
) -> np.ndarray:
    """
    For each event, compute:

        number of unique channels with hits in that event / denominator_n_channels

    Full detector:
        denominator_n_channels = 11276

    Selected range:
        denominator_n_channels = END - START
    """
    if denominator_n_channels <= 0:
        raise ValueError(
            f"denominator_n_channels must be positive, got {denominator_n_channels}"
        )

    n_events = len(offsets) - 1
    ratios = np.zeros(n_events, dtype=np.float64)

    for i in range(n_events):
        s = int(offsets[i])
        e = int(offsets[i + 1])

        event_channels = channels_flat[s:e]

        if event_channels.size == 0:
            ratios[i] = 0.0
        else:
            ratios[i] = len(np.unique(event_channels)) / float(denominator_n_channels)

    return ratios


def summarize_ratio(name: str, ratios: np.ndarray) -> dict[str, float]:
    ratios = np.asarray(ratios, dtype=np.float64)

    if ratios.size == 0:
        out = {
            "n": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    else:
        out = {
            "n": int(ratios.size),
            "mean": float(np.mean(ratios)),
            "std": float(np.std(ratios)),
            "min": float(np.min(ratios)),
            "median": float(np.median(ratios)),
            "p90": float(np.percentile(ratios, 90)),
            "p95": float(np.percentile(ratios, 95)),
            "p99": float(np.percentile(ratios, 99)),
            "max": float(np.max(ratios)),
        }

    print(f"{name}:")
    print(f"  n events: {out['n']}")
    print(f"  mean:     {out['mean']:.8f}")
    print(f"  std:      {out['std']:.8f}")
    print(f"  min:      {out['min']:.8f}")
    print(f"  median:   {out['median']:.8f}")
    print(f"  p90:      {out['p90']:.8f}")
    print(f"  p95:      {out['p95']:.8f}")
    print(f"  p99:      {out['p99']:.8f}")
    print(f"  max:      {out['max']:.8f}")

    return out


def print_ratio_comparison(
    *,
    full_ratios: np.ndarray,
    selected_ratios: np.ndarray,
    selected_name: str,
) -> dict[str, float | bool]:
    """
    Compare selected-range channel-hit ratio against full-detector ratio.

    Warning condition:

        abs(selected_mean - full_mean) > full_std

    Here full_std is the event-by-event standard deviation of:

        unique full-detector hit channels per event / 11276
    """
    print("-" * 80)
    print(f"Full-detector vs selected-range ratio comparison: {selected_name}")
    print("-" * 80)

    full_stats = summarize_ratio(
        "Full-detector per-event ratio = unique hit channels / all detector channels",
        full_ratios,
    )
    print()
    selected_stats = summarize_ratio(
        "Selected-range per-event ratio = unique selected hit channels / selected channels",
        selected_ratios,
    )

    diff = abs(float(selected_stats["mean"]) - float(full_stats["mean"]))
    threshold = float(full_stats["std"])

    warning = False

    print()
    print("Comparison:")
    print(f"  full mean:        {float(full_stats['mean']):.8f}")
    print(f"  selected mean:    {float(selected_stats['mean']):.8f}")
    print(f"  absolute diff:    {diff:.8f}")
    print(f"  warning cutoff:   full std = {threshold:.8f}")

    if threshold == 0.0:
        if diff > 0.0:
            warning = True
            print()
            print("WARNING:")
            print(
                "  Full-detector ratio std is 0, but the selected-range mean ratio "
                "differs from the full-detector mean."
            )
        else:
            print("  No warning: selected mean equals full mean and full std is 0.")
    elif diff > threshold:
        warning = True
        print()
        print("WARNING:")
        print(
            "  Selected channel range has a large deviation in channel-hit ratio."
        )
        print(
            "  The selected-range mean ratio differs from the full-detector mean "
            "by more than one full-detector standard deviation."
        )
    else:
        print("  No warning: selected ratio is within one full-detector std.")

    print("-" * 80)

    return {
        "full_mean": float(full_stats["mean"]),
        "full_std": float(full_stats["std"]),
        "selected_mean": float(selected_stats["mean"]),
        "selected_std": float(selected_stats["std"]),
        "abs_diff": float(diff),
        "warning": bool(warning),
    }


# ============================================================
# Filtering helpers
# ============================================================

def count_selected_hits_per_event(
    channels_flat: np.ndarray,
    offsets: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    """Return number of hits inside [start, end) for each event."""
    n_events = len(offsets) - 1
    out = np.zeros(n_events, dtype=np.int64)

    for i in range(n_events):
        s = int(offsets[i])
        e = int(offsets[i + 1])
        ev_channels = channels_flat[s:e]
        out[i] = int(np.count_nonzero((ev_channels >= start) & (ev_channels < end)))

    return out


def selected_unique_ratio_per_original_event(
    channels_flat: np.ndarray,
    offsets: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    """
    For each original event, compute selected-range active-channel ratio:

        unique selected channels in event / (end - start)

    Events with no selected hits get ratio 0.
    """
    n_events = len(offsets) - 1
    local_n_channels = end - start
    ratios = np.zeros(n_events, dtype=np.float64)

    for i in range(n_events):
        s = int(offsets[i])
        e = int(offsets[i + 1])

        ev_channels = channels_flat[s:e]
        mask = (ev_channels >= start) & (ev_channels < end)
        selected = ev_channels[mask]

        if selected.size == 0:
            ratios[i] = 0.0
        else:
            ratios[i] = len(np.unique(selected)) / float(local_n_channels)

    return ratios


def choose_zero_hit_events_to_balance_ratio(
    *,
    selected_ratios_all_events: np.ndarray,
    nonzero_event_mask: np.ndarray,
    target_mean: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray | None, str, bool]:
    """
    Choose a subset of zero-selected-hit events to preserve so that the selected
    ratio mean gets closer to target_mean.

    Returns
    -------
    keep_event_mask:
        Boolean event mask, or None if force-ratio cannot help.

    message:
        Explanation message.

    best_effort_warning:
        True if exact balancing was impossible but the function still returns
        the best possible event mask.
    """
    selected_ratios_all_events = np.asarray(selected_ratios_all_events, dtype=np.float64)
    nonzero_event_mask = np.asarray(nonzero_event_mask, dtype=bool)

    n_total = selected_ratios_all_events.size
    nonzero_indices = np.where(nonzero_event_mask)[0]
    zero_indices = np.where(~nonzero_event_mask)[0]

    if nonzero_indices.size == 0:
        return None, (
            "No events have hits in the selected channel range, so there is no "
            "nonempty selected-range dataset to balance."
        ), False

    if zero_indices.size == 0:
        return None, (
            "--force-ratio was requested, but there are no zero-selected-hit "
            "events available to preserve."
        ), False

    nonzero_ratios = selected_ratios_all_events[nonzero_indices]
    normal_mean = float(np.mean(nonzero_ratios))
    all_events_mean = float(np.mean(selected_ratios_all_events))

    normal_diff = abs(normal_mean - target_mean)
    all_diff = abs(all_events_mean - target_mean)

    if normal_mean < target_mean:
        return None, (
            "--force-ratio was requested, but the nonempty-event selected ratio "
            "is already below the full-detector ratio. Preserving zero-hit events "
            "would only make the selected ratio even smaller, so force-ratio "
            "cannot help."
        ), False

    # ------------------------------------------------------------
    # If preserving all available zero-hit events is still not enough,
    # exact balancing would require adding fake/new zero-hit events.
    #
    # New behavior:
    #   Do NOT ignore --force-ratio.
    #   Keep all available zero-hit events as the best possible balance.
    # ------------------------------------------------------------
    if all_events_mean > target_mean:
        keep_event_mask = np.ones(n_total, dtype=bool)

        return keep_event_mask, (
            "--force-ratio was requested, but even preserving all available "
            "zero-selected-hit events does not lower the selected ratio enough "
            "to match the full-detector ratio. Exact balancing would require "
            "adding fake/new zero-hit events, which this script will not do. "
            "Applying best-effort balancing by preserving all available "
            f"{zero_indices.size} zero-selected-hit event(s)."
        ), True

    if all_diff >= normal_diff and normal_mean != target_mean:
        return None, (
            "--force-ratio was requested, but preserving zero-selected-hit events "
            "does not improve the selected/full ratio agreement."
        ), False

    # We need to preserve some number k of zero-hit events.
    #
    # Let:
    #   S = sum ratios over nonzero selected-hit events
    #   N = number of nonzero selected-hit events
    #
    # Preserving k zero-ratio events gives:
    #   forced_mean = S / (N + k)
    #
    # Solve approximately:
    #   target_mean = S / (N + k)
    #   k = S / target_mean - N
    S = float(np.sum(nonzero_ratios))
    N = int(nonzero_indices.size)

    if target_mean <= 0.0:
        keep_event_mask = np.ones(n_total, dtype=bool)

        return keep_event_mask, (
            "--force-ratio was requested, but the full-detector target mean is "
            "0 while the selected range has nonzero-hit events. Exact balancing "
            "would require an undefined/infinite number of zero-hit events. "
            "Applying best-effort balancing by preserving all available "
            f"{zero_indices.size} zero-selected-hit event(s)."
        ), True

    k_float = S / target_mean - N
    k_floor = int(np.floor(k_float))
    k_ceil = int(np.ceil(k_float))

    candidate_ks = sorted(
        set(
            k
            for k in [k_floor - 1, k_floor, k_floor + 1, k_ceil, k_ceil + 1]
            if 0 <= k <= zero_indices.size
        )
    )

    # If the computed k is outside the available range, best effort means:
    #   - if k is too large: keep all zero-hit events
    #   - if k is negative: force-ratio cannot help
    if not candidate_ks:
        if k_float > zero_indices.size:
            keep_event_mask = np.ones(n_total, dtype=bool)

            return keep_event_mask, (
                "--force-ratio was requested, but the required number of "
                "zero-selected-hit events is larger than the number available. "
                "Exact balancing would require adding fake/new zero-hit events. "
                "Applying best-effort balancing by preserving all available "
                f"{zero_indices.size} zero-selected-hit event(s)."
            ), True

        return None, (
            "--force-ratio was requested, but the required number of zero-hit "
            "events is outside the available range and force-ratio cannot help."
        ), False

    best_k = None
    best_diff = None

    for k in candidate_ks:
        forced_mean = S / float(N + k)
        diff = abs(forced_mean - target_mean)

        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_k = k

    if best_k is None:
        return None, "Internal error: could not choose a zero-hit-event count.", False

    if best_k <= 0:
        return None, (
            "--force-ratio was requested, but the best balanced solution keeps "
            "zero additional zero-hit events. Ignoring the flag."
        ), False

    if best_diff is not None and best_diff >= normal_diff:
        return None, (
            "--force-ratio was requested, but the best available zero-hit-event "
            "subset does not improve the selected/full ratio agreement."
        ), False

    chosen_zero_indices = rng.choice(zero_indices, size=best_k, replace=False)

    keep_event_mask = np.zeros(n_total, dtype=bool)
    keep_event_mask[nonzero_indices] = True
    keep_event_mask[chosen_zero_indices] = True

    return keep_event_mask, (
        f"--force-ratio selected {best_k} zero-selected-hit event(s) out of "
        f"{zero_indices.size} available to bring selected ratio closer to "
        f"full-detector ratio."
    ), False


def build_filtered_from_event_mask(
    *,
    data: np.lib.npyio.NpzFile,
    start: int,
    end: int,
    keep_event_mask: np.ndarray,
) -> dict[str, np.ndarray | int]:
    """
    Build filtered output using a given event mask.

    Hits are always filtered to [start, end) and channels are reindexed.
    Events kept by keep_event_mask may have zero selected hits.
    """
    channels_flat = data["channels_flat"]
    offsets = data["offsets"]

    n_events = len(offsets) - 1
    local_n_channels = end - start

    if keep_event_mask.shape != (n_events,):
        raise ValueError(
            f"keep_event_mask has shape {keep_event_mask.shape}, expected {(n_events,)}"
        )

    selected_hit_mask = np.zeros(len(channels_flat), dtype=bool)
    new_offsets_list = [0]
    running_total = 0

    selected_local_parts: list[np.ndarray] = []

    for i in range(n_events):
        if not keep_event_mask[i]:
            continue

        old_s = int(offsets[i])
        old_e = int(offsets[i + 1])

        ev_channels = channels_flat[old_s:old_e]
        ev_mask = (ev_channels >= start) & (ev_channels < end)

        global_indices = np.arange(old_s, old_e)[ev_mask]
        selected_hit_mask[global_indices] = True

        ev_selected_global = ev_channels[ev_mask].astype(np.int64)
        ev_selected_local = ev_selected_global - start
        selected_local_parts.append(ev_selected_local)

        running_total += ev_selected_local.size
        new_offsets_list.append(running_total)

    new_offsets = np.asarray(new_offsets_list, dtype=offsets.dtype)

    if selected_local_parts:
        selected_local_channels = np.concatenate(selected_local_parts).astype(np.int64)
    else:
        selected_local_channels = np.zeros(0, dtype=np.int64)

    if selected_local_channels.size != int(new_offsets[-1]):
        raise ValueError(
            "Internal error: selected channel count does not match rebuilt offsets. "
            f"selected_local_channels.size={selected_local_channels.size}, "
            f"new_offsets[-1]={new_offsets[-1]}"
        )

    if selected_local_channels.size > 0:
        local_min = int(selected_local_channels.min())
        local_max = int(selected_local_channels.max())

        if local_min < 0 or local_max >= local_n_channels:
            raise ValueError(
                "Reindexed channels are outside the expected local range. "
                f"Expected [0, {local_n_channels}), got min={local_min}, max={local_max}"
            )

    selected_ratios = per_event_unique_channel_ratio(
        selected_local_channels,
        new_offsets,
        local_n_channels,
    )

    return {
        "selected_hit_mask": selected_hit_mask,
        "keep_event_mask": keep_event_mask,
        "new_offsets": new_offsets,
        "selected_local_channels": selected_local_channels,
        "selected_ratios": selected_ratios,
        "n_events_kept": int(np.count_nonzero(keep_event_mask)),
        "n_events_removed": int(n_events - np.count_nonzero(keep_event_mask)),
        "n_hits_kept": int(np.count_nonzero(selected_hit_mask)),
    }



def sort_filtered_output_by_evt_time(
    output_data: dict[str, np.ndarray],
    *,
    flat_keys: list[str],
    event_keys: list[str],
    name: str,
) -> dict[str, np.ndarray]:
    """
    Stably sort a filtered sparse-event output by ``evt_time``.

    Each event is moved as one complete unit. All event-level arrays use the
    same event permutation, every event's complete flat-hit slice is moved
    with it, and ``offsets`` is rebuilt. No event or hit slice is divided.
    """
    if "evt_time" not in output_data:
        raise KeyError(
            f"{name}: cannot sort by evt_time because the input NPZ has no "
            "'evt_time' event array."
        )
    if "offsets" not in output_data:
        raise KeyError(f"{name}: required array 'offsets' is missing.")

    evt_time = np.asarray(output_data["evt_time"])
    offsets = np.asarray(output_data["offsets"])

    if evt_time.ndim != 1:
        raise ValueError(
            f"{name}: evt_time must be one-dimensional, got {evt_time.shape}."
        )

    n_events = int(evt_time.size)
    if offsets.ndim != 1 or offsets.size != n_events + 1:
        raise ValueError(
            f"{name}: offsets shape {offsets.shape} is inconsistent with "
            f"{n_events:,} filtered events."
        )
    if offsets.size == 0 or offsets[0] != 0:
        raise ValueError(f"{name}: offsets must begin with 0.")
    if np.any(np.diff(offsets) < 0):
        raise ValueError(f"{name}: offsets must be non-decreasing.")

    # Stable sorting keeps the existing relative order for equal timestamps.
    event_order = np.argsort(evt_time, kind="stable")

    counts = offsets[1:] - offsets[:-1]
    sorted_counts = counts[event_order]
    sorted_offsets = np.empty(n_events + 1, dtype=offsets.dtype)
    sorted_offsets[0] = 0
    np.cumsum(sorted_counts, out=sorted_offsets[1:])

    total_hits = int(offsets[-1])
    flat_order = np.empty(total_hits, dtype=np.int64)
    destination = 0

    for event_index in event_order:
        old_start = int(offsets[event_index])
        old_stop = int(offsets[event_index + 1])
        length = old_stop - old_start

        if length:
            flat_order[destination:destination + length] = np.arange(
                old_start, old_stop, dtype=np.int64
            )
            destination += length

    if destination != total_hits:
        raise RuntimeError(
            f"{name}: rebuilt flat ordering contains {destination:,} hits; "
            f"expected {total_hits:,}."
        )

    sorted_output = dict(output_data)
    sorted_output["offsets"] = sorted_offsets

    for key in flat_keys:
        if key not in output_data:
            continue
        array = np.asarray(output_data[key])
        if array.ndim == 0 or array.shape[0] != total_hits:
            raise ValueError(
                f"{name}: flat array '{key}' has shape {array.shape}; "
                f"expected first dimension {total_hits:,}."
            )
        sorted_output[key] = array[flat_order]

    for key in event_keys:
        if key not in output_data:
            continue
        array = np.asarray(output_data[key])
        if array.ndim == 0 or array.shape[0] != n_events:
            raise ValueError(
                f"{name}: event array '{key}' has shape {array.shape}; "
                f"expected first dimension {n_events:,}."
            )
        sorted_output[key] = array[event_order]

    sorted_time = np.asarray(sorted_output["evt_time"])
    decreases = np.flatnonzero(np.diff(sorted_time) < 0)
    if decreases.size:
        raise RuntimeError(
            f"{name}: evt_time sorting verification failed at "
            f"{decreases.size:,} position(s)."
        )

    moved_events = int(np.count_nonzero(event_order != np.arange(n_events)))
    print(
        f"Sorted {name} by evt_time: {n_events:,} complete events, "
        f"{total_hits:,} synchronized hits, {moved_events:,} events moved."
    )

    return sorted_output

def make_output_path(
    input_path: Path,
    start: int,
    end: int,
    *,
    throw_empty: bool,
    force_ratio_effective: bool,
    run_filter: bool,
    reverse_run_filter: bool,
) -> Path:
    suffix = f"_ch{start}_{end}_reindexed"

    if throw_empty:
        suffix += "_nonempty"

    if run_filter:
        suffix += "_reverse_runfilter" if reverse_run_filter else "_runfilter"

    if force_ratio_effective:
        suffix += "_force_ratio"

    return input_path.with_name(input_path.stem + suffix + input_path.suffix)


# ============================================================
# Main filtering function
# ============================================================

def filter_npz_by_channel_range(
    input_path: Path,
    start: int,
    end: int,
    *,
    total_n_channels: int,
    throw_empty: bool = True,
    force_ratio: bool = False,
    random_seed: int = DEFAULT_RANDOM_SEED,
    run_filter: bool = False,
    runs_to_keep: list[int] | None = None,
    reverse_run_filter: bool = False,
) -> Path:
    if start >= end:
        raise ValueError(f"START must be < END, but got START={start}, END={end}")

    if total_n_channels <= 0:
        raise ValueError(
            f"total_n_channels must be positive, got {total_n_channels}"
        )

    input_path = input_path.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    if input_path.suffix != ".npz":
        raise ValueError(f"Input file must be an .npz file, got: {input_path}")

    force_ratio_effective = bool(force_ratio and throw_empty)

    output_path = make_output_path(
        input_path,
        start,
        end,
        throw_empty=throw_empty,
        force_ratio_effective=force_ratio_effective,
        run_filter=run_filter,
        reverse_run_filter=reverse_run_filter,
    )

    local_n_channels = end - start
    rng = np.random.default_rng(random_seed)

    if force_ratio and not throw_empty:
        print()
        print("WARNING:")
        print(
            "  --force-ratio/--force-balance was requested together with --keep-empty. "
            "Ignoring force balance because all empty events are being kept."
        )
        print()

    with np.load(input_path, allow_pickle=True) as data:
        # Only the core arrays needed for filtering are mandatory.
        # Additional hit-level and event-level arrays are preserved when present.
        required_keys = [
            "channels_flat",
            "integrals_flat",
            "offsets",
            "n_channels",
            "evt_run",
            "evt_subrun",
            "evt_num",
            "evt_file_idx",
            "filenames",
        ]

        optional_flat_keys = [
            "times_flat",
            "wires_flat",
            "planes_flat",
            "tpcs_flat",
            "widths_flat",
            "sumadcs_flat",
            "mults_flat",
            "hassps_flat",
        ]

        optional_event_keys = [
            "evt_time",
        ]

        missing = [key for key in required_keys if key not in data]
        if missing:
            raise KeyError(f"Missing required keys in NPZ file: {missing}")

        channels_flat = data["channels_flat"]
        offsets = data["offsets"]

        if offsets.ndim != 1:
            raise ValueError(f"offsets must be 1D, got shape {offsets.shape}")

        if len(offsets) == 0:
            raise ValueError("offsets is empty")

        n_events = len(offsets) - 1

        if n_events < 0:
            raise ValueError("offsets must contain at least one element")

        evt_run = np.asarray(data["evt_run"])
        if run_filter:
            requested_runs = np.asarray(
                [] if runs_to_keep is None else runs_to_keep,
                dtype=evt_run.dtype,
            )
            listed_event_mask = np.isin(evt_run, requested_runs)
            eligible_event_mask = (
                ~listed_event_mask if reverse_run_filter else listed_event_mask
            )

            if not np.any(eligible_event_mask):
                available_runs = np.unique(evt_run).tolist()
                action = "removed" if reverse_run_filter else "kept"
                raise ValueError(
                    f"Run filtering {action} all events in {input_path.name}. "
                    f"Listed runs: {requested_runs.tolist()}; "
                    f"available runs: {available_runs}"
                )

            action = "removing" if reverse_run_filter else "keeping"
            print(
                f"Run filter enabled for {input_path.name}: {action} listed runs "
                f"{requested_runs.tolist()} and keeping "
                f"{np.count_nonzero(eligible_event_mask):,} / {n_events:,} events."
            )
        else:
            requested_runs = np.asarray([], dtype=evt_run.dtype)
            eligible_event_mask = np.ones(n_events, dtype=bool)

        if offsets[0] != 0:
            raise ValueError(f"offsets[0] should be 0, got {offsets[0]}")

        if offsets[-1] != len(channels_flat):
            raise ValueError(
                f"offsets[-1] should equal len(channels_flat), but got "
                f"offsets[-1]={offsets[-1]} and len(channels_flat)={len(channels_flat)}"
            )

        if np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("offsets must be monotonically nondecreasing")

        # Validate mandatory and optional hit-level arrays.
        flat_keys = [
            "channels_flat",
            "integrals_flat",
        ]
        flat_keys.extend(key for key in optional_flat_keys if key in data)

        for key in flat_keys:
            array = np.asarray(data[key])
            if array.ndim == 0 or len(array) != len(channels_flat):
                raise ValueError(
                    f"{key} has shape {array.shape}, but expected first dimension "
                    f"{len(channels_flat)} to match channels_flat"
                )

        missing_optional_flat = [
            key for key in optional_flat_keys if key not in data
        ]
        if missing_optional_flat:
            print(
                "Optional hit-level arrays not present; skipping: "
                + ", ".join(missing_optional_flat)
            )

        # Validate mandatory and optional event-level arrays.
        event_keys = [
            "evt_run",
            "evt_subrun",
            "evt_num",
            "evt_file_idx",
        ]
        event_keys.extend(key for key in optional_event_keys if key in data)

        for key in event_keys:
            array = np.asarray(data[key])
            if array.ndim == 0 or len(array) != n_events:
                raise ValueError(
                    f"{key} has shape {array.shape}, but expected first dimension "
                    f"{n_events} from offsets"
                )

        missing_optional_event = [
            key for key in optional_event_keys if key not in data
        ]
        if missing_optional_event:
            print(
                "Optional event-level arrays not present; skipping: "
                + ", ".join(missing_optional_event)
            )

        # ------------------------------------------------------------
        # Full-detector initial ratio BEFORE channel range filtering.
        # ------------------------------------------------------------

        full_ratios_all_events = per_event_unique_channel_ratio(
            channels_flat,
            offsets,
            total_n_channels,
        )
        full_ratios = full_ratios_all_events[eligible_event_mask]

        full_mean = float(np.mean(full_ratios)) if full_ratios.size else 0.0

        full_unique_hit_channels = (
            len(np.unique(channels_flat)) if channels_flat.size else 0
        )
        full_overall_ratio = full_unique_hit_channels / float(total_n_channels)

        # ------------------------------------------------------------
        # Per-original-event selected ratio.
        #
        # This has length equal to the original number of events. Events with
        # no selected-range hits have ratio 0.
        # ------------------------------------------------------------

        selected_ratios_all_events = selected_unique_ratio_per_original_event(
            channels_flat,
            offsets,
            start,
            end,
        )

        selected_hit_counts_per_event = count_selected_hits_per_event(
            channels_flat,
            offsets,
            start,
            end,
        )

        nonzero_event_mask = selected_hit_counts_per_event > 0
        eligible_nonzero_event_mask = eligible_event_mask & nonzero_event_mask

        if throw_empty and not np.any(eligible_nonzero_event_mask):
            raise ValueError(
                f"No events have any pulses in channel range [{start}, {end}), "
                "and the default empty-event removal would remove every event. "
                "Use --keep-empty to retain them."
            )

        # ------------------------------------------------------------
        # Normal candidate:
        #   default behavior: keep every original event after filtering hits
        #   --throw-empty:    keep only events with at least one selected hit
        # ------------------------------------------------------------

        if throw_empty:
            normal_keep_event_mask = eligible_nonzero_event_mask.copy()
            normal_selected_name = (
                "normal candidate, zero-selected-hit events removed because "
                "empty events are removed by default"
            )
        else:
            normal_keep_event_mask = eligible_event_mask.copy()
            normal_selected_name = (
                "normal candidate, zero-selected-hit events kept because "
                "--keep-empty was requested"
            )

        normal = build_filtered_from_event_mask(
            data=data,
            start=start,
            end=end,
            keep_event_mask=normal_keep_event_mask,
        )

        normal_comparison = print_ratio_comparison(
            full_ratios=full_ratios,
            selected_ratios=np.asarray(normal["selected_ratios"]),
            selected_name=normal_selected_name,
        )

        # ------------------------------------------------------------
        # Force-ratio candidate:
        #   keep all nonzero selected-hit events,
        #   plus only as many zero-selected-hit events as needed.
        #
        # If exact balancing would require adding fake/new zero-hit events,
        # apply best effort by keeping all available zero-hit events.
        # ------------------------------------------------------------

        selected = normal
        force_ratio_applied = False
        force_ratio_message = ""
        force_ratio_ignored_reason = None
        best_effort_warning = False

        if force_ratio_effective:
            (
                forced_keep_event_mask,
                force_ratio_message,
                best_effort_warning,
            ) = choose_zero_hit_events_to_balance_ratio(
                selected_ratios_all_events=selected_ratios_all_events[eligible_event_mask],
                nonzero_event_mask=nonzero_event_mask[eligible_event_mask],
                target_mean=full_mean,
                rng=rng,
            )

            if forced_keep_event_mask is not None:
                eligible_indices = np.flatnonzero(eligible_event_mask)
                full_forced_mask = np.zeros(n_events, dtype=bool)
                full_forced_mask[eligible_indices] = forced_keep_event_mask
                forced_keep_event_mask = full_forced_mask

            if forced_keep_event_mask is None:
                force_ratio_ignored_reason = force_ratio_message
            else:
                forced = build_filtered_from_event_mask(
                    data=data,
                    start=start,
                    end=end,
                    keep_event_mask=forced_keep_event_mask,
                )

                forced_comparison = print_ratio_comparison(
                    full_ratios=full_ratios,
                    selected_ratios=np.asarray(forced["selected_ratios"]),
                    selected_name=(
                        "force-ratio candidate, subset/best-effort set of "
                        "zero-selected-hit events preserved"
                    ),
                )

                normal_diff = float(normal_comparison["abs_diff"])
                forced_diff = float(forced_comparison["abs_diff"])

                if forced_diff >= normal_diff:
                    force_ratio_ignored_reason = (
                        "--force-ratio was requested, but the chosen zero-hit-event "
                        "set does not improve the selected/full ratio agreement. "
                        "Ignoring --force-ratio."
                    )
                else:
                    selected = forced
                    force_ratio_applied = True

                    print()
                    print("Force-ratio applied:")
                    print(f"  {force_ratio_message}")
                    print()

                    if best_effort_warning:
                        print()
                        print("WARNING:")
                        print("  Exact force-ratio balancing was impossible.")
                        print("  Applying the closest possible balance instead.")
                        print()

                    if bool(forced_comparison["warning"]):
                        print()
                        print("WARNING:")
                        print(
                            "  --force-ratio improved the selected/full ratio agreement, "
                            "but the selected range still differs from the full detector "
                            "by more than one full-detector standard deviation."
                        )
                        print(
                            "  This means the selected 200-channel region may genuinely "
                            "have a different hit-channel occupancy than the full detector, "
                            "or not enough zero-hit events were available to balance it."
                        )
                        print()

        if force_ratio_effective and force_ratio_ignored_reason is not None:
            print()
            print("WARNING:")
            print(f"  {force_ratio_ignored_reason}")
            print("  Falling back to the normal nonempty-event output.")
            print()

        selected_hit_mask = np.asarray(selected["selected_hit_mask"])
        keep_event_mask = np.asarray(selected["keep_event_mask"])
        new_offsets = np.asarray(selected["new_offsets"])
        selected_local_channels = np.asarray(selected["selected_local_channels"])
        selected_ratios_final = np.asarray(selected["selected_ratios"])

        n_events_kept = int(selected["n_events_kept"])
        n_events_removed = int(selected["n_events_removed"])
        kept_hits = int(selected["n_hits_kept"])

        # ------------------------------------------------------------
        # Build output in exact key order.
        # np.savez_compressed preserves insertion order.
        # ------------------------------------------------------------

        output_data = {}

        output_data["channels_flat"] = selected_local_channels.astype(
            data["channels_flat"].dtype
        )
        output_data["integrals_flat"] = data["integrals_flat"][selected_hit_mask]
        output_data["offsets"] = new_offsets
        output_data["n_channels"] = np.array(local_n_channels, dtype=np.int64)

        # Preserve every supported optional hit-level array that exists.
        for key in optional_flat_keys:
            if key in data:
                output_data[key] = data[key][selected_hit_mask]

        # Event-level metadata.
        output_data["evt_run"] = data["evt_run"][keep_event_mask]
        output_data["evt_subrun"] = data["evt_subrun"][keep_event_mask]
        output_data["evt_num"] = data["evt_num"][keep_event_mask]
        output_data["evt_file_idx"] = data["evt_file_idx"][keep_event_mask]

        # Preserve optional event-level arrays only when present.
        for key in optional_event_keys:
            if key in data:
                output_data[key] = data[key][keep_event_mask]

        # File list stays unchanged because evt_file_idx still points into this list.
        output_data["filenames"] = data["filenames"]

        # Helpful metadata.
        output_data["original_channel_start"] = np.array(start, dtype=np.int64)
        output_data["original_channel_end"] = np.array(end, dtype=np.int64)
        output_data["original_total_n_channels"] = np.array(
            total_n_channels,
            dtype=np.int64,
        )
        output_data["channels_are_reindexed"] = np.array(True)
        # Legacy metadata retained for compatibility with existing readers.
        output_data["throw_empty_requested"] = np.array(bool(throw_empty))
        output_data["keep_empty_requested"] = np.array(not bool(throw_empty))
        output_data["force_ratio_requested"] = np.array(bool(force_ratio))
        output_data["force_ratio_effective"] = np.array(bool(force_ratio_effective))
        output_data["force_ratio_applied"] = np.array(bool(force_ratio_applied))
        output_data["force_ratio_best_effort"] = np.array(bool(best_effort_warning))
        output_data["random_seed"] = np.array(random_seed, dtype=np.int64)
        output_data["run_filter_requested"] = np.array(bool(run_filter))
        output_data["reverse_run_filter_requested"] = np.array(
            bool(reverse_run_filter)
        )
        output_data["run_filter_values"] = requested_runs.astype(np.int64)
        # Retained for compatibility. In reverse mode these are the listed runs
        # to remove rather than runs to keep.
        output_data["runs_to_keep"] = requested_runs.astype(np.int64)

        output_data["full_detector_overall_channel_hit_ratio"] = np.array(
            full_overall_ratio,
            dtype=np.float64,
        )
        output_data["full_detector_per_event_ratio_mean"] = np.array(
            float(np.mean(full_ratios)) if full_ratios.size else 0.0,
            dtype=np.float64,
        )
        output_data["full_detector_per_event_ratio_std"] = np.array(
            float(np.std(full_ratios)) if full_ratios.size else 0.0,
            dtype=np.float64,
        )
        output_data["selected_per_event_ratio_mean"] = np.array(
            float(np.mean(selected_ratios_final)) if selected_ratios_final.size else 0.0,
            dtype=np.float64,
        )
        output_data["selected_per_event_ratio_std"] = np.array(
            float(np.std(selected_ratios_final)) if selected_ratios_final.size else 0.0,
            dtype=np.float64,
        )

        # Sort only after filtering is complete. Each event's entire hit slice is
        # moved together, and offsets is rebuilt, so no event is split.
        output_data = sort_filtered_output_by_evt_time(
            output_data,
            flat_keys=[
                "channels_flat",
                "integrals_flat",
                *[key for key in optional_flat_keys if key in output_data],
            ],
            event_keys=[
                "evt_run",
                "evt_subrun",
                "evt_num",
                "evt_file_idx",
                "evt_time",
            ],
            name=input_path.name,
        )

        np.savez_compressed(output_path, **output_data)

    original_hits = len(channels_flat)
    original_events = n_events
    selected_unique_hit_channels = (
        len(np.unique(selected_local_channels))
        if selected_local_channels.size
        else 0
    )
    selected_overall_ratio = selected_unique_hit_channels / float(local_n_channels)

    zero_selected_events_total = int(
        np.count_nonzero(eligible_event_mask & (~nonzero_event_mask))
    )
    zero_selected_events_kept = int(
        np.count_nonzero(keep_event_mask & (~nonzero_event_mask))
    )

    print("=" * 80)
    print("Done filtering NPZ by channel range and reindexing channels")
    print("=" * 80)
    print(f"Input file:  {input_path}")
    print(f"Output file: {output_path}")
    print(f"Global channel range kept: [{start}, {end})")
    print(f"Local output channels:      [0, {local_n_channels})")
    print(f"Full detector channels:     {total_n_channels}")
    print(f"Output n_channels:          {local_n_channels}")
    print(f"Throw-empty requested:      {throw_empty}")
    print(f"Run filter requested:       {run_filter}")
    print(f"Reverse run filter:         {reverse_run_filter}")
    if run_filter:
        run_label = "Runs removed" if reverse_run_filter else "Runs kept"
        print(f"{run_label + ':':28s}{requested_runs.tolist()}")
    print(f"Force-ratio requested:      {force_ratio}")
    print(f"Force-ratio effective:      {force_ratio_effective}")
    print(f"Force-ratio applied:        {force_ratio_applied}")
    print(f"Force-ratio best effort:    {best_effort_warning}")
    print(f"Random seed:                {random_seed}")
    print()
    print("Overall channel coverage:")
    print(
        f"  Full detector unique hit channels / all channels: "
        f"{full_unique_hit_channels} / {total_n_channels} = {full_overall_ratio:.8f}"
    )
    print(
        f"  Selected unique hit channels / selected channels: "
        f"{selected_unique_hit_channels} / {local_n_channels} = {selected_overall_ratio:.8f}"
    )
    print()
    print("Zero-selected-hit events:")
    print(f"  Total available:           {zero_selected_events_total}")
    print(f"  Kept in output:            {zero_selected_events_kept}")
    print()
    print(f"Original hits:               {original_hits}")
    print(f"Kept hits:                   {kept_hits}")
    print(f"Removed hits:                {original_hits - kept_hits}")
    print(f"Original events:             {original_events}")
    print(f"Events kept:                 {n_events_kept}")
    print(f"Events removed:              {n_events_removed}")

    if kept_hits > 0:
        print(
            f"Reindexed channel min/max:   "
            f"{int(selected_local_channels.min())} / {int(selected_local_channels.max())}"
        )

    print("=" * 80)

    return output_path


# ============================================================
# CLI
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Filter train, good, and bad sparse NPZ files to the same channel "
            "range, reindex channels to local [0, END-START), and write one "
            "sliced NPZ file for each input."
        )
    )

    parser.add_argument(
        "--train",
        "--train-input",
        "--train_input",
        dest="train_input",
        type=str,
        default=str(DEFAULT_TRAIN_NPZ_PATH),
        help="Path to the training NPZ file.",
    )
    parser.add_argument(
        "--good",
        "--good-input",
        "--good_input",
        dest="good_input",
        type=str,
        default=str(DEFAULT_GOOD_NPZ_PATH),
        help="Path to the good-test NPZ file.",
    )
    parser.add_argument(
        "--bad",
        "--bad-input",
        "--bad_input",
        dest="bad_input",
        type=str,
        default=str(DEFAULT_BAD_NPZ_PATH),
        help="Path to the bad-test NPZ file.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=DEFAULT_START,
        help="Inclusive global channel start.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=DEFAULT_END,
        help="Exclusive global channel end.",
    )
    parser.add_argument(
        "--total-n-channels",
        "--total_n_channels",
        dest="total_n_channels",
        type=int,
        default=DEFAULT_TOTAL_N_CHANNELS,
        help=(
            "Total number of detector channels used for the full-detector ratio. "
            "Default: 11276."
        ),
    )
    parser.add_argument(
        "--keep-empty",
        "--keep_empty",
        dest="keep_empty",
        action="store_true",
        help=(
            "Keep events with zero hits in the selected channel range. By default, "
            "empty selected-range events are removed for train, good, and bad inputs."
        ),
    )
    parser.add_argument(
        "--force-ratio",
        "--force_ratio",
        "--force-balance",
        "--force_balance",
        "--force",
        "-f",
        dest="force_ratio",
        action="store_true",
        help=(
            "Only meaningful when empty events are removed (the default). Try to keep the "
            "selected-range per-event channel-hit ratio closer to the "
            "full-detector ratio by preserving only a needed subset of "
            "zero-selected-hit events. The same setting is applied to all "
            "three inputs."
        ),
    )
    parser.add_argument(
        "--run-filter",
        "--run_filter",
        action="store_true",
        help=(
            "Enable evt_run filtering. By default, each input keeps only the runs "
            "supplied by its corresponding run-list option. Combine with "
            "--reverse-run-filter to remove the listed runs instead."
        ),
    )
    parser.add_argument(
        "--reverse-run-filter",
        "--reverse_run_filter",
        action="store_true",
        help=(
            "Invert --run-filter: remove events whose evt_run is listed in the "
            "corresponding run list and keep all other runs. This option requires "
            "--run-filter."
        ),
    )
    parser.add_argument(
        "--train-runs",
        "--train_runs",
        nargs="+",
        type=int,
        default=DEFAULT_TRAIN_RUNS_TO_KEEP,
        help="evt_run values to keep from the training NPZ when --run-filter is set.",
    )
    parser.add_argument(
        "--good-runs",
        "--good_runs",
        nargs="+",
        type=int,
        default=DEFAULT_GOOD_RUNS_TO_KEEP,
        help="evt_run values to keep from the good-test NPZ when --run-filter is set.",
    )
    parser.add_argument(
        "--bad-runs",
        "--bad_runs",
        nargs="+",
        type=int,
        default=DEFAULT_BAD_RUNS_TO_KEEP,
        help="evt_run values to keep from the bad-test NPZ when --run-filter is set.",
    )

    parser.add_argument(
        "--random-seed",
        "--random_seed",
        dest="random_seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=(
            "Random seed used when --force-ratio chooses a subset of "
            "zero-selected-hit events to preserve."
        ),
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    inputs = [
        ("TRAIN", Path(args.train_input), args.train_runs),
        ("GOOD", Path(args.good_input), args.good_runs),
        ("BAD", Path(args.bad_input), args.bad_runs),
    ]

    if args.reverse_run_filter and not args.run_filter:
        parser.error("--reverse-run-filter requires --run-filter.")

    if args.run_filter:
        missing_lists = [label for label, _, runs in inputs if not runs]
        if missing_lists:
            parser.error(
                "--run-filter requires a non-empty run list for every input. "
                "Missing: " + ", ".join(missing_lists)
            )

    resolved_inputs = [path.expanduser().resolve() for _, path, _ in inputs]
    if len(set(resolved_inputs)) != len(resolved_inputs):
        raise ValueError(
            "The train, good, and bad input paths must refer to three different files."
        )

    output_paths: list[tuple[str, Path]] = []

    print("=" * 80)
    print("Slicing train, good, and bad NPZ files")
    print("=" * 80)
    print(f"Channel range: [{args.start}, {args.end})")
    for label, path, runs in inputs:
        print(f"{label:5s} input: {path}")
        if args.run_filter:
            action = "remove" if args.reverse_run_filter else "keep"
            print(f"{label:5s} runs to {action}: {runs}")
    print("=" * 80)

    for index, (label, input_path, runs_to_keep) in enumerate(inputs, start=1):
        print()
        print("#" * 80)
        print(f"[{index}/3] Processing {label}: {input_path}")
        print("#" * 80)

        output_path = filter_npz_by_channel_range(
            input_path,
            args.start,
            args.end,
            total_n_channels=args.total_n_channels,
            throw_empty=not args.keep_empty,
            force_ratio=args.force_ratio,
            random_seed=args.random_seed,
            run_filter=args.run_filter,
            runs_to_keep=runs_to_keep,
            reverse_run_filter=args.reverse_run_filter,
        )
        output_paths.append((label, output_path))

    print()
    print("=" * 80)
    print("All three NPZ files were sliced successfully")
    print("=" * 80)
    for label, output_path in output_paths:
        print(f"{label:5s} output: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
