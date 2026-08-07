#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np


# ============================================================
# Defaults
# ============================================================

DEFAULT_TIME_KEY = "evt_time"
DEFAULT_RUN_KEY = "evt_run"

# evt_time is stored in nanoseconds.
DEFAULT_TIME_SCALE = 1.0e9

# Timed-window settings to inspect.
DEFAULT_WINDOW_TIME_SIZE = 1000.0
DEFAULT_WINDOW_TIME_STRIDE = 15.0
DEFAULT_MAX_EVENTS_PER_WINDOW = 1000

# Event-count window to compare against.
DEFAULT_EVENT_WINDOW_SIZE = 1000
DEFAULT_EVENT_STRIDE = 10


# ============================================================
# Formatting helpers
# ============================================================

def format_duration(seconds: float) -> str:
    """Format a duration in a readable form."""
    if not np.isfinite(seconds):
        return "nan"

    sign = "-" if seconds < 0 else ""
    seconds = abs(float(seconds))

    days = int(seconds // 86400)
    seconds -= days * 86400

    hours = int(seconds // 3600)
    seconds -= hours * 3600

    minutes = int(seconds // 60)
    seconds -= minutes * 60

    if days > 0:
        return (
            f"{sign}{days} d {hours:02d} h "
            f"{minutes:02d} min {seconds:06.3f} s"
        )

    if hours > 0:
        return (
            f"{sign}{hours} h {minutes:02d} min "
            f"{seconds:06.3f} s"
        )

    if minutes > 0:
        return f"{sign}{minutes} min {seconds:06.3f} s"

    return f"{sign}{seconds:.6f} s"


def print_percentiles(
    values: np.ndarray,
    *,
    label: str,
    duration_values: bool = False,
) -> None:
    """Print common percentiles for a one-dimensional array."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    print(f"\n{label}")

    if values.size == 0:
        print("  No finite values.")
        return

    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    results = np.percentile(values, percentiles)

    for percentile, value in zip(percentiles, results):
        if duration_values:
            print(
                f"  p{percentile:>3}: "
                f"{value:14.6f} s  "
                f"({format_duration(value)})"
            )
        else:
            print(f"  p{percentile:>3}: {value:14.6f}")

    mean = float(np.mean(values))
    std = float(np.std(values))

    if duration_values:
        print(
            f"  mean: {mean:14.6f} s  "
            f"({format_duration(mean)})"
        )
        print(
            f"  std:  {std:14.6f} s  "
            f"({format_duration(std)})"
        )
    else:
        print(f"  mean: {mean:14.6f}")
        print(f"  std:  {std:14.6f}")


# ============================================================
# Input discovery
# ============================================================

def discover_npz_files(path: Path, recursive: bool) -> list[Path]:
    """Return NPZ files from a file or directory."""
    path = path.expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if path.is_file():
        if path.suffix.lower() != ".npz":
            raise ValueError(f"Input file is not an NPZ file: {path}")
        return [path]

    pattern = "**/*.npz" if recursive else "*.npz"
    files = sorted(path.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No .npz files found under: {path}"
        )

    return files


# ============================================================
# Loading
# ============================================================

def load_arrays(
    npz_files: Iterable[Path],
    *,
    time_key: str,
    run_key: str,
    time_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and concatenate evt_time and evt_run arrays.

    Returns
    -------
    times_seconds:
        Raw timestamps converted to seconds.

    runs:
        Run number per event. When absent, filled with -1.

    source_file_index:
        Input-file index per event.
    """
    time_parts: list[np.ndarray] = []
    run_parts: list[np.ndarray] = []
    file_index_parts: list[np.ndarray] = []

    expected_time_dtype = None

    for file_index, path in enumerate(npz_files):
        print(f"Loading: {path}")

        with np.load(path, allow_pickle=True) as data:
            if time_key not in data.files:
                print(
                    f"  WARNING: missing '{time_key}'; skipping."
                )
                continue

            raw_times = np.asarray(data[time_key]).reshape(-1)

            if expected_time_dtype is None:
                expected_time_dtype = raw_times.dtype

            if raw_times.size == 0:
                print("  WARNING: empty timestamp array; skipping.")
                continue

            if run_key in data.files:
                runs = np.asarray(data[run_key]).reshape(-1)

                if runs.size != raw_times.size:
                    raise ValueError(
                        f"{path}: '{run_key}' length {runs.size} "
                        f"does not match '{time_key}' length "
                        f"{raw_times.size}."
                    )
            else:
                runs = np.full(
                    raw_times.size,
                    -1,
                    dtype=np.int64,
                )

            times_seconds = raw_times.astype(
                np.float64,
                copy=False,
            ) / float(time_scale)

            finite_mask = np.isfinite(times_seconds)

            if not np.all(finite_mask):
                bad_count = int(np.sum(~finite_mask))
                print(
                    f"  WARNING: removing {bad_count} non-finite "
                    f"timestamps."
                )

                times_seconds = times_seconds[finite_mask]
                runs = runs[finite_mask]

            time_parts.append(times_seconds)
            run_parts.append(runs.astype(np.int64, copy=False))
            file_index_parts.append(
                np.full(
                    times_seconds.size,
                    file_index,
                    dtype=np.int64,
                )
            )

            print(f"  Events loaded: {times_seconds.size:,}")

    if not time_parts:
        raise ValueError(
            f"No usable '{time_key}' arrays were found."
        )

    return (
        np.concatenate(time_parts),
        np.concatenate(run_parts),
        np.concatenate(file_index_parts),
    )


# ============================================================
# Stitched timeline
# ============================================================

def build_stitched_timeline(
    raw_times_seconds: np.ndarray,
    runs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the same type of continuous stitched timeline used by timed windows.

    A new segment begins when:

    1. the run number changes, or
    2. the raw timestamp moves backward.

    Within each segment, relative raw timing is preserved. The new segment
    begins at the final stitched timestamp of the preceding segment, so no
    artificial time gap is inserted.

    Returns
    -------
    stitched_times:
        Continuous virtual timestamps in seconds, starting at zero.

    segment_start_mask:
        Boolean array marking the first event of each segment.
    """
    raw_times_seconds = np.asarray(
        raw_times_seconds,
        dtype=np.float64,
    )
    runs = np.asarray(runs)

    n_events = raw_times_seconds.size

    if n_events == 0:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=bool),
        )

    segment_start_mask = np.zeros(n_events, dtype=bool)
    segment_start_mask[0] = True

    if n_events > 1:
        run_changed = runs[1:] != runs[:-1]
        time_moved_backward = (
            raw_times_seconds[1:] < raw_times_seconds[:-1]
        )

        segment_start_mask[1:] = (
            run_changed | time_moved_backward
        )

    stitched = np.zeros(n_events, dtype=np.float64)

    segment_starts = np.flatnonzero(segment_start_mask)
    segment_ends = np.r_[
        segment_starts[1:],
        n_events,
    ]

    virtual_offset = 0.0

    for start, end in zip(segment_starts, segment_ends):
        segment_raw = raw_times_seconds[start:end]
        relative = segment_raw - segment_raw[0]

        # Keep relative timing but protect against tiny local numerical
        # backward fluctuations.
        relative = np.maximum.accumulate(relative)

        stitched[start:end] = virtual_offset + relative

        if end > start:
            virtual_offset = float(stitched[end - 1])

    return stitched, segment_start_mask


# ============================================================
# Sliding-window calculations
# ============================================================

def sliding_time_window_counts_by_run(
    raw_times_seconds: np.ndarray,
    runs: np.ndarray,
    *,
    duration_seconds: float,
    stride_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count events in sliding timed windows without crossing run boundaries.

    Each run is processed independently. No window can contain events from
    more than one run.

    Returns
    -------
    window_runs:
        Run number associated with each timed window.

    window_starts:
        Window start time relative to the beginning of that run, in seconds.

    window_counts:
        Number of real events contained in each timed window.
    """
    raw_times_seconds = np.asarray(
        raw_times_seconds,
        dtype=np.float64,
    )
    runs = np.asarray(
        runs,
        dtype=np.int64,
    )

    if raw_times_seconds.shape != runs.shape:
        raise ValueError(
            "raw_times_seconds and runs must have the same shape: "
            f"{raw_times_seconds.shape} != {runs.shape}"
        )

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive.")

    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive.")

    all_window_runs: list[np.ndarray] = []
    all_window_starts: list[np.ndarray] = []
    all_window_counts: list[np.ndarray] = []

    # Preserve the order in which runs first appear.
    _, first_indices = np.unique(
        runs,
        return_index=True,
    )
    ordered_runs = runs[np.sort(first_indices)]

    for run in ordered_runs:
        run_mask = runs == run
        run_times = raw_times_seconds[run_mask]

        if run_times.size == 0:
            continue

        # Preserve input event order. If timestamps reset backward inside a run,
        # split that run into independent timestamp segments as well.
        segment_start_mask = np.zeros(
            run_times.size,
            dtype=bool,
        )
        segment_start_mask[0] = True

        if run_times.size > 1:
            segment_start_mask[1:] = (
                run_times[1:] < run_times[:-1]
            )

        segment_starts = np.flatnonzero(
            segment_start_mask
        )
        segment_ends = np.r_[
            segment_starts[1:],
            run_times.size,
        ]

        for segment_start, segment_end in zip(
            segment_starts,
            segment_ends,
        ):
            segment_times = run_times[
                segment_start:segment_end
            ]

            if segment_times.size == 0:
                continue

            # Use time relative to the beginning of this run segment.
            relative_times = (
                segment_times - segment_times[0]
            )

            # Protect searchsorted against small non-monotonic fluctuations.
            relative_times = np.maximum.accumulate(
                relative_times
            )

            segment_duration = float(
                relative_times[-1]
            )

            if segment_duration <= 0:
                starts = np.array(
                    [0.0],
                    dtype=np.float64,
                )
            else:
                starts = np.arange(
                    0.0,
                    segment_duration + stride_seconds,
                    stride_seconds,
                    dtype=np.float64,
                )

            ends = starts + duration_seconds

            left = np.searchsorted(
                relative_times,
                starts,
                side="left",
            )
            right = np.searchsorted(
                relative_times,
                ends,
                side="left",
            )

            counts = (
                right - left
            ).astype(np.int64)

            all_window_runs.append(
                np.full(
                    starts.size,
                    int(run),
                    dtype=np.int64,
                )
            )
            all_window_starts.append(starts)
            all_window_counts.append(counts)

    if not all_window_counts:
        return (
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.int64),
        )

    return (
        np.concatenate(all_window_runs),
        np.concatenate(all_window_starts),
        np.concatenate(all_window_counts),
    )


def event_count_window_durations(
    stitched_times: np.ndarray,
    *,
    window_size: int,
    stride: int,
) -> np.ndarray:
    """Calculate durations of fixed-event-count windows."""
    stitched_times = np.asarray(
        stitched_times,
        dtype=np.float64,
    )

    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    if stride <= 0:
        raise ValueError("stride must be positive.")

    n_events = stitched_times.size

    if n_events < window_size:
        return np.asarray([], dtype=np.float64)

    starts = np.arange(
        0,
        n_events - window_size + 1,
        stride,
        dtype=np.int64,
    )

    ends = starts + window_size - 1

    return stitched_times[ends] - stitched_times[starts]


# ============================================================
# Reporting
# ============================================================

def report_raw_timestamp_statistics(
    times_seconds: np.ndarray,
    runs: np.ndarray,
    source_file_index: np.ndarray,
) -> None:
    """Print statistics for the original timestamp ordering."""
    n_events = times_seconds.size

    print("\n" + "=" * 80)
    print("RAW EVT_TIME SUMMARY")
    print("=" * 80)

    print(f"Number of events:          {n_events:,}")
    print(f"Number of input files:     {np.unique(source_file_index).size:,}")
    print(f"Number of unique runs:     {np.unique(runs).size:,}")
    print(f"Minimum raw timestamp:     {np.min(times_seconds):.9f} s")
    print(f"Maximum raw timestamp:     {np.max(times_seconds):.9f} s")
    print(
        "Raw max-min span:         "
        f"{format_duration(np.max(times_seconds) - np.min(times_seconds))}"
    )

    if n_events < 2:
        return

    differences = np.diff(times_seconds)
    run_changes = runs[1:] != runs[:-1]
    backward = differences < 0
    equal = differences == 0
    positive = differences > 0

    print(f"\nConsecutive pairs:         {differences.size:,}")
    print(
        f"Positive differences:      {np.sum(positive):,} "
        f"({100.0 * np.mean(positive):.3f}%)"
    )
    print(
        f"Zero differences:          {np.sum(equal):,} "
        f"({100.0 * np.mean(equal):.3f}%)"
    )
    print(
        f"Backward differences:      {np.sum(backward):,} "
        f"({100.0 * np.mean(backward):.3f}%)"
    )
    print(
        f"Run changes:               {np.sum(run_changes):,} "
        f"({100.0 * np.mean(run_changes):.3f}%)"
    )
    print(
        "Backward without run change: "
        f"{np.sum(backward & ~run_changes):,}"
    )

    print_percentiles(
        differences[positive],
        label="Positive consecutive-event time differences",
        duration_values=True,
    )

    if np.any(backward):
        print_percentiles(
            differences[backward],
            label="Backward timestamp jumps",
            duration_values=True,
        )

    thresholds = [
        1.0,
        5.0,
        10.0,
        30.0,
        60.0,
        300.0,
        600.0,
        3600.0,
        86400.0,
    ]

    print("\nLarge positive consecutive-event gaps")

    for threshold in thresholds:
        count = int(np.sum(differences > threshold))
        fraction = (
            100.0 * count / differences.size
            if differences.size
            else 0.0
        )

        print(
            f"  > {format_duration(threshold):>22}: "
            f"{count:10,}  ({fraction:8.4f}%)"
        )


def report_stitched_statistics(
    stitched_times: np.ndarray,
    segment_start_mask: np.ndarray,
) -> None:
    """Print statistics for the stitched virtual timeline."""
    print("\n" + "=" * 80)
    print("STITCHED TIMELINE SUMMARY")
    print("=" * 80)

    if stitched_times.size == 0:
        print("No events.")
        return

    n_segments = int(np.sum(segment_start_mask))
    duration = float(stitched_times[-1] - stitched_times[0])

    print(f"Number of stitched segments: {n_segments:,}")
    print(f"Total stitched duration:      {format_duration(duration)}")
    print(f"Total stitched seconds:       {duration:.6f}")

    if duration > 0:
        event_rate = stitched_times.size / duration
        print(f"Average event rate:            {event_rate:.6f} events/s")
        print(
            "Average event interval:        "
            f"{format_duration(1.0 / event_rate)}"
        )


def report_per_run_statistics(
    raw_times_seconds: np.ndarray,
    runs: np.ndarray,
) -> None:
    """Print a compact summary for each run."""
    print("\n" + "=" * 80)
    print("PER-RUN SUMMARY")
    print("=" * 80)

    unique_runs = np.unique(runs)

    header = (
        f"{'run':>10} "
        f"{'events':>12} "
        f"{'segments':>10} "
        f"{'duration':>18} "
        f"{'median dt':>15} "
        f"{'mean dt':>15}"
    )
    print(header)
    print("-" * len(header))

    for run in unique_runs:
        indices = np.flatnonzero(runs == run)
        run_times = raw_times_seconds[indices]

        if run_times.size <= 1:
            duration = 0.0
            median_dt = math.nan
            mean_dt = math.nan
            n_segments = 1
        else:
            differences = np.diff(run_times)
            segment_breaks = differences < 0
            n_segments = 1 + int(np.sum(segment_breaks))

            # Sum only forward movement within timestamp segments.
            duration = float(
                np.sum(differences[differences >= 0])
            )

            positive = differences[differences > 0]

            if positive.size:
                median_dt = float(np.median(positive))
                mean_dt = float(np.mean(positive))
            else:
                median_dt = math.nan
                mean_dt = math.nan

        print(
            f"{int(run):>10d} "
            f"{run_times.size:>12,} "
            f"{n_segments:>10,} "
            f"{format_duration(duration):>18} "
            f"{format_duration(median_dt):>15} "
            f"{format_duration(mean_dt):>15}"
        )


def report_timed_window_counts(
    starts: np.ndarray,
    counts: np.ndarray,
    *,
    duration_seconds: float,
    stride_seconds: float,
    maximum_events: int,
) -> None:
    """Print event-count statistics for fixed-duration sliding windows."""
    print("\n" + "=" * 80)
    print("TIMED-WINDOW EVENT COUNT SUMMARY")
    print("=" * 80)

    print(
        f"Window duration:              "
        f"{duration_seconds:.6f} s "
        f"({format_duration(duration_seconds)})"
    )
    print(
        f"Window stride:                "
        f"{stride_seconds:.6f} s "
        f"({format_duration(stride_seconds)})"
    )
    print(f"Maximum selected events:      {maximum_events:,}")
    print(f"Number of timed windows:      {counts.size:,}")

    if counts.size == 0:
        return

    print_percentiles(
        counts,
        label="Real event count per timed window",
        duration_values=False,
    )

    empty = counts == 0
    underfilled = counts < maximum_events
    exactly_full = counts == maximum_events
    oversubscribed = counts > maximum_events

    print("\nTimed-window fill categories")
    print(
        f"  Empty:                 {np.sum(empty):10,} "
        f"({100.0 * np.mean(empty):8.3f}%)"
    )
    print(
        f"  Fewer than {maximum_events:4d}:    "
        f"{np.sum(underfilled):10,} "
        f"({100.0 * np.mean(underfilled):8.3f}%)"
    )
    print(
        f"  Exactly {maximum_events:4d}:       "
        f"{np.sum(exactly_full):10,} "
        f"({100.0 * np.mean(exactly_full):8.3f}%)"
    )
    print(
        f"  More than {maximum_events:4d}:     "
        f"{np.sum(oversubscribed):10,} "
        f"({100.0 * np.mean(oversubscribed):8.3f}%)"
    )

    thresholds = sorted(
        set(
            [
                1,
                10,
                25,
                50,
                100,
                200,
                300,
                maximum_events,
            ]
        )
    )

    print("\nUnderfilled-window fractions")

    for threshold in thresholds:
        below = counts < threshold
        print(
            f"  count < {threshold:4d}: "
            f"{np.sum(below):10,} / {counts.size:,} "
            f"({100.0 * np.mean(below):8.3f}%)"
        )

    selected_counts = np.minimum(
        counts,
        maximum_events,
    )
    padded_positions = maximum_events - selected_counts

    print("\nEffective fixed-size model input")
    print(
        f"  Mean real selected events:   "
        f"{np.mean(selected_counts):.3f}"
    )
    print(
        f"  Median real selected events: "
        f"{np.median(selected_counts):.3f}"
    )
    print(
        f"  Mean padded positions:       "
        f"{np.mean(padded_positions):.3f}"
    )
    print(
        f"  Median padded positions:     "
        f"{np.median(padded_positions):.3f}"
    )
    print(
        f"  Overall padded fraction:     "
        f"{100.0 * np.sum(padded_positions) / (counts.size * maximum_events):.3f}%"
    )

    if np.any(empty):
        first_empty = np.flatnonzero(empty)[:10]

        print("\nFirst empty timed windows")
        for index in first_empty:
            print(
                f"  window={index:8d} "
                f"start={starts[index]:14.3f} s "
                f"end={starts[index] + duration_seconds:14.3f} s"
            )


def report_event_window_durations(
    durations: np.ndarray,
    *,
    window_size: int,
    stride: int,
) -> None:
    """Print duration statistics for fixed-event-count windows."""
    print("\n" + "=" * 80)
    print("FIXED-EVENT-COUNT WINDOW DURATION SUMMARY")
    print("=" * 80)

    print(f"Event window size:          {window_size:,}")
    print(f"Event stride:               {stride:,}")
    print(f"Number of event windows:    {durations.size:,}")

    print_percentiles(
        durations,
        label=(
            f"Elapsed duration of each {window_size}-event window"
        ),
        duration_values=True,
    )

    if durations.size:
        shorter = durations < DEFAULT_WINDOW_TIME_SIZE
        longer = durations > DEFAULT_WINDOW_TIME_SIZE

        print(
            f"\nWindows shorter than "
            f"{DEFAULT_WINDOW_TIME_SIZE:.0f} s: "
            f"{np.sum(shorter):,} "
            f"({100.0 * np.mean(shorter):.3f}%)"
        )
        print(
            f"Windows longer than "
            f"{DEFAULT_WINDOW_TIME_SIZE:.0f} s:  "
            f"{np.sum(longer):,} "
            f"({100.0 * np.mean(longer):.3f}%)"
        )


# ============================================================
# Optional CSV output
# ============================================================

def save_timed_window_csv(
    output_path: Path,
    window_runs: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    *,
    duration_seconds: float,
    maximum_events: int,
) -> None:
    """Save one row per run-contained timed window."""
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    window_runs = np.asarray(
        window_runs,
        dtype=np.int64,
    )
    starts = np.asarray(
        starts,
        dtype=np.float64,
    )
    counts = np.asarray(
        counts,
        dtype=np.int64,
    )

    if not (
        window_runs.size
        == starts.size
        == counts.size
    ):
        raise ValueError(
            "window_runs, starts, and counts must have "
            "the same length."
        )

    selected_counts = np.minimum(
        counts,
        maximum_events,
    )
    padded_counts = (
        maximum_events - selected_counts
    )

    table = np.column_stack(
        [
            np.arange(
                counts.size,
                dtype=np.int64,
            ),
            window_runs,
            starts,
            starts + duration_seconds,
            counts,
            selected_counts,
            padded_counts,
        ]
    )

    header = (
        "window_index,run,"
        "time_window_start_sec,"
        "time_window_end_sec,"
        "real_event_count,"
        "selected_event_count,"
        "padded_event_count"
    )

    np.savetxt(
        output_path,
        table,
        delimiter=",",
        header=header,
        comments="",
        fmt=[
            "%d",
            "%d",
            "%.9f",
            "%.9f",
            "%d",
            "%d",
            "%d",
        ],
    )

    print(
        f"\nSaved timed-window table: "
        f"{output_path}"
    )


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze evt_time values in one NPZ file or a directory "
            "of NPZ files."
        )
    )

    parser.add_argument(
        "path",
        type=Path,
        help="Path to an NPZ file or a directory containing NPZ files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories recursively for NPZ files.",
    )
    parser.add_argument(
        "--time-key",
        default=DEFAULT_TIME_KEY,
        help=f"Timestamp array key. Default: {DEFAULT_TIME_KEY}",
    )
    parser.add_argument(
        "--run-key",
        default=DEFAULT_RUN_KEY,
        help=f"Run-number array key. Default: {DEFAULT_RUN_KEY}",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=DEFAULT_TIME_SCALE,
        help=(
            "Raw timestamp units per second. Use 1e9 for nanoseconds. "
            f"Default: {DEFAULT_TIME_SCALE:g}"
        ),
    )
    parser.add_argument(
        "--window-time-size",
        type=float,
        default=DEFAULT_WINDOW_TIME_SIZE,
        help="Timed-window duration in seconds. Default: 600",
    )
    parser.add_argument(
        "--window-time-stride",
        type=float,
        default=DEFAULT_WINDOW_TIME_STRIDE,
        help="Timed-window stride in seconds. Default: 15",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=DEFAULT_MAX_EVENTS_PER_WINDOW,
        help="Maximum selected events per timed window. Default: 400",
    )
    parser.add_argument(
        "--event-window-size",
        type=int,
        default=DEFAULT_EVENT_WINDOW_SIZE,
        help="Fixed-event-count comparison window size. Default: 400",
    )
    parser.add_argument(
        "--event-stride",
        type=int,
        default=DEFAULT_EVENT_STRIDE,
        help="Fixed-event-count comparison stride. Default: 10",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV output containing the event count and padding "
            "for every timed window."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.time_scale <= 0:
        raise ValueError("--time-scale must be positive.")

    if args.max_events <= 0:
        raise ValueError("--max-events must be positive.")

    npz_files = discover_npz_files(
        args.path,
        recursive=args.recursive,
    )

    print("=" * 80)
    print("INPUT")
    print("=" * 80)
    print(f"Input path:       {args.path.expanduser().resolve()}")
    print(f"NPZ files found:  {len(npz_files):,}")
    print(f"Timestamp key:    {args.time_key}")
    print(f"Run key:          {args.run_key}")
    print(f"Raw units/second: {args.time_scale:g}")

    raw_times_seconds, runs, file_indices = load_arrays(
        npz_files,
        time_key=args.time_key,
        run_key=args.run_key,
        time_scale=args.time_scale,
    )

    report_raw_timestamp_statistics(
        raw_times_seconds,
        runs,
        file_indices,
    )

    report_per_run_statistics(
        raw_times_seconds,
        runs,
    )

    stitched_times, segment_start_mask = build_stitched_timeline(
        raw_times_seconds,
        runs,
    )

    report_stitched_statistics(
        stitched_times,
        segment_start_mask,
    )

    timed_runs, timed_starts, timed_counts = (
        sliding_time_window_counts_by_run(
            raw_times_seconds,
            runs,
            duration_seconds=args.window_time_size,
            stride_seconds=args.window_time_stride,
        )
    )

    report_timed_window_counts(
        timed_starts,
        timed_counts,
        duration_seconds=args.window_time_size,
        stride_seconds=args.window_time_stride,
        maximum_events=args.max_events,
    )

    event_durations = event_count_window_durations(
        stitched_times,
        window_size=args.event_window_size,
        stride=args.event_stride,
    )

    report_event_window_durations(
        event_durations,
        window_size=args.event_window_size,
        stride=args.event_stride,
    )

    if args.output_csv is not None:
        save_timed_window_csv(
            args.output_csv,
            timed_runs,
            timed_starts,
            timed_counts,
            duration_seconds=args.window_time_size,
            maximum_events=args.max_events,
    )


if __name__ == "__main__":
    main()