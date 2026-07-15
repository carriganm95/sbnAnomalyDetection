#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# User settings
# ============================================================

NPZ_PATH = Path(
    "/exp/sbnd/data/users/micarrig/DQM/tpc_data_v2/bad_runs.npz"
)

OUTPUT_PATH = Path(
    "/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection/graphing"
)

# Each entry is:
#   (window_size, stride)
WINDOW_SETTINGS = [
    (100, 100),
    (400, 100),
    (1000, 100),
]

# A window longer than this is checked for multiple runs.
LONG_WINDOW_THRESHOLD_DAYS = 1.0

HIST_BINS = 100

# Histogram x-axis upper bound.
# Example:
#   99.0  -> show up to the 99th percentile
#   99.5  -> show up to the 99.5th percentile
#   100.0 -> show the full distribution
X_AXIS_PERCENTILE = 99.0

OUTPUT_NAME = "between_event_time_histogram.png"


# ============================================================
# Time constants
# ============================================================

NS_PER_SECOND = 1_000_000_000.0
SECONDS_PER_MINUTE = 60.0
SECONDS_PER_DAY = 24.0 * 60.0 * 60.0

NS_PER_MINUTE = (
    NS_PER_SECOND
    * SECONDS_PER_MINUTE
)

NS_PER_DAY = (
    NS_PER_SECOND
    * SECONDS_PER_DAY
)


# ============================================================
# Time formatting
# ============================================================

def format_duration(seconds: float) -> str:
    """
    Format seconds as:

        days, HH:MM:SS
    """
    if not np.isfinite(seconds):
        return "nan"

    sign = "-" if seconds < 0 else ""
    seconds = abs(float(seconds))

    days = int(seconds // 86400)
    seconds %= 86400

    hours = int(seconds // 3600)
    seconds %= 3600

    minutes = int(seconds // 60)
    seconds %= 60

    return (
        f"{sign}{days} days, "
        f"{hours:02d}:{minutes:02d}:{seconds:09.6f}"
    )


# ============================================================
# Window construction
# ============================================================

def calculate_window_information(
    evt_time_ns: np.ndarray,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Form windows using the same index logic as:

        starts = range(
            0,
            n_events - window_size + 1,
            stride,
        )

    A window of N events contains:

        event[start]
        ...
        event[start + N - 1]

    Its elapsed time is:

        evt_time[start + N - 1]
        -
        evt_time[start]

    Returns
    -------
    starts
        Inclusive start index of each window.

    ends
        Inclusive end index of each window.

    durations_ns
        Elapsed time of each window in nanoseconds.
    """
    n_events = len(evt_time_ns)

    if window_size <= 0:
        raise ValueError(
            f"window_size must be positive, "
            f"got {window_size}"
        )

    if stride <= 0:
        raise ValueError(
            f"stride must be positive, "
            f"got {stride}"
        )

    if n_events < window_size:
        empty = np.empty(
            0,
            dtype=np.int64,
        )

        return (
            empty,
            empty,
            np.empty(
                0,
                dtype=np.float64,
            ),
        )

    starts = np.arange(
        0,
        n_events - window_size + 1,
        stride,
        dtype=np.int64,
    )

    ends = (
        starts
        + window_size
        - 1
    )

    durations_ns = (
        evt_time_ns[ends]
        - evt_time_ns[starts]
    )

    return (
        starts,
        ends,
        durations_ns,
    )


# ============================================================
# Summary printing
# ============================================================

def print_duration_summary(
    title: str,
    durations_ns: np.ndarray,
) -> None:
    """
    Print elapsed-time summary for a collection of windows.
    """
    print(title)

    if len(durations_ns) == 0:
        print("  No windows.")
        return

    durations_seconds = (
        durations_ns
        / NS_PER_SECOND
    )

    durations_minutes = (
        durations_ns
        / NS_PER_MINUTE
    )

    mean_minutes = float(
        np.mean(durations_minutes)
    )

    median_minutes = float(
        np.median(durations_minutes)
    )

    std_minutes = float(
        np.std(durations_minutes)
    )

    mean_seconds = float(
        np.mean(durations_seconds)
    )

    median_seconds = float(
        np.median(durations_seconds)
    )

    std_seconds = float(
        np.std(durations_seconds)
    )

    print(
        f"  Number of windows:  "
        f"{len(durations_ns):,}"
    )

    print(
        f"  Average time:        "
        f"{mean_minutes:.9f} min"
    )

    print(
        f"                       "
        f"{format_duration(mean_seconds)}"
    )

    print(
        f"  Median time:         "
        f"{median_minutes:.9f} min"
    )

    print(
        f"                       "
        f"{format_duration(median_seconds)}"
    )

    print(
        f"  Standard deviation:  "
        f"{std_minutes:.9f} min"
    )

    print(
        f"                       "
        f"{format_duration(std_seconds)}"
    )


# ============================================================
# Long-window run inspection
# ============================================================

def inspect_long_windows(
    evt_run: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    durations_ns: np.ndarray,
    threshold_days: float,
) -> np.ndarray:
    """
    Check windows longer than threshold_days.

    For each long window, inspect all evt_run values inside
    the window.

    Returns
    -------
    exclude_mask
        Boolean array with one entry per window.

        True means:

            window duration > threshold_days

        AND

            the window contains multiple evt_run values

        These are the windows excluded from the filtered
        timing summary.
    """
    threshold_ns = (
        threshold_days
        * NS_PER_DAY
    )

    exclude_mask = np.zeros(
        len(durations_ns),
        dtype=bool,
    )

    long_mask = (
        durations_ns
        > threshold_ns
    )

    long_indices = np.where(
        long_mask
    )[0]

    n_long = len(long_indices)

    n_single_run = 0
    n_multiple_runs = 0

    print()
    print(
        f"  Windows longer than "
        f"{threshold_days:g} day(s): "
        f"{n_long:,}"
    )

    for window_idx in long_indices:
        start = int(
            starts[window_idx]
        )

        end = int(
            ends[window_idx]
        )

        duration_ns = float(
            durations_ns[window_idx]
        )

        window_runs = evt_run[
            start:end + 1
        ]

        unique_runs = np.unique(
            window_runs
        )

        duration_seconds = (
            duration_ns
            / NS_PER_SECOND
        )

        if len(unique_runs) > 1:
            n_multiple_runs += 1

            exclude_mask[
                window_idx
            ] = True

            print()
            print(
                "  MULTIPLE RUNS FOUND"
            )

            print(
                f"    Window number:  "
                f"{window_idx}"
            )

            print(
                f"    Window indices: "
                f"{start} -> {end}"
            )

            print(
                f"    Duration:       "
                f"{format_duration(duration_seconds)}"
            )

            print(
                f"    Number of runs: "
                f"{len(unique_runs)}"
            )

            print(
                f"    Run numbers:    "
                f"{unique_runs.tolist()}"
            )

        else:
            n_single_run += 1

    print()
    print(
        "  Long-window run summary:"
    )

    print(
        f"    Single run:    "
        f"{n_single_run:,}"
    )

    print(
        f"    Multiple runs: "
        f"{n_multiple_runs:,}"
    )

    return exclude_mask


# ============================================================
# Main
# ============================================================

def main() -> None:
    npz_path = (
        NPZ_PATH
        .expanduser()
        .resolve()
    )

    output_dir = (
        OUTPUT_PATH
        .expanduser()
        .resolve()
    )

    if not npz_path.exists():
        raise FileNotFoundError(
            f"NPZ file does not exist:\n"
            f"{npz_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Loading: {npz_path}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    with np.load(
        npz_path,
        allow_pickle=False,
    ) as data:

        required_keys = [
            "evt_time",
            "evt_run",
        ]

        missing_keys = [
            key
            for key in required_keys
            if key not in data
        ]

        if missing_keys:
            raise KeyError(
                "The NPZ file is missing required keys:\n  - "
                + "\n  - ".join(missing_keys)
                + "\n\nAvailable keys:\n  - "
                + "\n  - ".join(data.files)
            )

        evt_time_ns = np.asarray(
            data["evt_time"],
            dtype=np.float64,
        ).reshape(-1)

        evt_run = np.asarray(
            data["evt_run"],
        ).reshape(-1)

    print(
        f"Number of events: "
        f"{len(evt_time_ns):,}"
    )

    print(
        f"evt_time dtype after loading: "
        f"{evt_time_ns.dtype}"
    )

    print(
        f"evt_run dtype after loading: "
        f"{evt_run.dtype}"
    )

    # --------------------------------------------------------
    # Validate arrays
    # --------------------------------------------------------

    if len(evt_run) != len(evt_time_ns):
        raise ValueError(
            "evt_run and evt_time have different lengths: "
            f"{len(evt_run):,} vs "
            f"{len(evt_time_ns):,}"
        )

    finite_mask = np.isfinite(
        evt_time_ns
    )

    if not np.all(finite_mask):
        n_bad = int(
            np.sum(~finite_mask)
        )

        raise ValueError(
            f"evt_time contains "
            f"{n_bad:,} non-finite values."
        )

    if len(evt_time_ns) < 2:
        raise ValueError(
            "Need at least two events to "
            "calculate time intervals."
        )

    # --------------------------------------------------------
    # Between-stored-event intervals
    # --------------------------------------------------------

    interval_ns = np.diff(
        evt_time_ns
    )

    interval_seconds = (
        interval_ns
        / NS_PER_SECOND
    )

    interval_minutes = (
        interval_ns
        / NS_PER_MINUTE
    )

    n_negative = int(
        np.sum(interval_ns < 0)
    )

    n_zero = int(
        np.sum(interval_ns == 0)
    )

    n_positive = int(
        np.sum(interval_ns > 0)
    )

    print()
    print("=" * 70)
    print(
        "BETWEEN-STORED-EVENT TIME INTERVALS"
    )
    print("=" * 70)

    print(
        f"Total intervals:   "
        f"{len(interval_ns):,}"
    )

    print(
        f"Negative:          "
        f"{n_negative:,}"
    )

    print(
        f"Zero:              "
        f"{n_zero:,}"
    )

    print(
        f"Positive:          "
        f"{n_positive:,}"
    )

    print()
    print(
        "Interval statistics:"
    )

    stats = [
        (
            "mean",
            np.mean(interval_seconds),
        ),
        (
            "std",
            np.std(interval_seconds),
        ),
        (
            "min",
            np.min(interval_seconds),
        ),
        (
            "median",
            np.median(interval_seconds),
        ),
        (
            "p90",
            np.percentile(
                interval_seconds,
                90,
            ),
        ),
        (
            "p95",
            np.percentile(
                interval_seconds,
                95,
            ),
        ),
        (
            "p99",
            np.percentile(
                interval_seconds,
                99,
            ),
        ),
        (
            "max",
            np.max(interval_seconds),
        ),
    ]

    for name, value in stats:
        print(
            f"  {name:8s}: "
            f"{value:15.9f} s"
            f"    "
            f"({format_duration(value)})"
        )

    # --------------------------------------------------------
    # Window elapsed times
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "WINDOW ELAPSED TIMES"
    )
    print("=" * 70)

    for window_size, stride in WINDOW_SETTINGS:
        print()
        print("#" * 70)

        print(
            f"WINDOW SIZE = {window_size}, "
            f"STRIDE = {stride}"
        )

        print("#" * 70)
        print()

        (
            starts,
            ends,
            durations_ns,
        ) = calculate_window_information(
            evt_time_ns=evt_time_ns,
            window_size=window_size,
            stride=stride,
        )

        if len(durations_ns) == 0:
            print(
                "Not enough events to form "
                "this window size."
            )

            continue

        # ----------------------------------------------------
        # Full summary
        # ----------------------------------------------------

        print_duration_summary(
            title="ALL WINDOWS:",
            durations_ns=durations_ns,
        )

        # ----------------------------------------------------
        # Check long windows for multiple runs
        # ----------------------------------------------------

        exclude_mask = inspect_long_windows(
            evt_run=evt_run,
            starts=starts,
            ends=ends,
            durations_ns=durations_ns,
            threshold_days=LONG_WINDOW_THRESHOLD_DAYS,
        )

        # ----------------------------------------------------
        # Summary excluding long multi-run windows
        # ----------------------------------------------------

        filtered_durations_ns = durations_ns[
            ~exclude_mask
        ]

        n_excluded = int(
            np.sum(exclude_mask)
        )

        print()
        print(
            "-" * 70
        )

        print_duration_summary(
            title=(
                "EXCLUDING LONG MULTIPLE-RUN WINDOWS:"
            ),
            durations_ns=filtered_durations_ns,
        )

        print(
            f"  Excluded windows:   "
            f"{n_excluded:,}"
        )

        print(
            f"  Exclusion rule:     "
            f"> {LONG_WINDOW_THRESHOLD_DAYS:g} day(s) "
            f"AND multiple evt_run values"
        )

        print(
            "-" * 70
        )

    # --------------------------------------------------------
    # Histogram
    # --------------------------------------------------------

    if not (
        0
        < X_AXIS_PERCENTILE
        <= 100
    ):
        raise ValueError(
            "X_AXIS_PERCENTILE must satisfy "
            "0 < percentile <= 100, got "
            f"{X_AXIS_PERCENTILE}"
        )

    positive_intervals_minutes = (
        interval_minutes[
            interval_minutes > 0
        ]
    )

    if (
        len(
            positive_intervals_minutes
        )
        == 0
    ):
        raise ValueError(
            "No positive between-event "
            "time intervals found."
        )

    x_upper = float(
        np.percentile(
            positive_intervals_minutes,
            X_AXIS_PERCENTILE,
        )
    )

    visible_intervals = (
        positive_intervals_minutes[
            positive_intervals_minutes
            <= x_upper
        ]
    )

    n_removed = (
        len(
            positive_intervals_minutes
        )
        - len(
            visible_intervals
        )
    )

    print()
    print("=" * 70)
    print(
        "HISTOGRAM SETTINGS"
    )
    print("=" * 70)

    print(
        f"X-axis percentile: "
        f"p{X_AXIS_PERCENTILE:g}"
    )

    print(
        f"X-axis upper bound: "
        f"{x_upper:.9f} minutes"
    )

    print(
        f"Positive intervals shown: "
        f"{len(visible_intervals):,}"
    )

    print(
        f"Intervals above x-limit: "
        f"{n_removed:,}"
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.hist(
        visible_intervals,
        bins=HIST_BINS,
        range=(
            0,
            x_upper,
        ),
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlim(
        0,
        x_upper,
    )

    ax.set_xlabel(
        "Time Between Consecutive "
        "Stored Events [minutes]"
    )

    ax.set_ylabel(
        "Number of Event Pairs "
        "[log scale]"
    )

    ax.set_title(
        "Distribution of Time Between "
        "Consecutive Stored Events\n"
        f"X-axis limited to "
        f"p{X_AXIS_PERCENTILE:g}"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    output_path = (
        output_dir
        / OUTPUT_NAME
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print()
    print("=" * 70)

    print(
        "Saved histogram to:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()