#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# User settings
# ============================================================

# This may be either:
#   1. One .npz file
#   2. A directory containing .npz files
INPUT_PATH = Path(
    "/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection/scratch/bad_events_test.npz"
)

OUTPUT_DIR = Path(
    "/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection/graphing/plot_time_window_scatter_output"
)

# Timed-window durations to test, in seconds.
TIME_WINDOW_SIZES_SECONDS = [
    1800
]

# Distance between consecutive window starts, in seconds.
TIME_WINDOW_STRIDE_SECONDS = 15.0

# True:
#   Do not include completely empty time windows in the plot.
#
# False:
#   Include empty windows as points at y = 0.
SKIP_EMPTY_WINDOWS = False

# Scatter-plot appearance.
POINT_SIZE = 12
POINT_ALPHA = 0.45
FIGURE_WIDTH = 14
FIGURE_HEIGHT = 6
DPI = 200

# Add a small horizontal displacement so overlapping points from the same run
# are easier to see. Run labels on the x-axis remain unchanged.
USE_HORIZONTAL_JITTER = True
JITTER_WIDTH = 0.18
RANDOM_SEED = 42


# ============================================================
# Input discovery
# ============================================================

def discover_npz_files(input_path: Path) -> list[Path]:
    """Return the NPZ files represented by an input file or directory."""
    input_path = input_path.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input path does not exist:\n{input_path}"
        )

    if input_path.is_file():
        if input_path.suffix.lower() != ".npz":
            raise ValueError(
                f"Input file is not an NPZ file:\n{input_path}"
            )

        return [input_path]

    npz_files = sorted(input_path.glob("*.npz"))

    if not npz_files:
        raise FileNotFoundError(
            f"No .npz files found directly inside:\n{input_path}"
        )

    return npz_files


# ============================================================
# Data loading
# ============================================================

def load_event_metadata(
    npz_files: list[Path],
) -> tuple[np.ndarray, np.ndarray]:
    """Load evt_time and evt_run from one or more NPZ files."""
    all_times: list[np.ndarray] = []
    all_runs: list[np.ndarray] = []

    for path in npz_files:
        print(f"Loading: {path}")

        with np.load(path, allow_pickle=False) as data:
            required_keys = [
                "evt_time",
                "evt_run",
            ]

            missing = [
                key
                for key in required_keys
                if key not in data
            ]

            if missing:
                raise KeyError(
                    f"{path} is missing required arrays: {missing}\n"
                    f"Available arrays: {data.files}"
                )

            evt_time = np.asarray(
                data["evt_time"],
                dtype=np.float64,
            ).reshape(-1)

            evt_run = np.asarray(
                data["evt_run"],
                dtype=np.int64,
            ).reshape(-1)

        if evt_time.size != evt_run.size:
            raise ValueError(
                f"{path}: evt_time and evt_run have different lengths: "
                f"{evt_time.size:,} and {evt_run.size:,}"
            )

        finite = np.isfinite(evt_time)

        if not np.all(finite):
            n_bad = int(np.sum(~finite))

            raise ValueError(
                f"{path}: evt_time contains "
                f"{n_bad:,} non-finite values."
            )

        all_times.append(evt_time)
        all_runs.append(evt_run)

        print(f"  Events: {evt_time.size:,}")
        print(f"  Runs:   {np.unique(evt_run).size:,}")

    if not all_times:
        raise RuntimeError("No event arrays were loaded.")

    return (
        np.concatenate(all_times),
        np.concatenate(all_runs),
    )


# ============================================================
# Timed-window construction
# ============================================================

def split_at_backward_jumps(
    times_ns: np.ndarray,
) -> list[np.ndarray]:
    """Split one run whenever its timestamp moves backward.

    Events are expected to be stored in their intended event order. A backward
    jump is treated as a new independent time segment so no timed window spans
    a timestamp reset.
    """
    times_ns = np.asarray(
        times_ns,
        dtype=np.float64,
    ).reshape(-1)

    if times_ns.size == 0:
        return []

    split_positions = (
        np.flatnonzero(
            np.diff(times_ns) < 0
        )
        + 1
    )

    return [
        segment
        for segment in np.split(
            times_ns,
            split_positions,
        )
        if segment.size > 0
    ]


def count_timed_windows_by_run(
    evt_time_ns: np.ndarray,
    evt_run: np.ndarray,
    *,
    window_size_seconds: float,
    stride_seconds: float,
    skip_empty: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Count events in fixed-duration windows for each run independently.

    No window can cross a run boundary. Backward timestamp jumps inside the
    same run are also treated as segment boundaries.

    Returns
    -------
    window_runs:
        Run number for each timed window.

    event_counts:
        Number of events inside each timed window.
    """
    if window_size_seconds <= 0:
        raise ValueError(
            "window_size_seconds must be greater than zero."
        )

    if stride_seconds <= 0:
        raise ValueError(
            "stride_seconds must be greater than zero."
        )

    window_size_ns = (
        float(window_size_seconds)
        * 1.0e9
    )

    stride_ns = (
        float(stride_seconds)
        * 1.0e9
    )

    output_runs: list[np.ndarray] = []
    output_counts: list[np.ndarray] = []

    # Preserve the order in which runs first appear.
    _, first_positions = np.unique(
        evt_run,
        return_index=True,
    )

    ordered_runs = evt_run[
        np.sort(first_positions)
    ]

    for run in ordered_runs:
        run_times = evt_time_ns[
            evt_run == run
        ]

        if run_times.size == 0:
            continue

        segments = split_at_backward_jumps(
            run_times
        )

        for segment_times in segments:
            if segment_times.size == 0:
                continue

            # Sort within this continuous run segment.
            segment_times = np.sort(
                segment_times
            )

            first_time = float(
                segment_times[0]
            )

            last_time = float(
                segment_times[-1]
            )

            # Create windows whose start is no later than the final event.
            #
            # This includes the trailing partial-duration window. It does not
            # create starts after the last stored event.
            starts = np.arange(
                first_time,
                last_time + 0.5 * stride_ns,
                stride_ns,
                dtype=np.float64,
            )

            ends = (
                starts
                + window_size_ns
            )

            left = np.searchsorted(
                segment_times,
                starts,
                side="left",
            )

            right = np.searchsorted(
                segment_times,
                ends,
                side="left",
            )

            counts = (
                right - left
            ).astype(np.int64)

            if skip_empty:
                keep = counts > 0
                counts = counts[keep]

            if counts.size == 0:
                continue

            output_runs.append(
                np.full(
                    counts.size,
                    int(run),
                    dtype=np.int64,
                )
            )

            output_counts.append(
                counts
            )

    if not output_counts:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    return (
        np.concatenate(output_runs),
        np.concatenate(output_counts),
    )


# ============================================================
# Plotting
# ============================================================

def format_seconds(seconds: float) -> str:
    """Return a compact duration label."""
    seconds = float(seconds)

    if seconds < 60:
        return f"{seconds:g} s"

    minutes = seconds / 60.0

    if minutes < 60:
        return f"{minutes:g} min"

    hours = minutes / 60.0

    return f"{hours:g} h"


def safe_duration_for_filename(seconds: float) -> str:
    """Convert a duration into a filename-safe label."""
    seconds = float(seconds)

    if seconds.is_integer():
        return f"{int(seconds)}s"

    return (
        f"{seconds:g}s"
        .replace(".", "p")
    )


def print_summary(
    window_runs: np.ndarray,
    counts: np.ndarray,
    *,
    window_size_seconds: float,
) -> None:
    """Print overall and per-run count summaries."""
    print()
    print("=" * 80)
    print(
        f"TIME WINDOW = "
        f"{format_seconds(window_size_seconds)}"
    )
    print("=" * 80)

    if counts.size == 0:
        print("No timed windows were created.")
        return

    print(f"Number of windows: {counts.size:,}")
    print(f"Number of runs:    {np.unique(window_runs).size:,}")
    print(f"Minimum count:     {np.min(counts):,}")
    print(f"Mean count:        {np.mean(counts):.3f}")
    print(f"Median count:      {np.median(counts):.3f}")
    print(f"Maximum count:     {np.max(counts):,}")
    print(f"Empty windows:     {np.sum(counts == 0):,}")

    print()
    print(
        f"{'run':>10s} "
        f"{'windows':>10s} "
        f"{'mean':>12s} "
        f"{'median':>12s} "
        f"{'min':>10s} "
        f"{'max':>10s}"
    )
    print("-" * 70)

    for run in np.unique(window_runs):
        run_counts = counts[
            window_runs == run
        ]

        print(
            f"{int(run):>10d} "
            f"{run_counts.size:>10,d} "
            f"{np.mean(run_counts):>12.3f} "
            f"{np.median(run_counts):>12.3f} "
            f"{np.min(run_counts):>10,d} "
            f"{np.max(run_counts):>10,d}"
        )


def plot_events_per_window_by_run(
    window_runs: np.ndarray,
    counts: np.ndarray,
    *,
    window_size_seconds: float,
    stride_seconds: float,
    output_dir: Path,
) -> Path:
    """Create one box-and-whisker plot for one timed-window duration."""
    if counts.size == 0:
        raise ValueError(
            "Cannot create a box plot with no timed windows."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    unique_runs = np.unique(window_runs)

    counts_by_run = [
        counts[window_runs == run]
        for run in unique_runs
    ]

    # Increase the width when many runs are present.
    figure_width = max(
        FIGURE_WIDTH,
        0.45 * len(unique_runs),
    )

    figure, axis = plt.subplots(
        figsize=(
            figure_width,
            FIGURE_HEIGHT,
        )
    )

    axis.boxplot(
        counts_by_run,
        tick_labels=[
            str(int(run))
            for run in unique_runs
        ],
        showfliers=True,
        showmeans=True,
        meanline=False,
        whis=1.5,
    )

    axis.set_xlabel(
        "Run Number"
    )

    axis.set_ylabel(
        "Number of Events per Time Window"
    )

    axis.set_title(
        "Distribution of Event Counts per Timed Window by Run\n"
        f"Window duration: "
        f"{format_seconds(window_size_seconds)}, "
        f"stride: {format_seconds(stride_seconds)}"
    )

    axis.tick_params(
        axis="x",
        labelrotation=60,
    )

    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")

    axis.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    axis.set_axisbelow(True)

    figure.tight_layout()

    duration_label = safe_duration_for_filename(
        window_size_seconds
    )

    output_path = (
        output_dir
        / (
            "events_per_window_by_run_boxplot_"
            f"{duration_label}.png"
        )
    )

    figure.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


# ============================================================
# Main
# ============================================================

def main() -> None:
    npz_files = discover_npz_files(
        INPUT_PATH
    )

    print("=" * 80)
    print("INPUT")
    print("=" * 80)
    print(f"Input path:       {INPUT_PATH.expanduser().resolve()}")
    print(f"NPZ files found:  {len(npz_files):,}")
    print(f"Output directory: {OUTPUT_DIR.expanduser().resolve()}")
    print(
        f"Window stride:    "
        f"{TIME_WINDOW_STRIDE_SECONDS:g} seconds"
    )
    print(
        f"Skip empty:       "
        f"{SKIP_EMPTY_WINDOWS}"
    )

    evt_time_ns, evt_run = load_event_metadata(
        npz_files
    )

    print()
    print(f"Total events loaded: {evt_time_ns.size:,}")
    print(f"Unique runs:         {np.unique(evt_run).size:,}")

    output_dir = (
        OUTPUT_DIR
        .expanduser()
        .resolve()
    )

    generated_paths: list[Path] = []

    for window_size_seconds in TIME_WINDOW_SIZES_SECONDS:
        window_runs, event_counts = (
            count_timed_windows_by_run(
                evt_time_ns,
                evt_run,
                window_size_seconds=window_size_seconds,
                stride_seconds=TIME_WINDOW_STRIDE_SECONDS,
                skip_empty=SKIP_EMPTY_WINDOWS,
            )
        )

        print_summary(
            window_runs,
            event_counts,
            window_size_seconds=window_size_seconds,
        )

        if event_counts.size == 0:
            print("Skipping plot because no windows were created.")
            continue

        output_path = plot_events_per_window_by_run(
            window_runs,
            event_counts,
            window_size_seconds=window_size_seconds,
            stride_seconds=TIME_WINDOW_STRIDE_SECONDS,
            output_dir=output_dir,
        )

        generated_paths.append(
            output_path
        )

        print(f"Saved plot: {output_path}")

    print()
    print("=" * 80)
    print("FINISHED")
    print("=" * 80)
    print(f"Plots generated: {len(generated_paths):,}")

    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()