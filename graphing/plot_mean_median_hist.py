#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# User settings
# ============================================================

NPZ_PATH = Path(
    "/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection/scratch/8800_to_9000_200_channels/bad_events_test.npz"
)

WINDOW_SIZE = 100
STRIDE = 100

HIST_BINS = 80

# Histogram x-axis upper bound percentile.
#
# Examples:
#   99.0  -> hide the largest 1% tail
#   99.5  -> hide the largest 0.5% tail
#   99.9  -> hide the largest 0.1% tail
#   100.0 -> show the full range
X_AXIS_PERCENTILE = 95.0

OUTPUT_DIR = Path("window_integral_histograms")


# ============================================================
# Window statistics
# ============================================================


def calculate_window_channel_stats(
    channels_flat: np.ndarray,
    integrals_flat: np.ndarray,
    offsets: np.ndarray,
    n_channels: int,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return one mean and stdev value per (window, channel)."""

    n_events = len(offsets) - 1

    if window_size <= 0:
        raise ValueError("WINDOW_SIZE must be > 0")

    if stride <= 0:
        raise ValueError("STRIDE must be > 0")

    if n_channels <= 0:
        raise ValueError(
            f"n_channels must be > 0, got {n_channels}"
        )

    if n_events < window_size:
        raise ValueError(
            f"Not enough events: "
            f"n_events={n_events}, "
            f"window_size={window_size}"
        )

    starts = range(
        0,
        n_events - window_size + 1,
        stride,
    )

    n_windows = len(starts)

    total_entries = (
        n_windows * n_channels
    )

    means = np.empty(
        total_entries,
        dtype=np.float32,
    )

    stdevs = np.empty(
        total_entries,
        dtype=np.float32,
    )

    out_start = 0

    for window_idx, evt_start in enumerate(starts):
        evt_end = (
            evt_start + window_size
        )

        # The NPZ uses CSR-style event storage.
        #
        # offsets[event] gives the starting hit index
        # for that event.
        #
        # Therefore all hits belonging to the full
        # window are stored continuously between:
        #
        # offsets[evt_start]
        # and
        # offsets[evt_end]
        hit_start = int(
            offsets[evt_start]
        )

        hit_end = int(
            offsets[evt_end]
        )

        channels = channels_flat[
            hit_start:hit_end
        ]

        integrals = integrals_flat[
            hit_start:hit_end
        ]

        # Ignore invalid channel IDs.
        valid = (
            (channels >= 0)
            & (channels < n_channels)
        )

        channels = channels[valid]
        integrals = integrals[valid]

        # ----------------------------------------------------
        # Integral sum per channel
        # ----------------------------------------------------

        sums = np.bincount(
            channels,
            weights=integrals,
            minlength=n_channels,
        ).astype(
            np.float64
        )

        # ----------------------------------------------------
        # Number of hits per channel
        # ----------------------------------------------------

        counts = np.bincount(
            channels,
            minlength=n_channels,
        ).astype(
            np.float64
        )

        # ----------------------------------------------------
        # Sum of integral squared per channel
        # ----------------------------------------------------

        integrals_64 = integrals.astype(
            np.float64
        )

        sum_sq = np.bincount(
            channels,
            weights=integrals_64 ** 2,
            minlength=n_channels,
        ).astype(
            np.float64
        )

        # ----------------------------------------------------
        # Mean and standard deviation arrays
        # ----------------------------------------------------

        channel_means = np.zeros(
            n_channels,
            dtype=np.float64,
        )

        channel_stdevs = np.zeros(
            n_channels,
            dtype=np.float64,
        )

        active = (
            counts > 0
        )

        # Mean integral
        channel_means[active] = (
            sums[active]
            / counts[active]
        )

        # Population variance:
        #
        # Var(X) = E[X^2] - E[X]^2
        variances = np.zeros(
            n_channels,
            dtype=np.float64,
        )

        variances[active] = (
            sum_sq[active]
            / counts[active]
            - channel_means[active] ** 2
        )

        # Protect against tiny negative values caused
        # by floating-point precision.
        channel_stdevs[active] = np.sqrt(
            np.maximum(
                variances[active],
                0.0,
            )
        )

        # ----------------------------------------------------
        # Store all channels for this window
        # ----------------------------------------------------

        out_end = (
            out_start + n_channels
        )

        means[
            out_start:out_end
        ] = channel_means

        stdevs[
            out_start:out_end
        ] = channel_stdevs

        out_start = out_end

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (window_idx + 1) % 1000 == 0
            or window_idx + 1 == n_windows
        ):
            print(
                f"Processed "
                f"{window_idx + 1}"
                f"/{n_windows} windows"
            )

    return (
        means,
        stdevs,
        n_windows,
    )


# ============================================================
# Plotting
# ============================================================


def plot_histogram(
    values: np.ndarray,
    title: str,
    xlabel: str,
    output_path: Path,
) -> None:
    """Plot histogram with percentile-based x-axis clipping."""

    if values.size == 0:
        raise ValueError(
            f"No values available for {xlabel}"
        )

    if not (
        0.0 < X_AXIS_PERCENTILE <= 100.0
    ):
        raise ValueError(
            "X_AXIS_PERCENTILE must satisfy "
            "0 < percentile <= 100"
        )

    finite_values = values[
        np.isfinite(values)
    ]

    if finite_values.size == 0:
        raise ValueError(
            f"All values are non-finite for {xlabel}"
        )

    x_upper = float(
        np.percentile(
            finite_values,
            X_AXIS_PERCENTILE,
        )
    )

    print()
    print(
        f"{xlabel}:"
    )

    print(
        f"  min:                 "
        f"{np.min(finite_values):.6f}"
    )

    print(
        f"  mean:                "
        f"{np.mean(finite_values):.6f}"
    )

    print(
        f"  std:                 "
        f"{np.std(finite_values):.6f}"
    )

    print(
        f"  median:              "
        f"{np.median(finite_values):.6f}"
    )

    print(
        f"  max:                 "
        f"{np.max(finite_values):.6f}"
    )

    print(
        f"  {X_AXIS_PERCENTILE:g}th percentile: "
        f"{x_upper:.6f}"
    )

    n_above = int(
        np.count_nonzero(
            finite_values > x_upper
        )
    )

    print(
        f"  entries above bound: "
        f"{n_above}"
    )

    # Avoid an invalid histogram range if all values
    # up to the requested percentile are zero.
    if x_upper <= 0.0:
        x_upper = float(
            np.max(finite_values)
        )

    if x_upper <= 0.0:
        x_upper = 1.0

    plt.figure(
        figsize=(10, 7)
    )

    plt.hist(
        finite_values,
        bins=HIST_BINS,
        range=(
            0.0,
            x_upper,
        ),
    )

    plt.xlabel(
        xlabel
    )

    plt.ylabel(
        "Entries"
    )

    plt.title(
        title
        + "\n"
        + f"X-axis upper bound: "
        + f"{X_AXIS_PERCENTILE:g}th percentile "
        + f"({x_upper:.6f})"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


# ============================================================
# Main
# ============================================================


def main() -> None:
    input_path = (
        NPZ_PATH
        .expanduser()
        .resolve()
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"NPZ file does not exist: "
            f"{input_path}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Loading: {input_path}"
    )

    # ========================================================
    # Load NPZ
    # ========================================================

    with np.load(
        input_path,
        allow_pickle=False,
    ) as data:
        required = {
            "channels_flat",
            "integrals_flat",
            "offsets",
            "n_channels",
        }

        missing = (
            required - set(data.files)
        )

        if missing:
            raise KeyError(
                "Missing required NPZ arrays: "
                f"{sorted(missing)}"
            )

        channels_flat = np.asarray(
            data["channels_flat"],
            dtype=np.int64,
        )

        integrals_flat = np.asarray(
            data["integrals_flat"],
            dtype=np.float32,
        )

        offsets = np.asarray(
            data["offsets"],
            dtype=np.int64,
        )

        n_channels = int(
            np.asarray(
                data["n_channels"]
            ).item()
        )

    # ========================================================
    # Validate arrays
    # ========================================================

    if (
        len(channels_flat)
        != len(integrals_flat)
    ):
        raise ValueError(
            "channels_flat and integrals_flat "
            "have different lengths: "
            f"{len(channels_flat)} vs "
            f"{len(integrals_flat)}"
        )

    if (
        offsets.ndim != 1
        or len(offsets) < 2
    ):
        raise ValueError(
            "offsets must be a 1D array "
            "with at least 2 entries"
        )

    if int(offsets[0]) != 0:
        raise ValueError(
            f"offsets[0] must be 0, "
            f"got {int(offsets[0])}"
        )

    if (
        int(offsets[-1])
        != len(channels_flat)
    ):
        raise ValueError(
            f"offsets[-1]="
            f"{int(offsets[-1])}, "
            f"but there are "
            f"{len(channels_flat)} hits"
        )

    if np.any(
        offsets[1:] < offsets[:-1]
    ):
        raise ValueError(
            "offsets is not monotonically "
            "non-decreasing"
        )

    n_events = (
        len(offsets) - 1
    )

    # ========================================================
    # Dataset information
    # ========================================================

    print()
    print(
        f"Events:              "
        f"{n_events}"
    )

    print(
        f"Channels:            "
        f"{n_channels}"
    )

    print(
        f"Total hits:          "
        f"{len(channels_flat)}"
    )

    print(
        f"Window size:         "
        f"{WINDOW_SIZE}"
    )

    print(
        f"Stride:              "
        f"{STRIDE}"
    )

    print(
        f"X-axis percentile:   "
        f"{X_AXIS_PERCENTILE}"
    )

    # ========================================================
    # Calculate window statistics
    # ========================================================

    means, stdevs, n_windows = (
        calculate_window_channel_stats(
            channels_flat=channels_flat,
            integrals_flat=integrals_flat,
            offsets=offsets,
            n_channels=n_channels,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
        )
    )

    expected_entries = (
        n_windows * n_channels
    )

    # ========================================================
    # Print result information
    # ========================================================

    print()
    print(
        f"Number of windows:         "
        f"{n_windows}"
    )

    print(
        f"Channels per window:        "
        f"{n_channels}"
    )

    print(
        f"Expected histogram entries: "
        f"{expected_entries}"
    )

    print(
        f"Mean entries:               "
        f"{len(means)}"
    )

    print(
        f"Stdev entries:              "
        f"{len(stdevs)}"
    )

    if len(means) != expected_entries:
        raise RuntimeError(
            "Mean entry count does not match "
            "n_windows * n_channels"
        )

    if len(stdevs) != expected_entries:
        raise RuntimeError(
            "Stdev entry count does not match "
            "n_windows * n_channels"
        )

    # ========================================================
    # Mean histogram
    # ========================================================

    mean_output = (
        OUTPUT_DIR
        / "integral_mean_histogram.png"
    )

    plot_histogram(
        values=means,
        title=(
            "Integral Mean per Window and Channel\n"
            f"window_size={WINDOW_SIZE}, "
            f"stride={STRIDE}, "
            f"entries={len(means)}"
        ),
        xlabel="Mean integral",
        output_path=mean_output,
    )

    # ========================================================
    # Standard deviation histogram
    # ========================================================

    stdev_output = (
        OUTPUT_DIR
        / "integral_stdev_histogram.png"
    )

    plot_histogram(
        values=stdevs,
        title=(
            "Integral Standard Deviation "
            "per Window and Channel\n"
            f"window_size={WINDOW_SIZE}, "
            f"stride={STRIDE}, "
            f"entries={len(stdevs)}"
        ),
        xlabel="Integral standard deviation",
        output_path=stdev_output,
    )

    # ========================================================
    # Done
    # ========================================================

    print()
    print("Saved:")
    print(mean_output)
    print(stdev_output)


if __name__ == "__main__":
    main()