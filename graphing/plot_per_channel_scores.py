#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Default settings
# ============================================================

DEFAULT_INFERENCE_RESULT_DIR = Path(
    "/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection/"
    "checkpoints/graph_vae/All_data/inference_result"
)

GOOD_FILENAME = "scores_good.npz"
BAD_FILENAME = "scores_bad.npz"

# Channel range:
# CHANNEL_START <= channel < CHANNEL_END
CHANNEL_START = 0
CHANNEL_END = 11276


# ============================================================
# Spike removal
# ============================================================

# If True, automatically remove isolated extreme spikes.
SMOOTHING = True

# Number of neighboring channels used to estimate the local baseline.
# Must be odd.
SMOOTHING_WINDOW = 1001

# Larger value = less aggressive spike removal.
SMOOTHING_THRESHOLD = 5


# ============================================================
# Output settings
# ============================================================

OUTPUT_FILENAME = "channel_mean_node_scores.png"

FIGSIZE = (16, 7)
DPI = 200


# ============================================================
# Command-line arguments
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot mean node scores per channel from one model's "
            "scores_good.npz and scores_bad.npz files."
        )
    )

    parser.add_argument(
        "inference_result_dir",
        nargs="?",
        default=str(DEFAULT_INFERENCE_RESULT_DIR),
        help=(
            "Directory containing scores_good.npz and scores_bad.npz. "
            "If omitted, DEFAULT_INFERENCE_RESULT_DIR is used."
        ),
    )

    return parser


# ============================================================
# Helper functions
# ============================================================

def load_mean_node_scores(
    npz_path: Path,
) -> Tuple[np.ndarray, int]:
    """
    Load node_scores and calculate the mean score for each channel
    over all events/windows.

    Expected node_scores shape:
        (num_events, num_channels)

    Individual NaN values are ignored.
    """

    npz_path = npz_path.expanduser().resolve()

    if not npz_path.exists():
        raise FileNotFoundError(
            "File does not exist: {}".format(npz_path)
        )

    if not npz_path.is_file():
        raise FileNotFoundError(
            "Path is not a file: {}".format(npz_path)
        )

    print("Loading: {}".format(npz_path))

    with np.load(npz_path, allow_pickle=True) as data:
        if "node_scores" not in data:
            raise KeyError(
                "'node_scores' not found in {}\n"
                "Available keys: {}".format(
                    npz_path,
                    list(data.keys()),
                )
            )

        node_scores = np.asarray(
            data["node_scores"],
            dtype=np.float64,
        )

    if node_scores.ndim != 2:
        raise ValueError(
            "Expected node_scores to be 2D, but got shape {}".format(
                node_scores.shape
            )
        )

    num_events = node_scores.shape[0]
    num_channels = node_scores.shape[1]

    print("  node_scores shape: {}".format(node_scores.shape))
    print("  number of events:  {:,}".format(num_events))
    print("  detected channels: {:,}".format(num_channels))
    print(
        "  total NaN values:  {:,}".format(
            int(np.isnan(node_scores).sum())
        )
    )

    # Avoid RuntimeWarning from np.nanmean on channels that are all NaN.
    finite_counts = np.sum(
        np.isfinite(node_scores),
        axis=0,
    )

    score_sums = np.nansum(
        node_scores,
        axis=0,
    )

    mean_node_scores = np.full(
        num_channels,
        np.nan,
        dtype=np.float64,
    )

    valid_channels = finite_counts > 0

    mean_node_scores[valid_channels] = (
        score_sums[valid_channels]
        / finite_counts[valid_channels]
    )

    print(
        "  all-NaN channels:  {:,}".format(
            int(np.count_nonzero(~valid_channels))
        )
    )

    return mean_node_scores, num_channels


def resolve_channel_range(
    channel_start: int,
    channel_end: int,
    num_channels: int,
) -> Tuple[int, int]:
    """
    Resolve requested channel bounds.
    """

    if num_channels <= 0:
        raise ValueError(
            "num_channels must be positive."
        )

    if channel_start < 0 or channel_start >= num_channels:
        print(
            "Invalid CHANNEL_START={}. "
            "Using minimum channel 0.".format(
                channel_start
            )
        )
        start = 0
    else:
        start = channel_start

    if channel_end <= 0 or channel_end > num_channels:
        print(
            "Invalid CHANNEL_END={}. "
            "Using maximum bound {}.".format(
                channel_end,
                num_channels,
            )
        )
        end = num_channels
    else:
        end = channel_end

    if start >= end:
        raise ValueError(
            "Invalid channel range: start={}, end={}".format(
                start,
                end,
            )
        )

    print()
    print("Resolved channel range:")
    print("  start: {}".format(start))
    print("  end:   {} (exclusive)".format(end))
    print(
        "  plotting {:,} channels".format(
            end - start
        )
    )

    return start, end


def remove_large_spikes(
    scores: np.ndarray,
    window_size: int,
    threshold: float,
    label: str,
) -> np.ndarray:
    """
    Remove isolated extreme spikes using a local median and local MAD.

    Removed points are replaced with NaN.

    This does not smooth or average normal data.
    """

    scores = np.asarray(
        scores,
        dtype=np.float64,
    ).copy()

    if scores.ndim != 1:
        raise ValueError(
            "Expected a 1D score array, but got shape {}".format(
                scores.shape
            )
        )

    if window_size < 3:
        raise ValueError(
            "SMOOTHING_WINDOW must be at least 3."
        )

    if threshold <= 0:
        raise ValueError(
            "SMOOTHING_THRESHOLD must be positive."
        )

    if window_size % 2 == 0:
        window_size += 1

        print(
            "SMOOTHING_WINDOW must be odd. "
            "Using {} instead.".format(
                window_size
            )
        )

    half_window = window_size // 2

    spike_mask = np.zeros(
        scores.shape,
        dtype=bool,
    )

    for i in range(scores.size):
        current_value = scores[i]

        if not np.isfinite(current_value):
            continue

        local_start = max(
            0,
            i - half_window,
        )

        local_end = min(
            scores.size,
            i + half_window + 1,
        )

        local = scores[
            local_start:local_end
        ]

        local = local[
            np.isfinite(local)
        ]

        if local.size < 3:
            continue

        local_median = np.median(local)

        mad = np.median(
            np.abs(
                local - local_median
            )
        )

        robust_sigma = 1.4826 * mad

        if (
            not np.isfinite(robust_sigma)
            or robust_sigma <= 0
        ):
            continue

        deviation = abs(
            current_value - local_median
        )

        if deviation > threshold * robust_sigma:
            spike_mask[i] = True

    num_removed = int(
        np.count_nonzero(spike_mask)
    )

    print(
        "  {}: removed {:,} extreme spike(s)".format(
            label,
            num_removed,
        )
    )

    scores[spike_mask] = np.nan

    return scores


# ============================================================
# Main plotting function
# ============================================================

def main(
    inference_result_dir: Optional[
        Union[Path, str]
    ] = None,
) -> None:
    """
    Plot one model's per-channel mean node scores.

    This function can be used in two ways.

    Command line:
        python plot_per_channel_scores.py /path/to/inference_result

    From another Python script:
        main("/path/to/inference_result")
    """

    # --------------------------------------------------------
    # Resolve inference directory
    # --------------------------------------------------------

    if inference_result_dir is None:
        parser = build_arg_parser()
        args = parser.parse_args()

        inference_result_dir = args.inference_result_dir

    inference_result_dir = Path(
        inference_result_dir
    ).expanduser().resolve()

    if not inference_result_dir.exists():
        raise FileNotFoundError(
            "Inference result directory does not exist: {}".format(
                inference_result_dir
            )
        )

    if not inference_result_dir.is_dir():
        raise NotADirectoryError(
            "Inference result path is not a directory: {}".format(
                inference_result_dir
            )
        )

    good_path = (
        inference_result_dir
        / GOOD_FILENAME
    )

    bad_path = (
        inference_result_dir
        / BAD_FILENAME
    )

    output_path = (
        inference_result_dir
        / OUTPUT_FILENAME
    )

    print()
    print("=" * 80)
    print("Per-channel node-score plotting")
    print(
        "Inference result directory: {}".format(
            inference_result_dir
        )
    )
    print(
        "Good score file: {}".format(
            good_path
        )
    )
    print(
        "Bad score file:  {}".format(
            bad_path
        )
    )
    print(
        "Output plot:     {}".format(
            output_path
        )
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Load good data
    # --------------------------------------------------------

    good_mean_scores, good_num_channels = (
        load_mean_node_scores(
            good_path
        )
    )

    print()

    # --------------------------------------------------------
    # Load bad data
    # --------------------------------------------------------

    bad_mean_scores, bad_num_channels = (
        load_mean_node_scores(
            bad_path
        )
    )

    # --------------------------------------------------------
    # Validate channel counts
    # --------------------------------------------------------

    if good_num_channels != bad_num_channels:
        raise ValueError(
            "Good and bad files have different numbers of channels:\n"
            "  good: {}\n"
            "  bad:  {}".format(
                good_num_channels,
                bad_num_channels,
            )
        )

    num_channels = good_num_channels

    # --------------------------------------------------------
    # Resolve channel range
    # --------------------------------------------------------

    channel_start, channel_end = (
        resolve_channel_range(
            CHANNEL_START,
            CHANNEL_END,
            num_channels,
        )
    )

    channels = np.arange(
        channel_start,
        channel_end,
        dtype=np.int64,
    )

    good_selected = good_mean_scores[
        channel_start:channel_end
    ].copy()

    bad_selected = bad_mean_scores[
        channel_start:channel_end
    ].copy()

    # --------------------------------------------------------
    # Remove isolated extreme spikes
    # --------------------------------------------------------

    if SMOOTHING:
        print()
        print(
            "Automatic spike removal enabled:"
        )

        good_selected = remove_large_spikes(
            scores=good_selected,
            window_size=SMOOTHING_WINDOW,
            threshold=SMOOTHING_THRESHOLD,
            label="Good",
        )

        bad_selected = remove_large_spikes(
            scores=bad_selected,
            window_size=SMOOTHING_WINDOW,
            threshold=SMOOTHING_THRESHOLD,
            label="Bad",
        )

    else:
        print()
        print(
            "Automatic spike removal disabled."
        )

    # --------------------------------------------------------
    # Valid masks
    # --------------------------------------------------------

    good_valid = np.isfinite(
        good_selected
    )

    bad_valid = np.isfinite(
        bad_selected
    )

    print()
    print("Finite plotted channels:")

    print(
        "  Good: {:,} / {:,}".format(
            int(
                np.count_nonzero(
                    good_valid
                )
            ),
            good_selected.size,
        )
    )

    print(
        "  Bad:  {:,} / {:,}".format(
            int(
                np.count_nonzero(
                    bad_valid
                )
            ),
            bad_selected.size,
        )
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    ax.plot(
        channels[good_valid],
        good_selected[good_valid],
        color="blue",
        linewidth=1.0,
        alpha=0.8,
        label="Good",
    )

    ax.plot(
        channels[bad_valid],
        bad_selected[bad_valid],
        color="red",
        linewidth=1.0,
        alpha=0.8,
        label="Bad",
    )

    ax.set_xlabel(
        "Channel",
        fontsize=14,
    )

    ax.set_ylabel(
        "Mean Node Score",
        fontsize=14,
    )

    ax.set_title(
        "Mean Node Score per Channel "
        "({}-{})".format(
            channel_start,
            channel_end - 1,
        ),
        fontsize=16,
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.set_xlim(
        channel_start,
        channel_end - 1,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=DPI,
    )

    plt.close(fig)

    print()
    print(
        "Saved plot to: {}".format(
            output_path
        )
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()