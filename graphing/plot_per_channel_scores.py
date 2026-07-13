#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# User settings
# ============================================================

INFERENCE_RESULT_DIR = Path(
    "/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection/"
    "checkpoints/graph_vae/All_data/inference_result"
)

GOOD_FILENAME = "scores_good.npz"
BAD_FILENAME = "scores_bad.npz"

# Channel range:
# CHANNEL_START <= channel < CHANNEL_END
CHANNEL_START = 0
CHANNEL_END = 11276

# ------------------------------------------------------------
# Spike removal
# ------------------------------------------------------------

# If True, automatically remove isolated extreme spikes.
SMOOTHING = True

# Number of neighboring channels used to estimate the local baseline.
# Must be odd.
SMOOTHING_WINDOW = 1001

# Larger value = less aggressive spike removal.
# A point is removed when its distance from the local median is more
# than this many local robust standard deviations.
SMOOTHING_THRESHOLD = 40.0

# Output plot filename
OUTPUT_FILENAME = "channel_mean_node_scores.png"

# Plot settings
FIGSIZE = (16, 7)
DPI = 200


# ============================================================
# Helper functions
# ============================================================

def load_mean_node_scores(npz_path: Path) -> tuple[np.ndarray, int]:
    """
    Load node_scores and calculate the mean score for each channel
    over all events/windows.

    Expected shape:
        (num_events, num_channels)

    Individual NaN values are ignored.
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"File does not exist: {npz_path}")

    print(f"Loading: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        if "node_scores" not in data:
            raise KeyError(
                f"'node_scores' not found in {npz_path}\n"
                f"Available keys: {list(data.keys())}"
            )

        node_scores = np.asarray(data["node_scores"], dtype=np.float64)

    if node_scores.ndim != 2:
        raise ValueError(
            f"Expected node_scores to be 2D, but got shape {node_scores.shape}"
        )

    num_events, num_channels = node_scores.shape

    print(f"  node_scores shape: {node_scores.shape}")
    print(f"  number of events:  {num_events:,}")
    print(f"  detected channels: {num_channels:,}")
    print(f"  total NaN values:  {np.isnan(node_scores).sum():,}")

    # Ignore individual NaN values.
    with np.errstate(invalid="ignore"):
        mean_node_scores = np.nanmean(node_scores, axis=0)

    print(
        f"  all-NaN channels:  "
        f"{np.isnan(mean_node_scores).sum():,}"
    )

    return mean_node_scores, num_channels


def resolve_channel_range(
    channel_start: int,
    channel_end: int,
    num_channels: int,
) -> tuple[int, int]:
    """
    Resolve requested channel bounds independently.
    """

    if channel_start < 0 or channel_start >= num_channels:
        print(
            f"Invalid CHANNEL_START={channel_start}. "
            "Using minimum channel 0."
        )
        start = 0
    else:
        start = channel_start

    if channel_end <= 0 or channel_end > num_channels:
        print(
            f"Invalid CHANNEL_END={channel_end}. "
            f"Using maximum bound {num_channels}."
        )
        end = num_channels
    else:
        end = channel_end

    if start >= end:
        raise ValueError(
            f"Invalid channel range: start={start}, end={end}"
        )

    print()
    print("Resolved channel range:")
    print(f"  start: {start}")
    print(f"  end:   {end} (exclusive)")
    print(f"  plotting {end - start:,} channels")

    return start, end


def remove_large_spikes(
    scores: np.ndarray,
    *,
    window_size: int,
    threshold: float,
    label: str,
) -> np.ndarray:
    """
    Remove isolated extreme spikes using a local median and local MAD.

    Removed points are replaced with NaN, so matplotlib skips only
    those individual points.

    This does not smooth or average the normal data.
    """

    scores = np.asarray(scores, dtype=np.float64).copy()

    if window_size < 3:
        raise ValueError("SMOOTHING_WINDOW must be at least 3.")

    if window_size % 2 == 0:
        window_size += 1
        print(
            f"SMOOTHING_WINDOW must be odd. "
            f"Using {window_size} instead."
        )

    half_window = window_size // 2

    spike_mask = np.zeros(scores.shape, dtype=bool)

    for i in range(scores.size):
        if not np.isfinite(scores[i]):
            continue

        start = max(0, i - half_window)
        end = min(scores.size, i + half_window + 1)

        local = scores[start:end]
        local = local[np.isfinite(local)]

        if local.size < 3:
            continue

        local_median = np.median(local)

        # Median absolute deviation
        mad = np.median(np.abs(local - local_median))

        # Convert MAD to a robust estimate of standard deviation.
        robust_sigma = 1.4826 * mad

        # Avoid division by zero in very flat regions.
        if robust_sigma <= 0:
            continue

        deviation = abs(scores[i] - local_median)

        if deviation > threshold * robust_sigma:
            spike_mask[i] = True

    num_removed = int(np.count_nonzero(spike_mask))

    print(
        f"  {label}: removed {num_removed:,} "
        f"extreme spike(s)"
    )

    # Replace only detected spikes with NaN.
    scores[spike_mask] = np.nan

    return scores


# ============================================================
# Main
# ============================================================

def main():
    good_path = INFERENCE_RESULT_DIR / GOOD_FILENAME
    bad_path = INFERENCE_RESULT_DIR / BAD_FILENAME
    output_path = INFERENCE_RESULT_DIR / OUTPUT_FILENAME

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    good_mean_scores, good_num_channels = load_mean_node_scores(
        good_path
    )

    print()

    bad_mean_scores, bad_num_channels = load_mean_node_scores(
        bad_path
    )

    if good_num_channels != bad_num_channels:
        raise ValueError(
            "Good and bad files have different numbers of channels:\n"
            f"  good: {good_num_channels}\n"
            f"  bad:  {bad_num_channels}"
        )

    num_channels = good_num_channels

    # --------------------------------------------------------
    # Resolve channel range
    # --------------------------------------------------------

    channel_start, channel_end = resolve_channel_range(
        CHANNEL_START,
        CHANNEL_END,
        num_channels,
    )

    channels = np.arange(channel_start, channel_end)

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
        print("Automatic spike removal enabled:")

        good_selected = remove_large_spikes(
            good_selected,
            window_size=SMOOTHING_WINDOW,
            threshold=SMOOTHING_THRESHOLD,
            label="Good",
        )

        bad_selected = remove_large_spikes(
            bad_selected,
            window_size=SMOOTHING_WINDOW,
            threshold=SMOOTHING_THRESHOLD,
            label="Bad",
        )
    else:
        print()
        print("Automatic spike removal disabled.")

    # --------------------------------------------------------
    # Valid masks
    # --------------------------------------------------------

    good_valid = np.isfinite(good_selected)
    bad_valid = np.isfinite(bad_selected)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(figsize=FIGSIZE)

    plt.plot(
        channels[good_valid],
        good_selected[good_valid],
        color="blue",
        linewidth=1.0,
        alpha=0.8,
        label="Good",
    )

    plt.plot(
        channels[bad_valid],
        bad_selected[bad_valid],
        color="red",
        linewidth=1.0,
        alpha=0.8,
        label="Bad",
    )

    plt.xlabel("Channel")
    plt.ylabel("Mean Node Score")
    plt.title(
        f"Mean Node Score per Channel "
        f"({channel_start}–{channel_end - 1})"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(channel_start, channel_end - 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI)
    plt.close()

    print()
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()