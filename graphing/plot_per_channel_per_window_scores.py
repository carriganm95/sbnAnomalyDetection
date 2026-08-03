#!/usr/bin/env python3

"""Plot per-channel GraphVAE node scores for selected inference windows.

Each row of ``node_scores`` is one window and each column is one channel.  If
more than one window is selected, this script takes the finite-value mean over
only those selected windows; it never averages over unselected windows.
"""

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg", force=True)
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

# This is used whenever no window selection is supplied by the caller.
# Comma-separated zero-based window indices and inclusive ranges are accepted.
# Examples: "1200", "1200,1205", "1200-1210".
DEFAULT_WINDOW_SPEC = "0-10"
DEFAULT_DATASETS = "both"  # "good", "bad", or "both"

# CHANNEL_START <= channel < CHANNEL_END.  None means all available channels.
CHANNEL_START = 0
CHANNEL_END: Optional[int] = None

# Disabled by default because a channel spike may be the anomaly of interest.
REMOVE_SPIKES = False
SMOOTHING_WINDOW = 1001  # must be odd
SMOOTHING_THRESHOLD = 5.0

OUTPUT_FILENAME = "channel_node_scores_selected_windows.png"
FIGSIZE = (16, 7)
DPI = 200


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-channel node scores for selected zero-based windows from "
            "scores_good.npz and/or scores_bad.npz. Multiple selected windows "
            "are averaged per channel."
        )
    )
    parser.add_argument(
        "inference_result_dir",
        nargs="?",
        default=str(DEFAULT_INFERENCE_RESULT_DIR),
        help="Directory containing scores_good.npz and scores_bad.npz.",
    )
    parser.add_argument(
        "--windows",
        default=DEFAULT_WINDOW_SPEC,
        help=(
            "Zero-based window indices/ranges, e.g. '1200', '1,4,8', or "
            "'1200-1210'. Ranges are inclusive. Use 'all' for every window."
        ),
    )
    parser.add_argument(
        "--datasets",
        choices=("good", "bad", "both"),
        default=DEFAULT_DATASETS,
        help="Which score file(s) to plot. Default: %(default)s.",
    )
    parser.add_argument(
        "--channel-start",
        type=int,
        default=CHANNEL_START,
        help="First channel to plot (inclusive). Default: %(default)s.",
    )
    parser.add_argument(
        "--channel-end",
        type=int,
        default=CHANNEL_END,
        help="Last channel bound (exclusive). Default: all channels.",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_FILENAME,
        help="Output filename or path. Default: %(default)s.",
    )
    parser.add_argument(
        "--remove-spikes",
        action="store_true",
        default=REMOVE_SPIKES,
        help="Remove isolated extreme channel spikes before plotting.",
    )
    return parser


def parse_window_spec(spec: str, num_windows: int) -> np.ndarray:
    """Return validated, unique, sorted zero-based window indices."""
    if num_windows <= 0:
        raise ValueError("The score file contains no windows.")

    spec = str(spec).strip().lower()
    if spec == "all":
        return np.arange(num_windows, dtype=np.int64)
    if not spec:
        raise ValueError("Window selection cannot be empty.")

    indices: set[int] = set()
    for raw_token in spec.split(","):
        token = raw_token.strip()
        if not token:
            raise ValueError(f"Invalid empty token in window selection {spec!r}.")

        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2 or not all(part.isdigit() for part in parts):
                raise ValueError(f"Invalid inclusive window range: {token!r}")
            start, end = (int(part) for part in parts)
            if start > end:
                raise ValueError(
                    f"Window range start must not exceed its end: {token!r}"
                )
            indices.update(range(start, end + 1))
        else:
            if not token.isdigit():
                raise ValueError(f"Invalid window index: {token!r}")
            indices.add(int(token))

    selected = np.asarray(sorted(indices), dtype=np.int64)
    invalid = selected[selected >= num_windows]
    if invalid.size:
        raise IndexError(
            "Selected window index/indices are outside the available range "
            f"0-{num_windows - 1}: {invalid.tolist()}"
        )
    return selected


def finite_mean(values: np.ndarray, axis: int) -> np.ndarray:
    """Mean over finite values without all-NaN RuntimeWarnings."""
    finite = np.isfinite(values)
    counts = finite.sum(axis=axis)
    sums = np.where(finite, values, 0.0).sum(axis=axis)
    result = np.full(counts.shape, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=result, where=counts > 0)
    return result


def load_selected_node_scores(
    npz_path: Path,
    window_spec: str,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Load node scores and average only the selected window rows."""
    npz_path = npz_path.expanduser().resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(f"Score file does not exist: {npz_path}")

    print(f"Loading: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as data:
        if "node_scores" not in data:
            raise KeyError(
                f"'node_scores' not found in {npz_path}. "
                f"Available keys: {list(data.keys())}"
            )
        node_scores = np.asarray(data["node_scores"], dtype=np.float64)

    if node_scores.ndim != 2:
        raise ValueError(
            f"Expected node_scores to be 2D, but got shape {node_scores.shape}."
        )

    num_windows, num_channels = node_scores.shape
    window_indices = parse_window_spec(window_spec, num_windows)
    selected_scores = finite_mean(node_scores[window_indices, :], axis=0)

    print(f"  node_scores shape: {node_scores.shape}")
    print(f"  selected windows:  {format_indices(window_indices)}")
    print(f"  selected count:    {window_indices.size:,}")
    print(
        "  all-NaN channels:  "
        f"{np.count_nonzero(~np.isfinite(selected_scores)):,}"
    )
    return selected_scores, num_channels, window_indices


def format_indices(indices: Sequence[int], max_items: int = 12) -> str:
    values = [int(value) for value in indices]
    if len(values) <= max_items:
        return ",".join(str(value) for value in values)
    head = ",".join(str(value) for value in values[: max_items // 2])
    tail = ",".join(str(value) for value in values[-max_items // 2 :])
    return f"{head},...,{tail}"


def resolve_channel_range(
    channel_start: int,
    channel_end: Optional[int],
    num_channels: int,
) -> Tuple[int, int]:
    end = num_channels if channel_end is None else channel_end
    if not 0 <= channel_start < num_channels:
        raise ValueError(
            f"channel_start must be within 0-{num_channels - 1}, got {channel_start}."
        )
    if not channel_start < end <= num_channels:
        raise ValueError(
            f"channel_end must be within {channel_start + 1}-{num_channels}, got {end}."
        )
    return channel_start, end


def remove_large_spikes(
    scores: np.ndarray,
    window_size: int,
    threshold: float,
    label: str,
) -> np.ndarray:
    """Replace isolated deviations from a local median/MAD baseline with NaN."""
    scores = np.asarray(scores, dtype=np.float64).copy()
    if window_size < 3:
        raise ValueError("SMOOTHING_WINDOW must be at least 3.")
    if threshold <= 0:
        raise ValueError("SMOOTHING_THRESHOLD must be positive.")
    if window_size % 2 == 0:
        window_size += 1

    half_window = window_size // 2
    spike_mask = np.zeros(scores.shape, dtype=bool)
    for i, current_value in enumerate(scores):
        if not np.isfinite(current_value):
            continue
        local = scores[max(0, i - half_window) : i + half_window + 1]
        local = local[np.isfinite(local)]
        if local.size < 3:
            continue
        local_median = np.median(local)
        robust_sigma = 1.4826 * np.median(np.abs(local - local_median))
        if robust_sigma > 0 and abs(current_value - local_median) > threshold * robust_sigma:
            spike_mask[i] = True

    scores[spike_mask] = np.nan
    print(f"  {label}: removed {np.count_nonzero(spike_mask):,} spike(s)")
    return scores


def resolve_output_path(inference_dir: Path, output: Union[Path, str]) -> Path:
    output_path = Path(output).expanduser()
    if not output_path.is_absolute():
        output_path = inference_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path.resolve()


def main(
    inference_result_dir: Optional[Union[Path, str]] = None,
    *,
    window_spec: Optional[str] = None,
    datasets: Optional[str] = None,
    channel_start: Optional[int] = None,
    channel_end: Optional[int] = None,
    output: Optional[Union[Path, str]] = None,
    remove_spikes: Optional[bool] = None,
) -> Path:
    """Create the selected-window per-channel plot and return its path."""
    if inference_result_dir is None:
        args = build_arg_parser().parse_args()
        inference_result_dir = args.inference_result_dir
        window_spec = args.windows
        datasets = args.datasets
        channel_start = args.channel_start
        channel_end = args.channel_end
        output = args.output
        remove_spikes = args.remove_spikes

    inference_dir = Path(inference_result_dir).expanduser().resolve()
    if not inference_dir.is_dir():
        raise NotADirectoryError(
            f"Inference result directory does not exist: {inference_dir}"
        )

    window_spec = DEFAULT_WINDOW_SPEC if window_spec is None else str(window_spec)
    datasets = DEFAULT_DATASETS if datasets is None else datasets
    channel_start = CHANNEL_START if channel_start is None else channel_start
    output = OUTPUT_FILENAME if output is None else output
    remove_spikes = REMOVE_SPIKES if remove_spikes is None else remove_spikes

    if datasets not in {"good", "bad", "both"}:
        raise ValueError("datasets must be 'good', 'bad', or 'both'.")

    requested = []
    if datasets in {"good", "both"}:
        requested.append(("Good", GOOD_FILENAME, "blue"))
    if datasets in {"bad", "both"}:
        requested.append(("Bad", BAD_FILENAME, "red"))

    loaded = []
    num_channels: Optional[int] = None
    for label, filename, color in requested:
        scores, current_num_channels, indices = load_selected_node_scores(
            inference_dir / filename,
            window_spec,
        )
        if num_channels is not None and current_num_channels != num_channels:
            raise ValueError(
                "Selected score files have different channel counts: "
                f"{num_channels} and {current_num_channels}."
            )
        num_channels = current_num_channels
        loaded.append((label, color, scores, indices))

    assert num_channels is not None
    start, end = resolve_channel_range(channel_start, channel_end, num_channels)
    channels = np.arange(start, end, dtype=np.int64)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for label, color, scores, indices in loaded:
        selected = scores[start:end].copy()
        if remove_spikes:
            selected = remove_large_spikes(
                selected,
                SMOOTHING_WINDOW,
                SMOOTHING_THRESHOLD,
                label,
            )
        valid = np.isfinite(selected)
        selection_label = format_indices(indices)
        prefix = "Window" if indices.size == 1 else "Windows"
        ax.plot(
            channels[valid],
            selected[valid],
            color=color,
            linewidth=1.0,
            alpha=0.85,
            label=f"{label} ({prefix} {selection_label})",
        )

    score_word = "Node Score" if loaded[0][3].size == 1 else "Mean Node Score"
    ax.set_xlabel("Channel", fontsize=14)
    ax.set_ylabel(score_word, fontsize=14)
    ax.set_title(
        f"{score_word} per Channel ({start}-{end - 1})",
        fontsize=16,
    )
    ax.set_xlim(start, end - 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = resolve_output_path(inference_dir, output)
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
    print(f"Saved plot to: {output_path}")
    return output_path


if __name__ == "__main__":
    main()