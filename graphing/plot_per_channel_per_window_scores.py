#!/usr/bin/env python3

"""Plot per-channel GraphVAE score distributions for selected windows.

Each row of ``node_scores`` is one window and each column is one channel. For
every channel, the box spans Q1-Q3, the center line is the median, and the
whiskers extend to the most extreme values within 1.5 IQR. Six PNG files are
written using plane membership read from the detector channel-map CSV.
"""

import argparse
import csv
from pathlib import Path
import re
from typing import Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Patch
import numpy as np


# ============================================================
# User settings
# ============================================================

DEFAULT_INFERENCE_RESULT_DIR = Path(
    "/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection/"
    "checkpoints/graph_vae/All_data/inference_result"
)
# Internal channel-map path used unless an explicit override is supplied.
CHANNEL_MAP_PATH = Path(
    "/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection/"
    "configs/SBNDTPCChannelMap_v2_with_positions.csv"
)

GOOD_FILENAME = "scores_good.npz"
BAD_FILENAME = "scores_bad.npz"

# Used when no window selection is supplied by the caller.
# Examples: "1200", "1200,1205", "1200-1210", or "all".
DEFAULT_WINDOW_SPEC = "0-10"
DEFAULT_DATASETS = "both"  # "good", "bad", or "both"

# Leave these as None for automatic CSV-header detection. Set an exact header
# name only if your channel-map file uses an unrecognized name.
CHANNEL_MAP_CHANNEL_COLUMN: Optional[str] = None
CHANNEL_MAP_PLANE_COLUMN: Optional[str] = None
CHANNEL_MAP_TPC_COLUMN: Optional[str] = None

EXPECTED_NUM_PLANES = 6

# This is a base filename. The plotter adds _plane_1 through _plane_6.
OUTPUT_FILENAME = "channel_node_scores_selected_windows.png"

WHISKER_IQR = 1.5
SHOW_FLIERS = True
FIGSIZE = (18, 7)
DPI = 180

GOOD_COLOR = "#2563eb"
BAD_COLOR = "#dc2626"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create six per-plane box-and-whisker plots of per-channel node "
            "scores across selected zero-based windows."
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
        "--channel-map",
        default=str(CHANNEL_MAP_PATH),
        help=(
            "CSV used to assign channels to detector planes. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_FILENAME,
        help=(
            "Base output filename/path. '_plane_N' is added before the suffix. "
            "Default: %(default)s."
        ),
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


def format_indices(indices: Sequence[int], max_items: int = 12) -> str:
    values = [int(value) for value in indices]
    if len(values) <= max_items:
        return ",".join(str(value) for value in values)
    head = ",".join(str(value) for value in values[: max_items // 2])
    tail = ",".join(str(value) for value in values[-max_items // 2 :])
    return f"{head},...,{tail}"


def normalize_header(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def detect_csv_column(
    fieldnames: Sequence[str],
    explicit_name: Optional[str],
    aliases: Sequence[str],
    role: str,
    required: bool = True,
) -> Optional[str]:
    """Resolve one CSV column by explicit name or normalized aliases."""
    fieldnames = [str(name) for name in fieldnames if name is not None]
    if explicit_name is not None:
        if explicit_name not in fieldnames:
            raise ValueError(
                f"Configured {role} column {explicit_name!r} is not in the CSV. "
                f"Available columns: {fieldnames}"
            )
        return explicit_name

    normalized_to_original = {
        normalize_header(name): name for name in fieldnames
    }
    for alias in aliases:
        match = normalized_to_original.get(normalize_header(alias))
        if match is not None:
            return match

    if required:
        raise ValueError(
            f"Could not identify the {role} column. Available columns: {fieldnames}. "
            f"Set CHANNEL_MAP_{role.upper()}_COLUMN to the exact header name."
        )
    return None


def natural_sort_key(value: str) -> tuple:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", str(value))
    )


def load_plane_channel_groups(
    channel_map_path: Union[Path, str],
    num_channels: int,
) -> list[tuple[str, np.ndarray]]:
    """Read the channel map and derive exactly six channel groups."""
    channel_map_path = Path(channel_map_path).expanduser().resolve()
    if not channel_map_path.is_file():
        raise FileNotFoundError(f"Channel-map CSV does not exist: {channel_map_path}")

    with channel_map_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Channel-map CSV has no header: {channel_map_path}")

        channel_column = detect_csv_column(
            reader.fieldnames,
            CHANNEL_MAP_CHANNEL_COLUMN,
            (
                "channel",
                "channel_id",
                "channelid",
                "offline_channel",
                "offlinechannel",
                "offline_channel_id",
                "offlchan",
                "offlinechan",
                "chan",
                "chan_id",
            ),
            "channel",
        )
        plane_column = detect_csv_column(
            reader.fieldnames,
            CHANNEL_MAP_PLANE_COLUMN,
            (
                "plane",
                "plane_id",
                "planeid",
                "wire_plane",
                "wireplane",
                "view",
                "view_id",
            ),
            "plane",
        )
        tpc_column = detect_csv_column(
            reader.fieldnames,
            CHANNEL_MAP_TPC_COLUMN,
            (
                "tpc",
                "tpc_id",
                "tpcid",
                "tpc_number",
                "tpcnumber",
                "drift_volume",
                "driftvolume",
                "apa",
                "apa_id",
                "eastwest",
                "east_west",
            ),
            "tpc",
            required=False,
        )

        rows: list[tuple[int, str, Optional[str]]] = []
        for row_number, row in enumerate(reader, start=2):
            raw_channel = str(row.get(channel_column, "")).strip()
            raw_plane = str(row.get(plane_column, "")).strip()
            raw_tpc = (
                str(row.get(tpc_column, "")).strip()
                if tpc_column is not None
                else None
            )
            if not raw_channel or not raw_plane:
                continue
            try:
                channel = int(float(raw_channel))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid channel value {raw_channel!r} on CSV row {row_number}."
                ) from exc
            if channel < 0 or channel >= num_channels:
                raise ValueError(
                    f"Channel-map channel {channel} on row {row_number} is outside "
                    f"node_scores columns 0-{num_channels - 1}. This commonly means "
                    "the score file uses reindexed subset channels."
                )
            rows.append((channel, raw_plane, raw_tpc or None))

    if not rows:
        raise ValueError(f"No usable channel/plane rows found in {channel_map_path}.")

    print(f"Channel map:       {channel_map_path}")
    print(f"  channel column:  {channel_column}")
    print(f"  plane column:    {plane_column}")
    print(f"  TPC column:      {tpc_column or 'not found'}")

    # A physical channel must not map to conflicting plane/TPC labels.
    channel_labels: dict[int, set[tuple[str, Optional[str]]]] = {}
    for channel, plane, tpc in rows:
        channel_labels.setdefault(channel, set()).add((plane, tpc))
    conflicts = {
        channel: labels
        for channel, labels in channel_labels.items()
        if len(labels) > 1
    }
    if conflicts:
        examples = list(conflicts.items())[:5]
        raise ValueError(
            "Some channels map to conflicting plane/TPC labels. Examples: "
            f"{examples}"
        )

    unique_planes = sorted({plane for _, plane, _ in rows}, key=natural_sort_key)
    groups: dict[tuple, set[int]] = {}
    labels: dict[tuple, str] = {}
    strategy = ""

    if len(unique_planes) == EXPECTED_NUM_PLANES:
        strategy = "six unique values from the plane column"
        for channel, plane, _tpc in rows:
            key = ("plane", plane)
            groups.setdefault(key, set()).add(channel)
            labels[key] = f"Plane {plane}"
    elif tpc_column is not None:
        for channel, plane, tpc in rows:
            key = ("tpc_plane", tpc, plane)
            groups.setdefault(key, set()).add(channel)
            labels[key] = f"TPC {tpc}, Plane {plane}"
        if len(groups) == EXPECTED_NUM_PLANES:
            strategy = "six unique (TPC, plane) pairs"
        else:
            groups.clear()
            labels.clear()

    if not groups:
        # Final fallback: split the channel-ordered map whenever its plane label
        # changes. This supports maps with repeated U/V/Y labels but no TPC field.
        ordered = sorted(
            (channel, next(iter(values))[0])
            for channel, values in channel_labels.items()
        )
        run_number = 0
        previous_plane: Optional[str] = None
        for channel, plane in ordered:
            if plane != previous_plane:
                run_number += 1
                previous_plane = plane
            key = ("run", run_number, plane)
            groups.setdefault(key, set()).add(channel)
            labels[key] = f"Plane {plane} (group {run_number})"
        strategy = "contiguous plane-label runs in channel order"

    if len(groups) != EXPECTED_NUM_PLANES:
        summary = {
            labels[key]: len(channels) for key, channels in groups.items()
        }
        raise ValueError(
            f"Expected {EXPECTED_NUM_PLANES} detector-plane groups but derived "
            f"{len(groups)} using {strategy}. Derived groups: {summary}"
        )

    plane_groups = [
        (labels[key], np.asarray(sorted(channels), dtype=np.int64))
        for key, channels in groups.items()
    ]
    plane_groups.sort(key=lambda item: int(item[1][0]))

    assigned_channels: set[int] = set()
    for label, channels in plane_groups:
        overlap = assigned_channels.intersection(channels.tolist())
        if overlap:
            raise ValueError(
                f"Channel-map groups overlap at channels such as {sorted(overlap)[:5]}."
            )
        assigned_channels.update(channels.tolist())
        print(
            f"  {label}: {channels.size:,} channels "
            f"({channels.min()}-{channels.max()})"
        )
    print(f"  grouping method: {strategy}")
    return plane_groups


def load_selected_node_scores(
    npz_path: Path,
    window_spec: str,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Load and return raw node-score rows for only the selected windows."""
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
    selected_scores = node_scores[window_indices, :].copy()

    print(f"  node_scores shape: {node_scores.shape}")
    print(f"  selected windows:  {format_indices(window_indices)}")
    print(f"  selected count:    {window_indices.size:,}")
    return selected_scores, num_channels, window_indices


def resolve_output_base(
    inference_dir: Path,
    output: Union[Path, str],
) -> Path:
    output_path = Path(output).expanduser()
    if not output_path.is_absolute():
        output_path = inference_dir / output_path
    if not output_path.suffix:
        output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path.resolve()


def get_output_paths(
    inference_result_dir: Union[Path, str],
    output: Union[Path, str] = OUTPUT_FILENAME,
) -> list[Path]:
    """Return the six filenames produced for the configured plane ranges."""
    inference_dir = Path(inference_result_dir).expanduser().resolve()
    base = resolve_output_base(inference_dir, output)
    return [
        base.with_name(f"{base.stem}_plane_{plane_number}{base.suffix}")
        for plane_number in range(1, EXPECTED_NUM_PLANES + 1)
    ]


def calculate_box_statistics(
    selected_scores: np.ndarray,
    channel_ids: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Calculate Q1, median, Q3, whiskers, and outliers per channel."""
    plane_scores = selected_scores[:, channel_ids]
    num_plane_channels = channel_ids.size
    q1 = np.full(num_plane_channels, np.nan)
    median = np.full(num_plane_channels, np.nan)
    q3 = np.full(num_plane_channels, np.nan)
    whisker_low = np.full(num_plane_channels, np.nan)
    whisker_high = np.full(num_plane_channels, np.nan)
    flier_channel_indices = []
    flier_values = []

    for channel_index in range(num_plane_channels):
        values = plane_scores[:, channel_index]
        values = values[np.isfinite(values)]
        if not values.size:
            continue

        q1[channel_index], median[channel_index], q3[channel_index] = (
            np.percentile(values, (25.0, 50.0, 75.0))
        )
        iqr = q3[channel_index] - q1[channel_index]
        low_fence = q1[channel_index] - WHISKER_IQR * iqr
        high_fence = q3[channel_index] + WHISKER_IQR * iqr
        inliers = values[(values >= low_fence) & (values <= high_fence)]
        whisker_low[channel_index] = np.min(inliers)
        whisker_high[channel_index] = np.max(inliers)

        if SHOW_FLIERS:
            outliers = values[(values < low_fence) | (values > high_fence)]
            flier_channel_indices.extend([channel_index] * outliers.size)
            flier_values.extend(outliers.tolist())

    return (
        q1,
        median,
        q3,
        whisker_low,
        whisker_high,
        np.asarray(flier_channel_indices, dtype=np.int64),
        np.asarray(flier_values, dtype=np.float64),
    )


def draw_boxplots(
    ax: plt.Axes,
    selected_scores: np.ndarray,
    channel_ids: np.ndarray,
    positions: np.ndarray,
    color: str,
    width: float,
) -> None:
    """Draw exact per-channel boxes efficiently using artist collections."""
    (
        q1,
        median,
        q3,
        whisker_low,
        whisker_high,
        flier_channel_indices,
        flier_values,
    ) = calculate_box_statistics(selected_scores, channel_ids)

    valid = np.isfinite(median)
    x = positions[valid]
    q1_valid = q1[valid]
    median_valid = median[valid]
    q3_valid = q3[valid]
    low_valid = whisker_low[valid]
    high_valid = whisker_high[valid]
    half_width = width / 2.0
    cap_half_width = width * 0.28

    boxes = [
        [
            (channel - half_width, lower),
            (channel + half_width, lower),
            (channel + half_width, upper),
            (channel - half_width, upper),
        ]
        for channel, lower, upper in zip(x, q1_valid, q3_valid)
    ]
    ax.add_collection(
        PolyCollection(
            boxes,
            facecolors=color,
            edgecolors=color,
            linewidths=0.25,
            alpha=0.30,
            zorder=2,
        )
    )

    whiskers = [
        [(channel, low), (channel, high)]
        for channel, low, high in zip(x, low_valid, high_valid)
    ]
    caps = [
        [(channel - cap_half_width, value), (channel + cap_half_width, value)]
        for channel, low, high in zip(x, low_valid, high_valid)
        for value in (low, high)
    ]
    medians = [
        [(channel - half_width, value), (channel + half_width, value)]
        for channel, value in zip(x, median_valid)
    ]
    ax.add_collection(
        LineCollection(whiskers, colors=color, linewidths=0.35, alpha=0.75, zorder=1)
    )
    ax.add_collection(
        LineCollection(caps, colors=color, linewidths=0.45, alpha=0.75, zorder=3)
    )
    ax.add_collection(
        LineCollection(medians, colors=color, linewidths=0.7, zorder=4)
    )

    if SHOW_FLIERS and flier_values.size:
        ax.scatter(
            positions[flier_channel_indices],
            flier_values,
            s=1.0,
            color=color,
            alpha=0.35,
            linewidths=0,
            zorder=5,
        )


def main(
    inference_result_dir: Optional[Union[Path, str]] = None,
    *,
    window_spec: Optional[str] = None,
    datasets: Optional[str] = None,
    channel_map: Optional[Union[Path, str]] = None,
    output: Optional[Union[Path, str]] = None,
) -> list[Path]:
    """Create and return six selected-window per-channel boxplot paths."""
    if inference_result_dir is None:
        args = build_arg_parser().parse_args()
        inference_result_dir = args.inference_result_dir
        window_spec = args.windows
        datasets = args.datasets
        channel_map = args.channel_map
        output = args.output

    inference_dir = Path(inference_result_dir).expanduser().resolve()
    if not inference_dir.is_dir():
        raise NotADirectoryError(
            f"Inference result directory does not exist: {inference_dir}"
        )

    window_spec = DEFAULT_WINDOW_SPEC if window_spec is None else str(window_spec)
    datasets = DEFAULT_DATASETS if datasets is None else datasets
    channel_map = CHANNEL_MAP_PATH if channel_map is None else channel_map
    output = OUTPUT_FILENAME if output is None else output
    if datasets not in {"good", "bad", "both"}:
        raise ValueError("datasets must be 'good', 'bad', or 'both'.")

    requested = []
    if datasets in {"good", "both"}:
        requested.append(("Good", GOOD_FILENAME, GOOD_COLOR))
    if datasets in {"bad", "both"}:
        requested.append(("Bad", BAD_FILENAME, BAD_COLOR))

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
    plane_groups = load_plane_channel_groups(channel_map, num_channels)
    output_paths = get_output_paths(inference_dir, output)

    for plane_number, ((plane_label, channel_ids), output_path) in enumerate(
        zip(plane_groups, output_paths),
        start=1,
    ):
        channels = channel_ids.astype(np.float64)
        fig, ax = plt.subplots(figsize=FIGSIZE)

        if len(loaded) == 2:
            offsets = (-0.20, 0.20)
            width = 0.34
        else:
            offsets = (0.0,)
            width = 0.68

        legend_handles = []
        for (label, color, scores, _indices), offset in zip(loaded, offsets):
            draw_boxplots(
                ax,
                scores,
                channel_ids,
                channels + offset,
                color,
                width,
            )
            legend_handles.append(
                Patch(facecolor=color, edgecolor=color, alpha=0.35, label=label)
            )

        selected_indices = loaded[0][3]
        selection_text = (
            f"window {selected_indices[0]}"
            if selected_indices.size == 1
            else f"{selected_indices.size:,} selected windows"
        )
        ax.set_xlabel("Channel", fontsize=14)
        ax.set_ylabel("Node Anomaly Score", fontsize=14)
        if np.all(np.diff(channel_ids) == 1):
            channel_summary = f"channels {channel_ids[0]}-{channel_ids[-1]}"
        else:
            channel_summary = (
                f"{channel_ids.size:,} mapped channels, "
                f"IDs {channel_ids.min()}-{channel_ids.max()}"
            )
        ax.set_title(
            f"{plane_label}: Per-Channel Score Distribution Across {selection_text} "
            f"({channel_summary})",
            fontsize=15,
        )
        ax.set_xlim(channel_ids.min() - 1, channel_ids.max() + 1)
        ax.autoscale_view(scalex=False, scaley=True)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(handles=legend_handles, loc="upper right")
        fig.tight_layout()
        fig.savefig(output_path, dpi=DPI)
        plt.close(fig)
        print(f"Saved {plane_label} plot to: {output_path}")

    return output_paths


if __name__ == "__main__":
    main()