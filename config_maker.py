#!/usr/bin/env python3
"""
Generate GraphVAE sweep YAML configuration files.

The generator supports both GraphVAE window definitions used by the newer
configuration format:

    event mode:
        data.window_mode = "event"
        sweep data.window_size and data.stride

    time mode:
        data.window_mode = "time"
        sweep durations entered in seconds, then write
        data.window_duration and data.stride_duration in nanoseconds

Set WINDOW_MODE below, or override it with --window-mode. The active mode uses
its corresponding sweep lists. The inactive mode is still written with simple
valid default values so every generated YAML contains the complete window
configuration.

Typical usage
-------------
Use WINDOW_MODE and the sweep lists defined in this file:

    python generate_graph_vae_sweep.py

Force the stride to equal the window size/duration:

    python generate_graph_vae_sweep.py --same-stride

Override the configured mode:

    python generate_graph_vae_sweep.py --window-mode event
    python generate_graph_vae_sweep.py --window-mode time

Override the starting index:

    python generate_graph_vae_sweep.py --start 200
"""

from pathlib import Path
from itertools import product
from numbers import Real
from typing import Union
import argparse
import copy
import math
import yaml


# ============================================================
# User settings
# ============================================================

# Base YAML file to vary.
BASE_YAML_PATH = Path("configs/graph_vae.yaml")

# Directory where generated YAML files will be saved.
OUTPUT_YAML_DIR = Path("tuning_configs/graph_vae_sweep")

# Starting index for generated YAML names and run directories.
START_INDEX = 0

# Select which type of window sweep to generate:
#   "event" -> sweep EVENT_WINDOW_SIZES and EVENT_STRIDES
#   "time"  -> sweep TIME_WINDOW_DURATIONS and TIME_STRIDE_DURATIONS
WINDOW_MODE = "time"

# Event-window sweep values. Units are numbers of events.
EVENT_WINDOW_SIZES = [400]
EVENT_STRIDES = [10, 20, 50, 80, 100]

# Time-window sweep values entered in SECONDS.
#
# The generated YAML stores window_duration and stride_duration in nanoseconds
# because meta.time is measured in ns. Run names remain expressed in seconds.
#
# Examples:
#   0.00004 seconds = 40,000 ns
#   0.0004  seconds = 400,000 ns
TIME_WINDOW_DURATIONS_SECONDS = [2000, 20000, 40000, 80000, 100000]
TIME_STRIDE_DURATIONS_SECONDS = [200, 400, 1000]

# Conversion used when writing the generated YAML.
NANOSECONDS_PER_SECOND = 1_000_000_000

# Values written for the inactive mode. They are not used by the dataset, but
# keeping them in every YAML makes switching modes simple and predictable.
DEFAULT_EVENT_WINDOW_SIZE = 100
DEFAULT_EVENT_STRIDE = 100
# Inactive time-window defaults are also specified in seconds and converted
# to nanoseconds when written to YAML.
DEFAULT_TIME_WINDOW_DURATION_SECONDS = 5.0
DEFAULT_TIME_STRIDE_DURATION_SECONDS = 5.0

# Other sweep values.
ADJACENCY_RADII = [4]
BATCH_SIZES = [64]
LEARNING_RATES = [0.001]
BETAS = [0.5]

DEFAULT_WEIGHT_DECAY = 1.0e-4
DEFAULT_MAX_EPOCHS = 200

# Project-relative checkpoint base directory.
CHECKPOINT_BASE_DIR = Path("checkpoints/graph_vae")


# ============================================================
# CLI arguments
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GraphVAE event-window or time-window sweep YAML configs."
    )

    parser.add_argument(
        "--window-mode",
        choices=("event", "time"),
        default=None,
        help=(
            "Override WINDOW_MODE from this script. In event mode, sweep "
            "EVENT_WINDOW_SIZES/EVENT_STRIDES. In time mode, sweep "
            "TIME_WINDOW_DURATIONS_SECONDS/TIME_STRIDE_DURATIONS_SECONDS."
        ),
    )

    parser.add_argument(
        "--same-stride",
        "--same_stride",
        "--same",
        action="store_true",
        help=(
            "Set stride equal to window size in event mode, or set "
            "stride_duration equal to window_duration in time mode."
        ),
    )

    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help=(
            "Override START_INDEX for generated YAML filenames and run names. "
            "If omitted, START_INDEX from this script is used."
        ),
    )

    return parser.parse_args()


# ============================================================
# YAML and formatting helpers
# ============================================================

def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}

    if not isinstance(cfg, dict):
        raise ValueError(f"Base YAML did not load as a dictionary: {path}")

    return cfg


def save_yaml(config: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            default_flow_style=False,
        )


def format_number_for_name(value: Real) -> str:
    """Convert an integer or float to a compact filename-safe string."""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Non-finite value cannot be used in a run name: {value}")

    text = f"{numeric:g}"
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def yaml_number(value: Real) -> Union[int, float]:
    """Keep whole-number values as ints and other real values as floats."""
    numeric = float(value)
    if numeric.is_integer():
        return int(numeric)
    return numeric


def seconds_to_nanoseconds(seconds: Real) -> int:
    """Convert a duration in seconds to an integer number of nanoseconds."""
    numeric_seconds = float(seconds)
    nanoseconds = numeric_seconds * NANOSECONDS_PER_SECOND

    if not math.isfinite(nanoseconds):
        raise ValueError(f"Non-finite time duration: {seconds!r} seconds.")

    # Round rather than truncate so decimal inputs such as 0.00004 convert
    # reliably to exactly 40000 ns despite floating-point representation.
    rounded = round(nanoseconds)

    if rounded <= 0:
        raise ValueError(
            f"Time duration must convert to at least 1 ns; got {seconds!r} seconds."
        )

    return int(rounded)


# ============================================================
# Validation
# ============================================================

def validate_nonempty(name: str, values: list) -> None:
    if not values:
        raise ValueError(f"{name} cannot be empty.")


def validate_positive_numbers(name: str, values: list[Real]) -> None:
    validate_nonempty(name, values)
    for value in values:
        if not isinstance(value, Real) or isinstance(value, bool):
            raise TypeError(f"Every value in {name} must be numeric; got {value!r}.")
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"Every value in {name} must be finite and > 0; got {value!r}.")


def validate_sweep_settings(window_mode: str) -> None:
    if window_mode not in {"event", "time"}:
        raise ValueError(f"WINDOW_MODE must be 'event' or 'time'; got {window_mode!r}.")

    if window_mode == "event":
        validate_positive_numbers("EVENT_WINDOW_SIZES", EVENT_WINDOW_SIZES)
        validate_positive_numbers("EVENT_STRIDES", EVENT_STRIDES)

        for value in EVENT_WINDOW_SIZES + EVENT_STRIDES:
            if not float(value).is_integer():
                raise ValueError(
                    "EVENT_WINDOW_SIZES and EVENT_STRIDES must contain integer event counts; "
                    f"got {value!r}."
                )
    else:
        validate_positive_numbers(
            "TIME_WINDOW_DURATIONS_SECONDS",
            TIME_WINDOW_DURATIONS_SECONDS,
        )
        validate_positive_numbers(
            "TIME_STRIDE_DURATIONS_SECONDS",
            TIME_STRIDE_DURATIONS_SECONDS,
        )

    validate_nonempty("ADJACENCY_RADII", ADJACENCY_RADII)
    validate_positive_numbers("BATCH_SIZES", BATCH_SIZES)
    validate_positive_numbers("LEARNING_RATES", LEARNING_RATES)
    validate_nonempty("BETAS", BETAS)

    for radius in ADJACENCY_RADII:
        if not isinstance(radius, int) or isinstance(radius, bool) or radius < 0:
            raise ValueError(
                "Every value in ADJACENCY_RADII must be an integer >= 0; "
                f"got {radius!r}."
            )

    for batch_size in BATCH_SIZES:
        if not float(batch_size).is_integer():
            raise ValueError(
                f"Every value in BATCH_SIZES must be an integer; got {batch_size!r}."
            )

    for beta in BETAS:
        if not isinstance(beta, Real) or isinstance(beta, bool):
            raise TypeError(f"Every value in BETAS must be numeric; got {beta!r}.")
        if not math.isfinite(float(beta)) or float(beta) < 0:
            raise ValueError(f"Every value in BETAS must be finite and >= 0; got {beta!r}.")

    validate_positive_numbers(
        "inactive event-window defaults",
        [DEFAULT_EVENT_WINDOW_SIZE, DEFAULT_EVENT_STRIDE],
    )
    validate_positive_numbers(
        "inactive time-window defaults",
        [
            DEFAULT_TIME_WINDOW_DURATION_SECONDS,
            DEFAULT_TIME_STRIDE_DURATION_SECONDS,
        ],
    )


# ============================================================
# Config generation
# ============================================================

def get_active_window_values(
    window_mode: str,
    same_stride: bool,
) -> tuple[list[Real], list[Real]]:
    if window_mode == "event":
        window_values = EVENT_WINDOW_SIZES
        stride_values = EVENT_STRIDES
    else:
        window_values = TIME_WINDOW_DURATIONS_SECONDS
        stride_values = TIME_STRIDE_DURATIONS_SECONDS

    # The returned stride list is only used when same_stride is false.
    if same_stride:
        return window_values, []

    return window_values, stride_values


def make_run_name(
    index: int,
    window_mode: str,
    window_value: Real,
    stride_value: Real,
    adjacency_radius: int,
    batch_size: int,
    lr: float,
    beta: float,
) -> str:
    window_name = format_number_for_name(window_value)
    stride_name = format_number_for_name(stride_value)
    lr_name = format_number_for_name(lr)
    beta_name = format_number_for_name(beta)

    if window_mode == "event":
        window_part = f"win{window_name}_stride{stride_name}"
    else:
        window_part = f"time{window_name}s_tstride{stride_name}s"

    return (
        f"{index:04d}_"
        f"{window_part}_"
        f"rad{adjacency_radius}_"
        f"bs{batch_size}_"
        f"lr{lr_name}_"
        f"beta{beta_name}"
    )


def set_window_config(
    data_config: dict,
    window_mode: str,
    window_value: Real,
    stride_value: Real,
) -> None:
    """Write both modes; timed input values are seconds, YAML values are ns."""
    data_config["window_mode"] = window_mode

    if window_mode == "event":
        data_config["window_size"] = int(window_value)
        data_config["stride"] = int(stride_value)

        # Simple inactive-mode defaults.
        data_config["window_duration"] = seconds_to_nanoseconds(
            DEFAULT_TIME_WINDOW_DURATION_SECONDS
        )
        data_config["stride_duration"] = seconds_to_nanoseconds(
            DEFAULT_TIME_STRIDE_DURATION_SECONDS
        )
    else:
        # Simple inactive-mode defaults.
        data_config["window_size"] = int(DEFAULT_EVENT_WINDOW_SIZE)
        data_config["stride"] = int(DEFAULT_EVENT_STRIDE)

        # Timed sweep values are configured in seconds for readability, but
        # the actual GraphVAE config expects nanoseconds to match meta.time.
        data_config["window_duration"] = seconds_to_nanoseconds(window_value)
        data_config["stride_duration"] = seconds_to_nanoseconds(stride_value)


def make_config(
    base_config: dict,
    run_name: str,
    window_mode: str,
    window_value: Real,
    stride_value: Real,
    adjacency_radius: int,
    batch_size: int,
    lr: float,
    beta: float,
) -> dict:
    cfg = copy.deepcopy(base_config)

    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg.setdefault("training", {})
    cfg.setdefault("inference", {})

    cfg["model_type"] = "graph_vae"

    set_window_config(
        data_config=cfg["data"],
        window_mode=window_mode,
        window_value=window_value,
        stride_value=stride_value,
    )
    cfg["data"]["adjacency_radius"] = int(adjacency_radius)

    cfg["training"]["lr"] = float(lr)
    cfg["training"]["beta"] = float(beta)
    cfg["training"]["weight_decay"] = float(DEFAULT_WEIGHT_DECAY)
    cfg["training"]["batch_size"] = int(batch_size)
    cfg["training"]["max_epochs"] = int(DEFAULT_MAX_EPOCHS)

    checkpoint_dir = CHECKPOINT_BASE_DIR / run_name
    final_model_path = checkpoint_dir / "graph_vae_final.pt"

    cfg["training"]["checkpoint_dir"] = str(checkpoint_dir)
    cfg["training"]["output_path"] = str(final_model_path)

    cfg["inference"]["checkpoint_path"] = str(final_model_path)
    cfg["inference"]["output_path"] = str(checkpoint_dir / "inference_scores.npz")

    return cfg


def main() -> None:
    args = parse_args()

    window_mode = WINDOW_MODE if args.window_mode is None else args.window_mode
    validate_sweep_settings(window_mode)

    start_index = START_INDEX if args.start is None else int(args.start)
    if start_index < 0:
        raise ValueError(f"Starting index must be >= 0; got {start_index}.")

    OUTPUT_YAML_DIR.mkdir(parents=True, exist_ok=True)
    base_config = load_yaml(BASE_YAML_PATH)

    window_values, configured_stride_values = get_active_window_values(
        window_mode=window_mode,
        same_stride=args.same_stride,
    )

    index = start_index
    generated = 0
    skipped = 0

    for window_value in window_values:
        stride_values = [window_value] if args.same_stride else configured_stride_values

        for stride_value, adjacency_radius, batch_size, lr, beta in product(
            stride_values,
            ADJACENCY_RADII,
            BATCH_SIZES,
            LEARNING_RATES,
            BETAS,
        ):
            # Keep the original sweep behavior: overlapping or non-overlapping
            # windows are allowed, but a stride larger than the window is skipped.
            if float(stride_value) > float(window_value):
                active_window_key = (
                    "window_size" if window_mode == "event" else "window_duration"
                )
                active_stride_key = (
                    "stride" if window_mode == "event" else "stride_duration"
                )
                print(
                    "Skipping invalid combination: "
                    f"window_mode={window_mode}, "
                    f"{active_window_key}={window_value}, "
                    f"{active_stride_key}={stride_value}, "
                    f"adjacency_radius={adjacency_radius}, "
                    f"batch_size={batch_size}, lr={lr}, beta={beta} "
                    f"({active_stride_key} must be <= {active_window_key})"
                )
                skipped += 1
                continue

            run_name = make_run_name(
                index=index,
                window_mode=window_mode,
                window_value=window_value,
                stride_value=stride_value,
                adjacency_radius=int(adjacency_radius),
                batch_size=int(batch_size),
                lr=float(lr),
                beta=float(beta),
            )

            cfg = make_config(
                base_config=base_config,
                run_name=run_name,
                window_mode=window_mode,
                window_value=window_value,
                stride_value=stride_value,
                adjacency_radius=int(adjacency_radius),
                batch_size=int(batch_size),
                lr=float(lr),
                beta=float(beta),
            )

            output_yaml_path = OUTPUT_YAML_DIR / f"{run_name}.yaml"
            save_yaml(cfg, output_yaml_path)
            print(f"Wrote {output_yaml_path}")

            index += 1
            generated += 1

    print()
    print(f"Window mode: {window_mode}")
    if window_mode == "time":
        print("Timed sweep input unit: seconds")
        print("Generated YAML time unit: nanoseconds")
    print(f"Generated {generated} YAML files in {OUTPUT_YAML_DIR}")
    print(f"Skipped {skipped} invalid combinations")
    print(f"First index: {start_index:04d}")
    print(f"Last index: {index - 1:04d}" if generated > 0 else "No YAML files generated")


if __name__ == "__main__":
    main()
