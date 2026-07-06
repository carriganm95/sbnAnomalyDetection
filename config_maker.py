#!/usr/bin/env python3
"""
Generate GraphVAE sweep YAML configuration files.

This script reads one base GraphVAE YAML file, modifies selected sweep
parameters, and writes one YAML config per parameter combination. Each generated
config gets its own run name, checkpoint directory, final model path, and
inference output path.

Typical usage
-------------
Run the default sweep using the values set in this file:

    python generate_graph_vae_sweep.py

Run the sweep while forcing stride = window_size for every window size:

    python generate_graph_vae_sweep.py --same-stride

Input / output settings
-----------------------
BASE_YAML_PATH:
    Path to the base GraphVAE YAML config that will be copied and modified.

OUTPUT_YAML_DIR:
    Directory where generated YAML files will be written.

START_INDEX:
    Starting integer index used in generated YAML filenames and run names.

Sweep settings
--------------
WINDOW_SIZES:
    List of data.window_size values to test.

STRIDES:
    List of data.stride values to test.
    This list is ignored when --same-stride is used.

ADJACENCY_RADII:
    List of data.adjacency_radius values to test.

BATCH_SIZES:
    List of training.batch_size values to test.

LEARNING_RATES:
    List of training.lr values to test.

BETAS:
    List of training.beta values to test.

The sweep uses a full Cartesian product:

    WINDOW_SIZES × STRIDES × ADJACENCY_RADII × BATCH_SIZES × LEARNING_RATES × BETAS

When --same-stride is used, the sweep becomes:

    for each window_size:
        stride = window_size

    WINDOW_SIZES × ADJACENCY_RADII × BATCH_SIZES × LEARNING_RATES × BETAS

Validation behavior
-------------------
The script skips invalid combinations where:

    stride > window_size
    adjacency_radius < 0
    batch_size < 1
    beta < 0

Command-line flags
------------------
--same-stride:
    Ignore the STRIDES list and automatically set stride = window_size
    for every window size in WINDOW_SIZES.

--start:
    Override START_INDEX from the command line.
"""

from pathlib import Path
from itertools import product
import copy
import argparse
import yaml


# ============================================================
# User settings
# ============================================================

# Base YAML file to vary.
BASE_YAML_PATH = Path("configs/graph_vae.yaml")

# Directory where generated YAML files will be saved.
OUTPUT_YAML_DIR = Path("tuning_configs/graph_vae_sweep")

# Starting index for generated YAML names.
# Example:
#   START_INDEX = 0  -> 0000_win100_stride50_rad4_bs16_lr0p001_beta1.yaml
START_INDEX = 0

# Parameter values to sweep.
WINDOW_SIZES = [50, 100, 200]
STRIDES = [50, 100]
ADJACENCY_RADII = [4, 8]

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
        description="Generate GraphVAE sweep YAML configs."
    )

    parser.add_argument(
        "--same-stride",
        "--same_stride",
        "--same",
        action="store_true",
        help=(
            "Ignore STRIDES and automatically set stride = window_size "
            "for every window size in the sweep."
        ),
    )

    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help=(
            "Override START_INDEX for generated YAML filenames and run names. "
            "If omitted, the START_INDEX variable in this script is used."
        ),
    )

    return parser.parse_args()


# ============================================================
# YAML helpers
# ============================================================

def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}

    if not isinstance(cfg, dict):
        raise ValueError(f"Base YAML did not load as a dictionary: {path}")

    return cfg


def save_yaml(config: dict, path: Path) -> None:
    with path.open("w") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            default_flow_style=False,
        )


def format_float_for_name(value: float) -> str:
    """
    Convert a float to a filename-safe string.

    Examples:
        0.003 -> 0p003
        0.001 -> 0p001
        1e-4  -> 0p0001
        1.0   -> 1
    """
    text = f"{value:g}"
    text = text.replace(".", "p")
    text = text.replace("-", "m")
    return text


def validate_sweep_lists() -> None:
    """Validate sweep lists before generating configs."""
    if len(WINDOW_SIZES) == 0:
        raise ValueError("WINDOW_SIZES cannot be empty.")

    if len(STRIDES) == 0:
        raise ValueError("STRIDES cannot be empty.")

    if len(ADJACENCY_RADII) == 0:
        raise ValueError("ADJACENCY_RADII cannot be empty.")

    if len(BATCH_SIZES) == 0:
        raise ValueError("BATCH_SIZES cannot be empty.")

    if len(LEARNING_RATES) == 0:
        raise ValueError("LEARNING_RATES cannot be empty.")

    if len(BETAS) == 0:
        raise ValueError("BETAS cannot be empty.")


# ============================================================
# Config generation
# ============================================================

def make_run_name(
    index: int,
    window_size: int,
    stride: int,
    adjacency_radius: int,
    batch_size: int,
    lr: float,
    beta: float,
) -> str:
    lr_name = format_float_for_name(lr)
    beta_name = format_float_for_name(beta)

    return (
        f"{index:04d}_"
        f"win{window_size}_"
        f"stride{stride}_"
        f"rad{adjacency_radius}_"
        f"bs{batch_size}_"
        f"lr{lr_name}_"
        f"beta{beta_name}"
    )


def make_config(
    base_config: dict,
    run_name: str,
    window_size: int,
    stride: int,
    adjacency_radius: int,
    batch_size: int,
    lr: float,
    beta: float,
) -> dict:
    cfg = copy.deepcopy(base_config)

    # Make sure required sections exist.
    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg.setdefault("training", {})
    cfg.setdefault("inference", {})

    # Force this to be a GraphVAE config.
    cfg["model_type"] = "graph_vae"

    # Vary data graph/window settings.
    cfg["data"]["window_size"] = int(window_size)
    cfg["data"]["stride"] = int(stride)
    cfg["data"]["adjacency_radius"] = int(adjacency_radius)

    # Apply training settings.
    cfg["training"]["lr"] = float(lr)
    cfg["training"]["beta"] = float(beta)
    cfg["training"]["weight_decay"] = float(DEFAULT_WEIGHT_DECAY)
    cfg["training"]["batch_size"] = int(batch_size)
    cfg["training"]["max_epochs"] = int(DEFAULT_MAX_EPOCHS)

    # Give each run its own checkpoint directory.
    checkpoint_dir = CHECKPOINT_BASE_DIR / run_name

    cfg["training"]["checkpoint_dir"] = str(checkpoint_dir)
    cfg["training"]["output_path"] = str(checkpoint_dir / "graph_vae_final.pt")

    # Match inference paths to this run.
    cfg["inference"]["checkpoint_path"] = str(checkpoint_dir / "graph_vae_final.pt")
    cfg["inference"]["output_path"] = str(checkpoint_dir / "inference_scores.npz")

    return cfg


def main() -> None:
    args = parse_args()

    validate_sweep_lists()

    start_index = START_INDEX if args.start is None else int(args.start)

    OUTPUT_YAML_DIR.mkdir(parents=True, exist_ok=True)

    base_config = load_yaml(BASE_YAML_PATH)

    index = start_index
    generated = 0
    skipped = 0

    for window_size in WINDOW_SIZES:
        if args.same_stride:
            stride_values = [window_size]
        else:
            stride_values = STRIDES

        for stride, adjacency_radius, batch_size, lr, beta in product(
            stride_values,
            ADJACENCY_RADII,
            BATCH_SIZES,
            LEARNING_RATES,
            BETAS,
        ):
            if stride > window_size:
                print(
                    f"Skipping invalid combination: "
                    f"window_size={window_size}, stride={stride}, "
                    f"adjacency_radius={adjacency_radius}, "
                    f"batch_size={batch_size}, lr={lr}, beta={beta} "
                    f"(stride must be <= window_size)"
                )
                skipped += 1
                continue

            if adjacency_radius < 0:
                print(
                    f"Skipping invalid combination: "
                    f"window_size={window_size}, stride={stride}, "
                    f"adjacency_radius={adjacency_radius}, "
                    f"batch_size={batch_size}, lr={lr}, beta={beta} "
                    f"(adjacency_radius must be >= 0)"
                )
                skipped += 1
                continue

            if batch_size < 1:
                print(
                    f"Skipping invalid combination: "
                    f"window_size={window_size}, stride={stride}, "
                    f"adjacency_radius={adjacency_radius}, "
                    f"batch_size={batch_size}, lr={lr}, beta={beta} "
                    f"(batch_size must be >= 1)"
                )
                skipped += 1
                continue

            if beta < 0:
                print(
                    f"Skipping invalid combination: "
                    f"window_size={window_size}, stride={stride}, "
                    f"adjacency_radius={adjacency_radius}, "
                    f"batch_size={batch_size}, lr={lr}, beta={beta} "
                    f"(beta must be >= 0)"
                )
                skipped += 1
                continue

            run_name = make_run_name(
                index=index,
                window_size=window_size,
                stride=stride,
                adjacency_radius=adjacency_radius,
                batch_size=batch_size,
                lr=lr,
                beta=beta,
            )

            cfg = make_config(
                base_config=base_config,
                run_name=run_name,
                window_size=window_size,
                stride=stride,
                adjacency_radius=adjacency_radius,
                batch_size=batch_size,
                lr=lr,
                beta=beta,
            )

            output_yaml_path = OUTPUT_YAML_DIR / f"{run_name}.yaml"
            save_yaml(cfg, output_yaml_path)

            print(f"Wrote {output_yaml_path}")

            index += 1
            generated += 1

    print()
    print(f"Generated {generated} YAML files in {OUTPUT_YAML_DIR}")
    print(f"Skipped {skipped} invalid combinations")
    print(f"First index: {start_index:04d}")
    print(f"Last index: {index - 1:04d}" if generated > 0 else "No YAML files generated")


if __name__ == "__main__":
    main()