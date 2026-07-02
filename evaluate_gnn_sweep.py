#!/usr/bin/env python3
"""Evaluate GNN inference_scores.npz files with threshold-based good/bad prediction.

Expected layout:

    checkpoints/gnn/MODEL_NAME/inference_scores.npz

Each NPZ should contain at least:

    scores
    scores_max
    first_run

For every model directory, this script:
  1. reads inference_scores.npz
  2. transforms scores and scores_max according to NORMALIZATION_MODE
  3. applies each threshold in THRESHOLDS
     - for tanh/sigmoid/global_max, thresholds are normalized values in [0, 1]
     - for none, thresholds are absolute raw score values
  4. evaluates window-level predictions against known good/bad run labels
  5. writes a compact per-model metrics CSV by default
  6. writes detailed full/per-run CSV files only with --full-summary
  7. optionally restricts output to scores-only or scores_max-only with CLI flags
  8. optionally ranks CSV output by precision, recall, F1, or accuracy
  9. optionally changes output sequence with --threshold-first
 10. optionally runs the plotting script afterward with --with-plot

Available command-line flags:

    --full-summary
        Also write the full detailed summary CSV and the per-run ratio CSV.
        Without this flag, only the compact summary CSV is written.

    --scores-only
        Only evaluate and store the scores_only method.

    --scores-max-only
        Only evaluate and store the scores_max_only method.

    --precision-rank
        Sort the output CSV rows by precision from highest to lowest.

    --recall-rank
        Sort the output CSV rows by recall from highest to lowest.

    --f1-rank
        Sort the output CSV rows by F1 score from highest to lowest.

    --accuracy-rank
        Sort the output CSV rows by accuracy from highest to lowest.

    --threshold-first
        Change the unranked output order. By default, the script evaluates all
        thresholds for one model before moving to the next model. With this flag,
        it evaluates all models for one threshold before moving to the next threshold.

    --start N
        Only evaluate/plot model directories whose leading numeric prefix is
        greater than or equal to N. This overrides MODEL_START_INDEX set in this
        script. Example: --start 81 uses 0081_..., 0082_..., etc.

    --with-plot
        After writing threshold_metrics_summary.csv, run the plotting scripts located
        in the graphing/ directory relative to this script. This includes the
        existing sweep-metrics plots and the good/bad score distribution plots.

Notes:
  - --scores-only and --scores-max-only are mutually exclusive.
  - The rank flags are mutually exclusive.
  - If no method-selection flag is given, the script evaluates:
        scores_only
        scores_max_only
        both_<BOTH_RULE>
  - If no rank flag is given, rows are written in the selected scan/order sequence.
  - --threshold-first only changes row order when no rank flag is used.
  - --start takes priority over MODEL_START_INDEX.

Example usage:

    python evaluate_gnn_sweep.py
    python evaluate_gnn_sweep.py --full-summary
    python evaluate_gnn_sweep.py --scores-only
    python evaluate_gnn_sweep.py --scores-max-only
    python evaluate_gnn_sweep.py --scores-only --f1-rank
    python evaluate_gnn_sweep.py --full-summary --accuracy-rank
    python evaluate_gnn_sweep.py --threshold-first
    python evaluate_gnn_sweep.py --start 81
    python evaluate_gnn_sweep.py --start 81 --with-plot

Definitions:
  - true bad window: first_run is in bad_runs
  - true good window: first_run is in good_runs
  - predicted bad: transformed score > threshold
  - predicted good: transformed score < threshold
  - when NORMALIZATION_MODE = "none", the transformed score is the raw score
  - values exactly equal to threshold are controlled by equal_is_bad

Confusion matrix convention:
  TP = true bad, predicted bad
  TN = true good, predicted good
  FP = true good, predicted bad
  FN = true bad, predicted good
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Literal

import numpy as np
import yaml


# ============================================================
# User settings
# ============================================================

CHECKPOINTS_GNN_DIR = Path("checkpoints/gnn")
NPZ_NAME = "inference_scores.npz"

# Only evaluate/plot model directories whose numeric prefix is >= this value.
# Example:
#   MODEL_START_INDEX = None  -> use all model directories
#   MODEL_START_INDEX = 81    -> use 0081_..., 0082_..., ..., 0160_...
MODEL_START_INDEX: int | None = None

OUTPUT_DIR = Path("threshold_evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Plotting script locations, relative to this file.
# These are run when --with-plot is used.
PLOT_SWEEP_METRICS_SCRIPT_RELATIVE = Path("graphing/plot_sweep_metrics.py")
PLOT_GOOD_BAD_DISTRIBUTION_SCRIPT_RELATIVE = Path(
    "graphing/plot_good_bad_scores.py"
)

# Output directory for good/bad score distribution plots.
GOOD_BAD_DISTRIBUTION_OUTPUT_DIR = Path(
    "/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection/"
    "threshold_evaluation/plots/04_good_bad_distribution"
)

# Extra arguments always passed to the sweep-metrics plotting script when --with-plot is used.
# Example: PLOT_SWEEP_METRICS_EXTRA_ARGS = ["--full-summary"]
PLOT_SWEEP_METRICS_EXTRA_ARGS: list[str] = []

# Extra arguments always passed to the good/bad distribution plotting script when --with-plot is used.
# Example: PLOT_GOOD_BAD_DISTRIBUTION_EXTRA_ARGS = ["--bins", "100"]
PLOT_GOOD_BAD_DISTRIBUTION_EXTRA_ARGS: list[str] = []

# Choose normalization transform.
# Options: "tanh", "sigmoid", "global_max", "none"
NORMALIZATION_MODE = "tanh"

# Scope used to determine the transform scale.
# "per_model": each model's scores are normalized using only that model's NPZ values.
# "global": scores are normalized using all model NPZ values together.
NORMALIZATION_SCOPE = "per_model"

# Scale mode for tanh/sigmoid.
# "max": use the maximum finite value as the scale reference.
# "percentile": use NORMALIZATION_SCALE_PERCENTILE as the scale reference.
# "manual": use MANUAL_SCORE_SCALE and MANUAL_SCORE_MAX_SCALE.
NORMALIZATION_SCALE_MODE = "max"
NORMALIZATION_SCALE_PERCENTILE = 99.0
MANUAL_SCORE_SCALE = 1.0
MANUAL_SCORE_MAX_SCALE = 1.0

# After tanh/sigmoid transform, rescale so the largest finite value maps to 1.
# This keeps the threshold range exactly meaningful as [0, 1].
RESCALE_TRANSFORM_TO_UNIT_MAX = True

# Classification thresholds.
# predicted bad if transformed value > threshold by default.
# For NORMALIZATION_MODE in {"tanh", "sigmoid", "global_max"}, thresholds are normalized
# values and must be in [0, 1].
# For NORMALIZATION_MODE = "none", thresholds are absolute raw score values.
# The script evaluates every model at every threshold in this list.
THRESHOLDS = [0.5]

# What to do with values exactly equal to the threshold.
# False: value == threshold is predicted good.
# True:  value == threshold is predicted bad.
equal_is_bad = False

# How to combine scores and scores_max for the "both" evaluation.
# "or":   predicted bad if scores OR scores_max is above threshold.
# "and":  predicted bad if scores AND scores_max are above threshold.
# "mean": predicted bad if mean(scores, scores_max) is above threshold.
# "max":  predicted bad if max(scores, scores_max) is above threshold.
BOTH_RULE = "or"

# Known labels from run number.
good_runs = {
    18445, 19724, 20141, 20142, 20144
}

bad_runs = {
    19627, 19946, 20104
}

# How to handle runs that are not in good_runs or bad_runs.
#
# Options:
#   "ignore" -> unknown runs are excluded from evaluation
#   "good"   -> unknown runs are treated as true good/normal
#   "bad"    -> unknown runs are treated as true bad/anomalous
#
# For anomaly detection, "good" is often useful when bad_runs is a curated list
# of known bad runs and everything else should be treated as normal.
UNKNOWN_RUN_POLICY = "good"


# ============================================================
# Normalization helpers
# ============================================================

def finite_1d(a: np.ndarray) -> np.ndarray:
    """Return finite values as a flattened float array."""
    a = np.asarray(a, dtype=float).ravel()
    return a[np.isfinite(a)]


def safe_positive_scale(scale: float, fallback_values: np.ndarray, name: str) -> float:
    """Make sure a transform scale is finite and positive."""
    if np.isfinite(scale) and scale > 0:
        return float(scale)

    finite = finite_1d(fallback_values)
    if finite.size > 0:
        max_value = float(np.nanmax(finite))
        if np.isfinite(max_value) and max_value > 0:
            print(f"WARNING: bad {name} scale={scale}; falling back to max={max_value}")
            return max_value

    print(f"WARNING: bad {name} scale={scale}; falling back to 1.0")
    return 1.0


def stable_sigmoid_shifted(z: np.ndarray) -> np.ndarray:
    """Shifted sigmoid mapping z>=0 approximately to [0, 1)."""
    z = np.asarray(z, dtype=float)
    z_clip = np.clip(z, -700, 700)
    return 2.0 / (1.0 + np.exp(-z_clip)) - 1.0


def transform_scores(
    values: np.ndarray,
    *,
    mode: str,
    scale: float,
    reference_max: float,
    rescale_to_unit_max: bool = True,
) -> np.ndarray:
    """Transform scores for thresholding.

    For nonnegative anomaly scores:
      - none: raw x, so thresholds are absolute raw score values
      - global_max: x / reference_max, so thresholds are in [0, 1]
      - tanh: tanh(x / scale), optionally divided by tanh(reference_max / scale)
      - sigmoid: 2*sigmoid(x / scale)-1, optionally divided by transform(reference_max)

    For tanh/sigmoid/global_max, the optional final division makes the largest
    reference value map to exactly 1.
    """
    x = np.asarray(values, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    finite_mask = np.isfinite(x)

    if not np.any(finite_mask):
        return out

    scale = safe_positive_scale(scale, x[finite_mask], "normalization")
    reference_max = safe_positive_scale(reference_max, x[finite_mask], "reference_max")

    xf = x[finite_mask]

    if mode == "none":
        transformed = xf.copy()
        denom = 1.0
    elif mode == "global_max":
        transformed = xf / reference_max
        denom = 1.0
    elif mode == "tanh":
        transformed = np.tanh(xf / scale)
        denom = np.tanh(reference_max / scale) if rescale_to_unit_max else 1.0
    elif mode == "sigmoid":
        transformed = stable_sigmoid_shifted(xf / scale)
        denom = stable_sigmoid_shifted(np.asarray([reference_max / scale]))[0] if rescale_to_unit_max else 1.0
    else:
        raise ValueError(
            f"Unknown NORMALIZATION_MODE={mode!r}. "
            "Use 'tanh', 'sigmoid', 'global_max', or 'none'."
        )

    if not np.isfinite(denom) or denom <= 0:
        denom = 1.0

    transformed = transformed / denom

    # Small numerical overshoots can happen after the division.
    if mode != "none":
        transformed = np.clip(transformed, 0.0, 1.0)

    out[finite_mask] = transformed
    return out


def choose_scale(values: np.ndarray, *, scale_mode: str, percentile: float, manual_scale: float) -> float:
    finite = finite_1d(values)
    if finite.size == 0:
        return 1.0

    if scale_mode == "max":
        return float(np.nanmax(finite))
    if scale_mode == "percentile":
        return float(np.nanpercentile(finite, percentile))
    if scale_mode == "manual":
        return float(manual_scale)

    raise ValueError(
        f"Unknown NORMALIZATION_SCALE_MODE={scale_mode!r}. "
        "Use 'max', 'percentile', or 'manual'."
    )


def predict_bad(transformed_values: np.ndarray, threshold: float, *, equal_is_bad: bool) -> np.ndarray:
    """Return boolean predicted-bad mask.

    transformed_values are raw scores when NORMALIZATION_MODE == "none", and
    normalized values when NORMALIZATION_MODE is tanh/sigmoid/global_max.
    """
    if equal_is_bad:
        return transformed_values >= threshold
    return transformed_values > threshold


# ============================================================
# Evaluation helpers
# ============================================================

def confusion_and_metrics(y_true_bad: np.ndarray, y_pred_bad: np.ndarray) -> Dict[str, float]:
    """Compute confusion matrix and derived metrics."""
    y_true_bad = np.asarray(y_true_bad, dtype=bool)
    y_pred_bad = np.asarray(y_pred_bad, dtype=bool)

    tp = int(np.sum(y_true_bad & y_pred_bad))
    tn = int(np.sum((~y_true_bad) & (~y_pred_bad)))
    fp = int(np.sum((~y_true_bad) & y_pred_bad))
    fn = int(np.sum(y_true_bad & (~y_pred_bad)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else math.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else math.nan
    f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0 else math.nan
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else math.nan

    # Useful rates by true class.
    bad_run_pred_bad_ratio = recall
    good_run_pred_bad_ratio = fp / (fp + tn) if (fp + tn) > 0 else math.nan
    bad_run_pred_good_ratio = fn / (tp + fn) if (tp + fn) > 0 else math.nan
    good_run_pred_good_ratio = tn / (fp + tn) if (fp + tn) > 0 else math.nan

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "accuracy": accuracy,
        "true_bad_pred_bad_ratio": bad_run_pred_bad_ratio,
        "true_bad_pred_good_ratio": bad_run_pred_good_ratio,
        "true_good_pred_bad_ratio": good_run_pred_bad_ratio,
        "true_good_pred_good_ratio": good_run_pred_good_ratio,
    }


def build_true_labels(first_run: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return true labels, evaluation mask, and run-label debug masks.

    Label convention:
      - y_true_bad=True means true bad/anomalous
      - y_true_bad=False means true good/normal

    UNKNOWN_RUN_POLICY controls runs not listed in good_runs or bad_runs:
      - "ignore": exclude unknown runs from evaluation
      - "good":   treat unknown runs as true good
      - "bad":    treat unknown runs as true bad
    """
    runs = np.asarray(first_run).astype(int)

    is_explicit_good = np.isin(runs, list(good_runs))
    is_explicit_bad = np.isin(runs, list(bad_runs))
    is_unknown = ~(is_explicit_good | is_explicit_bad)

    policy = UNKNOWN_RUN_POLICY.lower()

    if policy == "ignore":
        eval_mask = is_explicit_good | is_explicit_bad
        y_true_bad = is_explicit_bad
        is_default_good = np.zeros_like(runs, dtype=bool)
        is_default_bad = np.zeros_like(runs, dtype=bool)

    elif policy == "good":
        eval_mask = np.ones_like(runs, dtype=bool)
        y_true_bad = is_explicit_bad
        is_default_good = is_unknown
        is_default_bad = np.zeros_like(runs, dtype=bool)

    elif policy == "bad":
        eval_mask = np.ones_like(runs, dtype=bool)
        y_true_bad = is_explicit_bad | is_unknown
        is_default_good = np.zeros_like(runs, dtype=bool)
        is_default_bad = is_unknown

    else:
        raise ValueError(
            "UNKNOWN_RUN_POLICY must be one of: 'ignore', 'good', 'bad'; "
            f"got {UNKNOWN_RUN_POLICY!r}"
        )

    label_info = {
        "is_explicit_good": is_explicit_good,
        "is_explicit_bad": is_explicit_bad,
        "is_unknown": is_unknown,
        "is_default_good": is_default_good,
        "is_default_bad": is_default_bad,
    }

    return y_true_bad, eval_mask, label_info


def combine_predictions(
    pred_score_bad: np.ndarray,
    pred_score_max_bad: np.ndarray,
    norm_score: np.ndarray,
    norm_score_max: np.ndarray,
    threshold: float,
    *,
    both_rule: str,
    equal_is_bad: bool,
) -> np.ndarray:
    """Combine scores and scores_max into a single predicted-bad mask."""
    both_rule = both_rule.lower()

    if both_rule == "or":
        return pred_score_bad | pred_score_max_bad
    if both_rule == "and":
        return pred_score_bad & pred_score_max_bad
    if both_rule == "mean":
        mean_value = 0.5 * (norm_score + norm_score_max)
        return predict_bad(mean_value, threshold, equal_is_bad=equal_is_bad)
    if both_rule == "max":
        max_value = np.maximum(norm_score, norm_score_max)
        return predict_bad(max_value, threshold, equal_is_bad=equal_is_bad)

    raise ValueError("BOTH_RULE must be one of: 'or', 'and', 'mean', 'max'")


# ============================================================
# Helpers for run names, config dicts, and model directory discovery
# ============================================================

def get_model_index(model_dir_name: str) -> int | None:
    """
    Extract the leading numeric model index from a model directory name.

    Examples:
        0000_win20_stride10_hist1 -> 0
        0081_win100_stride10_hist1 -> 81
        test_model -> None
    """
    prefix = model_dir_name.split("_", 1)[0]

    if not prefix.isdigit():
        return None

    return int(prefix)


# ============================================================
# File discovery and main loop
# ============================================================

def find_npz_files(
    checkpoints_dir: Path,
    npz_name: str,
    *,
    model_start_index: int | None = None,
) -> list[Path]:
    """Find checkpoints/gnn/MODEL_NAME/inference_scores.npz files."""
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {checkpoints_dir}")

    npz_files = []

    for child in sorted(checkpoints_dir.iterdir()):
        if not child.is_dir():
            continue

        model_index = get_model_index(child.name)

        if model_start_index is not None:
            if model_index is None:
                print(
                    f"Skipping {child.name}: cannot read numeric prefix "
                    f"while RESOLVED_START_INDEX={model_start_index}"
                )
                continue

            if model_index < model_start_index:
                continue

        npz_path = child / npz_name
        if npz_path.exists():
            npz_files.append(npz_path)

    return npz_files


def load_required_arrays(npz_path: Path) -> dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    required = ["scores", "scores_max", "first_run"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{npz_path} is missing required keys: {missing}")

    scores = np.asarray(data["scores"], dtype=float).ravel()
    scores_max = np.asarray(data["scores_max"], dtype=float).ravel()
    first_run = np.asarray(data["first_run"]).ravel()

    if not (scores.shape == scores_max.shape == first_run.shape):
        raise ValueError(
            f"Shape mismatch in {npz_path}: "
            f"scores={scores.shape}, scores_max={scores_max.shape}, first_run={first_run.shape}"
        )

    return {
        "scores": scores,
        "scores_max": scores_max,
        "first_run": first_run,
    }


def load_training_params_from_model_dir(model_dir: Path) -> dict[str, float | int | str]:
    """Read batch_size and learning_rate from the model's copied YAML config."""
    candidate_configs = [
        model_dir / "config_run.yaml",
        model_dir / "config_original.yaml",
        model_dir / f"{model_dir.name}.yaml",
    ]

    for config_path in candidate_configs:
        if not config_path.exists():
            continue

        try:
            with config_path.open("r") as f:
                config = yaml.safe_load(f) or {}
        except Exception as exc:
            print(f"WARNING: failed to read config {config_path}: {exc}")
            continue

        training = config.get("training", {}) or {}

        return {
            "batch_size": training.get("batch_size", ""),
            "learning_rate": training.get("lr", ""),
        }

    print(f"WARNING: no copied YAML config found in {model_dir}; batch_size/lr left blank")
    return {
        "batch_size": "",
        "learning_rate": "",
    }


def write_empty_compact_summary_csv(path: Path) -> None:
    """Write an empty compact summary CSV so old results are not accidentally reused."""
    compact_summary_fields = [
        "model_name",
        "batch_size",
        "learning_rate",
        "method",
        "threshold",
        "normalization_mode",
        "threshold_units",
        "TP", "TN", "FP", "FN",
        "precision", "recall", "F1", "accuracy",
    ]

    write_csv(path, [], compact_summary_fields)


def write_csv(path: Path, rows: list[dict], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_subprocess_command(cmd: list[str], *, title: str) -> None:
    """Run one subprocess command with readable logging."""
    print()
    print("=" * 80)
    print(title)
    print(" ".join(cmd))
    print("=" * 80)

    subprocess.run(cmd, check=True)


def run_plotting_script(args: argparse.Namespace, *, model_start_index: int | None) -> None:
    """Run all plotting scripts requested by --with-plot.

    This runs:
      1. graphing/plot_sweep_metrics.py
      2. graphing/plot_good_bad_scores.py

    The good/bad score distribution script receives --start so that plotting uses
    the same model-start filter as the threshold evaluation.
    """
    script_dir = Path(__file__).resolve().parent

    # ------------------------------------------------------------
    # 1. Existing threshold/sweep metrics plotting script
    # ------------------------------------------------------------
    sweep_plot_script = script_dir / PLOT_SWEEP_METRICS_SCRIPT_RELATIVE

    if not sweep_plot_script.exists():
        raise FileNotFoundError(
            f"Sweep-metrics plotting script does not exist: {sweep_plot_script}"
        )

    sweep_plot_cmd = [
        sys.executable,
        str(sweep_plot_script),
        *PLOT_SWEEP_METRICS_EXTRA_ARGS,
    ]

    # Forward method-selection flags because they affect which CSV rows are plotted.
    if args.scores_only:
        sweep_plot_cmd.append("--scores-only")

    if args.scores_max_only:
        sweep_plot_cmd.append("--scores-max-only")

    # Forward only the plotting summary flag, not evaluator --full-summary.
    if args.plot_summary:
        sweep_plot_cmd.append("--plot-summary")

    run_subprocess_command(
        sweep_plot_cmd,
        title="Running sweep-metrics plotting script:",
    )

    # ------------------------------------------------------------
    # 2. Good/bad score distribution plotting script
    # ------------------------------------------------------------
    good_bad_plot_script = script_dir / PLOT_GOOD_BAD_DISTRIBUTION_SCRIPT_RELATIVE

    if not good_bad_plot_script.exists():
        raise FileNotFoundError(
            f"Good/bad distribution plotting script does not exist: {good_bad_plot_script}"
        )

    checkpoints_dir = (script_dir / CHECKPOINTS_GNN_DIR).resolve()

    good_bad_plot_cmd = [
        sys.executable,
        str(good_bad_plot_script),
        str(checkpoints_dir),
        "--output-dir",
        str(GOOD_BAD_DISTRIBUTION_OUTPUT_DIR),
        *PLOT_GOOD_BAD_DISTRIBUTION_EXTRA_ARGS,
    ]

    # Forward method-selection flags.
    # In the distribution script:
    #   --scores-only      means plot only scores
    #   --scores-max-only  means plot only scores_max
    #   neither flag       means plot both
    if args.scores_only:
        good_bad_plot_cmd.append("--scores-only")

    if args.scores_max_only:
        good_bad_plot_cmd.append("--scores-max-only")

    # Forward the resolved start index so evaluation and distribution plots
    # use the same model subset. This covers both --start and MODEL_START_INDEX.
    if model_start_index is not None:
        good_bad_plot_cmd.extend(["--start", str(model_start_index)])

    run_subprocess_command(
        good_bad_plot_cmd,
        title="Running good/bad score distribution plotting script:",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoints/gnn/*/inference_scores.npz with threshold-based good/bad prediction."
    )
    parser.add_argument(
        "--full-summary",
        "--full_summary",
        action="store_true",
        help=(
            "Also write the full detailed CSV files. By default, only a compact "
            "summary CSV is written."
        ),
    )
    parser.add_argument(
        "--threshold-first",
        "--threshold_first",
        action="store_true",
        help=(
            "Change the unranked CSV row order. By default, the script writes all "
            "thresholds for one model before moving to the next model. With this flag, "
            "it writes all models for one threshold before moving to the next threshold. "
            "If a rank flag is used, ranking overrides this order."
        ),
    )
    parser.add_argument(
        "--with-plot",
        "--with_plot",
        "--with-plots",
        "--with_plots",
        action="store_true",
        help=(
            "After writing the compact summary CSV, run the plotting script in the "
            "graphing/ directory relative to this script."
        ),
    )

    parser.add_argument(
        "--plot-summary",
        "--plot_summary",
        action="store_true",
        help=(
            "Only used with --with-plot. Forward --plot-summary to the plotting "
            "script so it also writes its own plot summary CSV files."
        ),
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help=(
            "Only evaluate/plot model directories whose leading numeric prefix is "
            "greater than or equal to this value. This overrides MODEL_START_INDEX "
            "set in the script. Example: --start 81 uses 0081_..., 0082_..., etc."
        ),
    )

    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument(
        "--scores-only",
        "--scores_only",
        action="store_true",
        help=(
            "Only evaluate and store the scores_only method. "
            "If neither --scores-only nor --scores-max-only is given, all methods are evaluated."
        ),
    )
    method_group.add_argument(
        "--scores-max-only",
        "--scores_max_only",
        action="store_true",
        help=(
            "Only evaluate and store the scores_max_only method. "
            "If neither --scores-only nor --scores-max-only is given, all methods are evaluated."
        ),
    )

    rank_group = parser.add_mutually_exclusive_group()
    rank_group.add_argument(
        "--precision-rank",
        "--precision_rank",
        action="store_true",
        help="Sort output CSV rows by precision from high to low.",
    )
    rank_group.add_argument(
        "--recall-rank",
        "--recall_rank",
        action="store_true",
        help="Sort output CSV rows by recall from high to low.",
    )
    rank_group.add_argument(
        "--f1-rank",
        "--f1_rank",
        action="store_true",
        help="Sort output CSV rows by F1 from high to low.",
    )
    rank_group.add_argument(
        "--accuracy-rank",
        "--accuracy_rank",
        action="store_true",
        help="Sort output CSV rows by accuracy from high to low.",
    )
    return parser.parse_args()


def resolve_model_start_index(args: argparse.Namespace) -> int | None:
    """
    Decide which model index to start from.

    Priority:
      1. --start command-line argument
      2. MODEL_START_INDEX variable in this script
      3. None, meaning no filtering
    """
    if args.start is not None:
        if args.start < 0:
            raise ValueError(f"--start must be >= 0; got {args.start}")
        return args.start

    if MODEL_START_INDEX is not None and MODEL_START_INDEX < 0:
        raise ValueError(f"MODEL_START_INDEX must be >= 0; got {MODEL_START_INDEX}")

    return MODEL_START_INDEX


def selected_rank_metric(args: argparse.Namespace) -> str | None:
    """Return the requested ranking metric, or None for original processing order."""
    if args.precision_rank:
        return "precision"
    if args.recall_rank:
        return "recall"
    if args.f1_rank:
        return "F1"
    if args.accuracy_rank:
        return "accuracy"
    return None


def sort_rows_by_metric(rows: list[dict], metric: str | None) -> list[dict]:
    """Sort rows by a metric descending, keeping NaN/invalid values at the bottom."""
    if metric is None:
        return rows

    def sort_key(row: dict) -> tuple[int, float, str, float, str]:
        value = row.get(metric, math.nan)
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            value_float = math.nan

        invalid_flag = 0 if np.isfinite(value_float) else 1
        sortable_value = -value_float if invalid_flag == 0 else 0.0

        threshold = row.get("threshold", math.nan)
        try:
            threshold_float = float(threshold)
        except (TypeError, ValueError):
            threshold_float = math.nan

        return (
            invalid_flag,
            sortable_value,
            str(row.get("method", "")),
            threshold_float if np.isfinite(threshold_float) else math.inf,
            str(row.get("model_name", "")),
        )

    return sorted(rows, key=sort_key)


def threshold_uses_raw_scores() -> bool:
    """Return True when thresholds should be interpreted in raw score units."""
    return NORMALIZATION_MODE == "none"


def threshold_unit_label() -> str:
    """Human-readable label for threshold units."""
    if threshold_uses_raw_scores():
        return "raw_score_absolute"
    return "normalized_0_to_1"


def get_thresholds() -> list[float]:
    """Return the user-configured thresholds as a validated float list.

    Threshold meaning depends on NORMALIZATION_MODE:
      - none: absolute raw score values; only finiteness is required
      - tanh/sigmoid/global_max: normalized values in [0, 1]
    """
    try:
        thresholds = list(THRESHOLDS)
    except TypeError:
        thresholds = [THRESHOLDS]

    if len(thresholds) == 0:
        raise ValueError("THRESHOLDS must contain at least one threshold")

    thresholds = [float(t) for t in thresholds]

    if threshold_uses_raw_scores():
        for threshold in thresholds:
            if not np.isfinite(threshold):
                raise ValueError(
                    f"Every raw-score threshold in THRESHOLDS must be finite; got {threshold}"
                )
    else:
        for threshold in thresholds:
            if not (0.0 <= threshold <= 1.0):
                raise ValueError(
                    "Every normalized threshold in THRESHOLDS must be between 0 and 1 "
                    f"when NORMALIZATION_MODE={NORMALIZATION_MODE!r}; got {threshold}"
                )

    return thresholds


def build_evaluations_for_threshold(
    *,
    norm_scores: np.ndarray,
    norm_scores_max: np.ndarray,
    threshold: float,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    """Build selected prediction masks for one threshold."""
    pred_score_bad = predict_bad(norm_scores, threshold, equal_is_bad=equal_is_bad)
    pred_score_max_bad = predict_bad(norm_scores_max, threshold, equal_is_bad=equal_is_bad)
    pred_both_bad = combine_predictions(
        pred_score_bad,
        pred_score_max_bad,
        norm_scores,
        norm_scores_max,
        threshold,
        both_rule=BOTH_RULE,
        equal_is_bad=equal_is_bad,
    )

    if args.scores_only:
        return {"scores_only": pred_score_bad}

    if args.scores_max_only:
        return {"scores_max_only": pred_score_max_bad}

    return {
        "scores_only": pred_score_bad,
        "scores_max_only": pred_score_max_bad,
        f"both_{BOTH_RULE}": pred_both_bad,
    }


def prepare_model_arrays(
    *,
    model_name: str,
    arrays: dict[str, np.ndarray],
    global_score_scale: float | None,
    global_score_max_scale: float | None,
    global_score_ref_max: float | None,
    global_score_max_ref_max: float | None,
) -> dict | None:
    """Clean, label, and normalize one model's arrays once.

    Threshold-dependent predictions are made later from the normalized arrays.
    """
    scores = arrays["scores"]
    scores_max = arrays["scores_max"]
    first_run = arrays["first_run"].astype(int)

    finite_mask = np.isfinite(scores) & np.isfinite(scores_max) & np.isfinite(first_run)
    scores = scores[finite_mask]
    scores_max = scores_max[finite_mask]
    first_run = first_run[finite_mask]

    y_true_bad, eval_mask, label_info = build_true_labels(first_run)

    if not np.any(eval_mask):
        print(
            f"WARNING: {model_name}: no windows selected for evaluation "
            f"with UNKNOWN_RUN_POLICY={UNKNOWN_RUN_POLICY!r}; skipping metrics."
        )
        return None

    n_explicit_good = int(np.sum(label_info["is_explicit_good"] & eval_mask))
    n_explicit_bad = int(np.sum(label_info["is_explicit_bad"] & eval_mask))
    n_unknown_total = int(np.sum(label_info["is_unknown"]))
    n_default_good = int(np.sum(label_info["is_default_good"] & eval_mask))
    n_default_bad = int(np.sum(label_info["is_default_bad"] & eval_mask))
    n_true_good = int(np.sum((~y_true_bad) & eval_mask))
    n_true_bad = int(np.sum(y_true_bad & eval_mask))

    print(
        f"{model_name}: label policy={UNKNOWN_RUN_POLICY!r}; "
        f"explicit_good={n_explicit_good}, explicit_bad={n_explicit_bad}, "
        f"unknown_total={n_unknown_total}, default_good={n_default_good}, "
        f"default_bad={n_default_bad}, true_good_eval={n_true_good}, "
        f"true_bad_eval={n_true_bad}"
    )

    if n_true_good == 0:
        print(
            f"WARNING: {model_name}: no true-good windows were evaluated. "
            "TN and FP will both be 0, so accuracy/F1 may be misleading."
        )

    if n_true_bad == 0:
        print(
            f"WARNING: {model_name}: no true-bad windows were evaluated. "
            "TP and FN will both be 0, so metrics may be misleading."
        )

    if NORMALIZATION_SCOPE == "per_model":
        score_scale = choose_scale(
            scores,
            scale_mode=NORMALIZATION_SCALE_MODE,
            percentile=NORMALIZATION_SCALE_PERCENTILE,
            manual_scale=MANUAL_SCORE_SCALE,
        )
        score_max_scale = choose_scale(
            scores_max,
            scale_mode=NORMALIZATION_SCALE_MODE,
            percentile=NORMALIZATION_SCALE_PERCENTILE,
            manual_scale=MANUAL_SCORE_MAX_SCALE,
        )
        score_ref_max = safe_positive_scale(np.nanmax(finite_1d(scores)), scores, "score_ref_max")
        score_max_ref_max = safe_positive_scale(np.nanmax(finite_1d(scores_max)), scores_max, "score_max_ref_max")
    else:
        score_scale = global_score_scale
        score_max_scale = global_score_max_scale
        score_ref_max = global_score_ref_max
        score_max_ref_max = global_score_max_ref_max

    norm_scores = transform_scores(
        scores,
        mode=NORMALIZATION_MODE,
        scale=score_scale,
        reference_max=score_ref_max,
        rescale_to_unit_max=RESCALE_TRANSFORM_TO_UNIT_MAX,
    )
    norm_scores_max = transform_scores(
        scores_max,
        mode=NORMALIZATION_MODE,
        scale=score_max_scale,
        reference_max=score_max_ref_max,
        rescale_to_unit_max=RESCALE_TRANSFORM_TO_UNIT_MAX,
    )

    return {
        "model_name": model_name,
        "scores": scores,
        "scores_max": scores_max,
        "first_run": first_run,
        "y_true_bad": y_true_bad,
        "eval_mask": eval_mask,
        "label_info": label_info,
        "norm_scores": norm_scores,
        "norm_scores_max": norm_scores_max,
        "score_scale": score_scale,
        "score_max_scale": score_max_scale,
        "score_ref_max": score_ref_max,
        "score_max_ref_max": score_max_ref_max,
    }


def append_metrics_for_model_threshold(
    *,
    prepared: dict,
    threshold: float,
    args: argparse.Namespace,
    summary_rows: list[dict],
    run_rows: list[dict],
) -> None:
    """Append summary rows and per-run rows for one model at one threshold."""
    model_name = prepared["model_name"]
    scores = prepared["scores"]
    first_run = prepared["first_run"]
    y_true_bad = prepared["y_true_bad"]
    eval_mask = prepared["eval_mask"]
    label_info = prepared["label_info"]
    norm_scores = prepared["norm_scores"]
    norm_scores_max = prepared["norm_scores_max"]

    evals = build_evaluations_for_threshold(
        norm_scores=norm_scores,
        norm_scores_max=norm_scores_max,
        threshold=threshold,
        args=args,
    )

    for method, pred_bad in evals.items():
        metrics = confusion_and_metrics(y_true_bad[eval_mask], pred_bad[eval_mask])
        row = {
            "model_name": model_name,
            "batch_size": prepared["batch_size"],
            "learning_rate": prepared["learning_rate"],
            "method": method,
            "threshold": threshold,
            "normalization_mode": NORMALIZATION_MODE,
            "normalization_scope": NORMALIZATION_SCOPE,
            "normalization_scale_mode": NORMALIZATION_SCALE_MODE,
            "unknown_run_policy": UNKNOWN_RUN_POLICY,
            "threshold_units": threshold_unit_label(),
            "score_scale": prepared["score_scale"],
            "score_max_scale": prepared["score_max_scale"],
            "score_reference_max": prepared["score_ref_max"],
            "score_max_reference_max": prepared["score_max_ref_max"],
            "n_windows_total": int(scores.size),
            "n_windows_evaluated": int(np.sum(eval_mask)),
            "n_explicit_good_windows": int(np.sum(label_info["is_explicit_good"] & eval_mask)),
            "n_explicit_bad_windows": int(np.sum(label_info["is_explicit_bad"] & eval_mask)),
            "n_unknown_windows_total": int(np.sum(label_info["is_unknown"])),
            "n_default_good_windows": int(np.sum(label_info["is_default_good"] & eval_mask)),
            "n_default_bad_windows": int(np.sum(label_info["is_default_bad"] & eval_mask)),
            "n_true_good_windows": int(np.sum((~y_true_bad) & eval_mask)),
            "n_true_bad_windows": int(np.sum(y_true_bad & eval_mask)),
            **metrics,
        }
        summary_rows.append(row)

    # Per-run ratios for debugging and sanity checks.
    for run in sorted(np.unique(first_run[eval_mask])):
        run_mask = eval_mask & (first_run == run)
        if run in bad_runs:
            true_label = "bad"
        elif run in good_runs:
            true_label = "good"
        elif UNKNOWN_RUN_POLICY.lower() == "good":
            true_label = "default_good"
        elif UNKNOWN_RUN_POLICY.lower() == "bad":
            true_label = "default_bad"
        else:
            true_label = "unknown"
        if not np.any(run_mask):
            continue

        for method, pred_bad in evals.items():
            n = int(np.sum(run_mask))
            n_pred_bad = int(np.sum(pred_bad[run_mask]))
            n_pred_good = n - n_pred_bad
            run_rows.append({
                "model_name": model_name,
                "threshold": threshold,
                "run": int(run),
                "true_label": true_label,
                "method": method,
                "n_windows": n,
                "n_pred_bad": n_pred_bad,
                "n_pred_good": n_pred_good,
                "pred_bad_ratio": n_pred_bad / n if n > 0 else math.nan,
                "pred_good_ratio": n_pred_good / n if n > 0 else math.nan,
            })


def main() -> int:
    args = parse_args()
    thresholds = get_thresholds()
    model_start_index = resolve_model_start_index(args)

    npz_files = find_npz_files(
        CHECKPOINTS_GNN_DIR,
        NPZ_NAME,
        model_start_index=model_start_index,
    )

    if not npz_files:
        compact_summary_csv = OUTPUT_DIR / "threshold_metrics_summary.csv"
        write_empty_compact_summary_csv(compact_summary_csv)

        print("=" * 80)
        print("WARNING: no model directories matched the requested model-start filter.")
        print(f"CHECKPOINTS_GNN_DIR       = {CHECKPOINTS_GNN_DIR}")
        print(f"NPZ_NAME                  = {NPZ_NAME}")
        print(f"MODEL_START_INDEX         = {MODEL_START_INDEX}")
        print(f"--start                   = {args.start}")
        print(f"RESOLVED_START_INDEX      = {model_start_index}")
        print()
        print(f"Wrote empty summary CSV: {compact_summary_csv}")
        print("No metrics were evaluated.")
        print("No plots were produced.")
        print("=" * 80)

        return 0

    print("=" * 80)
    print(f"Found {len(npz_files)} NPZ files under {CHECKPOINTS_GNN_DIR}")
    print(f"MODEL_START_INDEX         = {MODEL_START_INDEX}")
    print(f"--start                   = {args.start}")
    print(f"RESOLVED_START_INDEX      = {model_start_index}")
    print(f"NORMALIZATION_MODE        = {NORMALIZATION_MODE}")
    print(f"NORMALIZATION_SCOPE       = {NORMALIZATION_SCOPE}")
    print(f"NORMALIZATION_SCALE_MODE  = {NORMALIZATION_SCALE_MODE}")
    print(f"UNKNOWN_RUN_POLICY        = {UNKNOWN_RUN_POLICY}")
    print(f"THRESHOLDS                = {thresholds}")
    print(f"THRESHOLD_UNITS           = {threshold_unit_label()}")
    if args.scores_only:
        method_selection = "scores_only"
    elif args.scores_max_only:
        method_selection = "scores_max_only"
    else:
        method_selection = "all: scores_only, scores_max_only, both"

    rank_metric = selected_rank_metric(args)
    if rank_metric is not None:
        output_order = f"ranked by {rank_metric}"
    elif args.threshold_first:
        output_order = "threshold-first: all models per threshold"
    else:
        output_order = "model-first: all thresholds per model"

    print(f"BOTH_RULE                 = {BOTH_RULE}")
    print(f"METHOD_SELECTION          = {method_selection}")
    print(f"RANK_METRIC               = {rank_metric if rank_metric is not None else 'none'}")
    print(f"OUTPUT_ORDER              = {output_order}")
    print("=" * 80)

    loaded = {}
    for npz_path in npz_files:
        model_name = npz_path.parent.name
        try:
            loaded[model_name] = load_required_arrays(npz_path)
        except Exception as exc:
            print(f"WARNING: failed to load {npz_path}: {exc}")

    if not loaded:
        print("No valid NPZ files loaded.")
        return 1

    # Global scales are computed once if requested.
    if NORMALIZATION_SCOPE == "global":
        all_scores = np.concatenate([d["scores"] for d in loaded.values()])
        all_scores_max = np.concatenate([d["scores_max"] for d in loaded.values()])
        global_score_scale = choose_scale(
            all_scores,
            scale_mode=NORMALIZATION_SCALE_MODE,
            percentile=NORMALIZATION_SCALE_PERCENTILE,
            manual_scale=MANUAL_SCORE_SCALE,
        )
        global_score_max_scale = choose_scale(
            all_scores_max,
            scale_mode=NORMALIZATION_SCALE_MODE,
            percentile=NORMALIZATION_SCALE_PERCENTILE,
            manual_scale=MANUAL_SCORE_MAX_SCALE,
        )
        global_score_ref_max = safe_positive_scale(np.nanmax(finite_1d(all_scores)), all_scores, "global_score_ref_max")
        global_score_max_ref_max = safe_positive_scale(np.nanmax(finite_1d(all_scores_max)), all_scores_max, "global_score_max_ref_max")
    elif NORMALIZATION_SCOPE == "per_model":
        global_score_scale = global_score_max_scale = None
        global_score_ref_max = global_score_max_ref_max = None
    else:
        raise ValueError("NORMALIZATION_SCOPE must be 'per_model' or 'global'")

    prepared_by_model: dict[str, dict] = {}
    for model_name, arrays in loaded.items():
        prepared = prepare_model_arrays(
            model_name=model_name,
            arrays=arrays,
            global_score_scale=global_score_scale,
            global_score_max_scale=global_score_max_scale,
            global_score_ref_max=global_score_ref_max,
            global_score_max_ref_max=global_score_max_ref_max,
        )
        if prepared is not None:
            model_dir = CHECKPOINTS_GNN_DIR / model_name
            training_params = load_training_params_from_model_dir(model_dir)
            prepared["batch_size"] = training_params["batch_size"]
            prepared["learning_rate"] = training_params["learning_rate"]

            prepared_by_model[model_name] = prepared

    if not prepared_by_model:
        print("No models with known-label windows were prepared.")
        return 1

    summary_rows = []
    run_rows = []

    model_names = list(prepared_by_model.keys())

    if args.threshold_first:
        for threshold in thresholds:
            for model_name in model_names:
                append_metrics_for_model_threshold(
                    prepared=prepared_by_model[model_name],
                    threshold=threshold,
                    args=args,
                    summary_rows=summary_rows,
                    run_rows=run_rows,
                )
    else:
        for model_name in model_names:
            for threshold in thresholds:
                append_metrics_for_model_threshold(
                    prepared=prepared_by_model[model_name],
                    threshold=threshold,
                    args=args,
                    summary_rows=summary_rows,
                    run_rows=run_rows,
                )
            print(
                f"Processed {model_name}: evaluated {int(np.sum(prepared_by_model[model_name]['eval_mask']))} "
                f"known-label windows at {len(thresholds)} threshold(s)"
            )

    if args.threshold_first:
        for model_name in model_names:
            print(
                f"Processed {model_name}: evaluated {int(np.sum(prepared_by_model[model_name]['eval_mask']))} "
                f"known-label windows at {len(thresholds)} threshold(s)"
            )

    if not summary_rows:
        print("No summary rows produced.")
        return 1

    # Compact summary: this is always written.
    compact_summary_fields = [
        "model_name",
        "batch_size",
        "learning_rate",
        "method",
        "threshold",
        "normalization_mode",
        "threshold_units",
        "TP", "TN", "FP", "FN",
        "precision", "recall", "F1", "accuracy",
    ]

    # Full summary: only written when --full-summary is passed.
    full_summary_fields = [
        "model_name", "batch_size", "learning_rate", "method", "threshold",
        "normalization_mode", "normalization_scope", "normalization_scale_mode",
        "unknown_run_policy", "threshold_units",
        "score_scale", "score_max_scale", "score_reference_max", "score_max_reference_max",
        "n_windows_total", "n_windows_evaluated",
        "n_explicit_good_windows", "n_explicit_bad_windows",
        "n_unknown_windows_total", "n_default_good_windows", "n_default_bad_windows",
        "n_true_good_windows", "n_true_bad_windows",
        "TP", "TN", "FP", "FN",
        "precision", "recall", "F1", "accuracy",
        "true_bad_pred_bad_ratio", "true_bad_pred_good_ratio",
        "true_good_pred_bad_ratio", "true_good_pred_good_ratio",
    ]
    run_fields = [
        "model_name", "threshold", "run", "true_label", "method", "n_windows",
        "n_pred_bad", "n_pred_good", "pred_bad_ratio", "pred_good_ratio",
    ]

    compact_rows = []
    for row in summary_rows:
        compact_rows.append({
            "model_name": row["model_name"],
            "batch_size": row["batch_size"],
            "learning_rate": row["learning_rate"],
            "method": row["method"],
            "threshold": row["threshold"],
            # Backward-compatible alias expected by graphing/plot_sweep_metrics.py.
            "normalization_mode": row["normalization_mode"],
            "threshold_units": row["threshold_units"],
            "TP": row["TP"],
            "TN": row["TN"],
            "FP": row["FP"],
            "FN": row["FN"],
            "precision": row["precision"],
            "recall": row["recall"],
            "F1": row["F1"],
            "accuracy": row["accuracy"],
        })

    # If requested, rank the model/method/threshold rows by the selected metric.
    # This affects the compact CSV and the full summary CSV. The per-run CSV is
    # left in the selected model/threshold order because it is mainly for debugging.
    compact_rows = sort_rows_by_metric(compact_rows, rank_metric)
    summary_rows_for_output = sort_rows_by_metric(summary_rows, rank_metric)

    compact_summary_csv = OUTPUT_DIR / "threshold_metrics_summary.csv"
    write_csv(compact_summary_csv, compact_rows, compact_summary_fields)

    full_summary_csv = OUTPUT_DIR / "threshold_metrics_full_summary.csv"
    run_csv = OUTPUT_DIR / "threshold_per_run_ratios.csv"

    if args.full_summary:
        write_csv(full_summary_csv, summary_rows_for_output, full_summary_fields)
        write_csv(run_csv, run_rows, run_fields)

    print("\n" + "=" * 80)
    print(f"Saved compact summary CSV: {compact_summary_csv}")
    if rank_metric is not None:
        print(f"CSV model/method/threshold rows ranked by {rank_metric} from high to low")
    elif args.threshold_first:
        print("CSV rows written in threshold-first order")
    else:
        print("CSV rows written in model-first order")
    if args.full_summary:
        print(f"Saved full summary CSV: {full_summary_csv}")
        print(f"Saved per-run CSV: {run_csv}")
    else:
        print("Skipped full summary/per-run CSVs. Use --full-summary to write them.")
    print("=" * 80)

    for method in sorted(set(row["method"] for row in summary_rows)):
        method_rows = [row for row in summary_rows if row["method"] == method]
        method_rows = [row for row in method_rows if np.isfinite(row["F1"])]
        method_rows.sort(key=lambda r: r["F1"], reverse=True)
        print(f"\nTop model/threshold combinations by F1 for {method}:")
        for row in method_rows[:10]:
            print(
                f"  {row['model_name']:30s} "
                f"threshold={row['threshold']:.4g} "
                f"F1={row['F1']:.4f} "
                f"precision={row['precision']:.4f} "
                f"recall={row['recall']:.4f} "
                f"TP={row['TP']} TN={row['TN']} FP={row['FP']} FN={row['FN']}"
            )

    if args.with_plot:
        run_plotting_script(args, model_start_index=model_start_index)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
