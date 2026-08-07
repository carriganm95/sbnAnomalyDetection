#!/usr/bin/env python3
"""Collect graph VAE good-vs-bad evaluation metrics from saved text logs.

Expected layout:

    checkpoints/graph_vae/MODEL_NAME/inference_result/goodvsbad_eval.txt

Each ``goodvsbad_eval.txt`` is expected to be the captured output from a command
like:

    python -m sbn_anomaly.infer.window_score \
        --scores checkpoints/graph_vae/MODEL_NAME/inference_result/scores_good.npz \
        --compare checkpoints/graph_vae/MODEL_NAME/inference_result/scores_bad.npz \
        --labels good bad \
        --aggregator group_max_mean \
        --channel-map configs/SBNDTPCChannelMap_v2_with_positions.csv \
        --plot checkpoints/graph_vae/MODEL_NAME/inference_result/goodvsbad.png \
        --percentile 90.0

For every model directory, this script:
  1. looks for ``inference_result/goodvsbad_eval.txt``
  2. extracts AUC, threshold, precision, recall/TPR, F1, FPR, and confusion matrix
  3. also extracts useful context such as good/bad window counts and aggregator
  4. writes a compact CSV summary
  5. optionally writes a full detailed CSV summary with --full-summary
  6. optionally writes a SQLite table into ``graph_vae_sweep.sqlite3`` with --with-sqlite
  7. optionally ranks rows by precision, recall, F1, AUC, accuracy, or FPR

Available command-line flags:

    --start N
        Only scan model directories whose leading numeric prefix is >= N.
        Example: --start 81 scans 0081_..., 0082_..., etc.

    --eval-name NAME
        Name of the evaluation text file inside inference_result/.
        Default: goodvsbad_eval.txt

    --full-summary
        Also write the full detailed summary CSV.
        By default, only the compact summary CSV is written.

    --with-sqlite
        Write parsed metrics to graph_vae_sweep.sqlite3.
        By default, SQLite output is skipped.

    --db-path PATH
        SQLite database path. Default: graph_vae_sweep.sqlite3

    --precision-rank
    --recall-rank
    --f1-rank
    --auc-rank
    --accuracy-rank
        Sort output rows from high to low by the chosen metric.

    --fpr-rank
        Sort output rows from low to high FPR.

Example usage:

    python evaluate_graph_vae_sweep.py
    python evaluate_graph_vae_sweep.py --start 20
    python evaluate_graph_vae_sweep.py --f1-rank
    python evaluate_graph_vae_sweep.py --auc-rank
    python evaluate_graph_vae_sweep.py --full-summary
    python evaluate_graph_vae_sweep.py --with-sqlite
    python evaluate_graph_vae_sweep.py --full-summary --with-sqlite

Default behavior:

    python evaluate_graph_vae_sweep.py

writes only:

    threshold_evaluation/graph_vae/threshold_metrics_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


# ============================================================
# User settings
# ============================================================

CHECKPOINTS_GRAPH_VAE_DIR = Path("checkpoints/graph_vae")
INFERENCE_RESULT_DIR_NAME = "inference_result"
EVAL_TXT_NAME = "goodvsbad_eval.txt"
SWEEP_DB_PATH = Path("graph_vae_sweep.sqlite3")
SQLITE_TABLE_NAME = "goodvsbad_eval_metrics"

# Only evaluate model directories whose numeric prefix is >= this value.
# Example:
#   MODEL_START_INDEX = None  -> use all model directories
#   MODEL_START_INDEX = 81    -> use 0081_..., 0082_..., ..., 0160_...
MODEL_START_INDEX: int | None = None

OUTPUT_DIR = Path("threshold_evaluation/graph_vae")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMPACT_SUMMARY_CSV = OUTPUT_DIR / "threshold_metrics_summary.csv"
FULL_SUMMARY_CSV = OUTPUT_DIR / "goodvsbad_eval_full_summary.csv"


# ============================================================
# Generic helpers
# ============================================================

def get_model_index(model_dir_name: str) -> int | None:
    """Extract leading numeric model index from a model directory name."""
    prefix = model_dir_name.split("_", 1)[0]
    if not prefix.isdigit():
        return None
    return int(prefix)


def resolve_model_start_index(args: argparse.Namespace) -> int | None:
    """Resolve model-start filter from CLI first, then script setting."""
    if args.start is not None:
        if args.start < 0:
            raise ValueError(f"--start must be >= 0; got {args.start}")
        return args.start

    if MODEL_START_INDEX is not None and MODEL_START_INDEX < 0:
        raise ValueError(f"MODEL_START_INDEX must be >= 0; got {MODEL_START_INDEX}")

    return MODEL_START_INDEX


def write_csv(path: Path, rows: list[dict], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_float(value: str | None) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def safe_int(value: str | None) -> int | str:
    if value is None:
        return ""
    try:
        return int(value)
    except (TypeError, ValueError):
        return ""


def finite_or_blank(value: object) -> object:
    """Use blanks instead of NaN in CSV/SQLite output."""
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


# ============================================================
# File discovery
# ============================================================

def find_eval_txt_files(
    checkpoints_dir: Path,
    *,
    inference_result_dir_name: str,
    eval_txt_name: str,
    model_start_index: int | None = None,
) -> list[Path]:
    """Find checkpoints/graph_vae/MODEL/inference_result/goodvsbad_eval.txt files."""
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {checkpoints_dir}")

    eval_files: list[Path] = []

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

        eval_path = child / inference_result_dir_name / eval_txt_name
        if eval_path.exists():
            eval_files.append(eval_path)

    return eval_files


# ============================================================
# Config/model-name metadata helpers
# ============================================================

def parse_value_from_model_name(model_name: str, key: str) -> str:
    """Parse values like bs32, lr0p001, beta0p5 from the model directory name."""
    match = re.search(rf"(?:^|_){re.escape(key)}([^_]+)", model_name)
    if not match:
        return ""
    return match.group(1).replace("p", ".")


def load_training_params_from_model_dir(model_dir: Path) -> dict[str, object]:
    """Read common graph VAE params from copied YAML config, falling back to model name."""
    candidate_configs = [
        model_dir / "config_run.yaml",
        model_dir / "config_original.yaml",
        model_dir / f"{model_dir.name}.yaml",
    ]

    result: dict[str, object] = {
        "batch_size": "",
        "learning_rate": "",
        "beta": "",
        "window_size": "",
        "stride": "",
        "radius": "",
    }

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
        data = config.get("data", {}) or {}
        model = config.get("model", {}) or {}
        graph = config.get("graph", {}) or {}

        result["batch_size"] = training.get("batch_size", result["batch_size"])
        result["learning_rate"] = training.get(
            "lr",
            training.get("learning_rate", result["learning_rate"]),
        )
        result["beta"] = training.get("beta", model.get("beta", result["beta"]))
        result["window_size"] = data.get(
            "window_size",
            model.get("window_size", result["window_size"]),
        )
        result["stride"] = data.get(
            "stride",
            model.get("stride", result["stride"]),
        )
        result["radius"] = graph.get(
            "radius",
            model.get("radius", result["radius"]),
        )
        break

    # Fallback to common model-name tokens, for example:
    # 0000_win100_stride100_rad4_bs32_lr0p001_beta0p5
    model_name = model_dir.name
    result["batch_size"] = result["batch_size"] or parse_value_from_model_name(model_name, "bs")
    result["learning_rate"] = result["learning_rate"] or parse_value_from_model_name(model_name, "lr")
    result["beta"] = result["beta"] or parse_value_from_model_name(model_name, "beta")
    result["window_size"] = result["window_size"] or parse_value_from_model_name(model_name, "win")
    result["stride"] = result["stride"] or parse_value_from_model_name(model_name, "stride")
    result["radius"] = result["radius"] or parse_value_from_model_name(model_name, "rad")

    return result


# ============================================================
# goodvsbad_eval.txt parser
# ============================================================

def parse_label_stats(text: str, label: str) -> dict[str, object]:
    """Parse a line like '# good: aggregator=... windows=138 mean=...' or '# bad: ...'."""
    prefix = rf"^#\s*{re.escape(label)}:\s*(?P<body>.*)$"
    match = re.search(prefix, text, flags=re.MULTILINE)
    out: dict[str, object] = {}
    if not match:
        return out

    body = match.group("body")
    for key, raw_value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", body):
        value = raw_value.strip().rstrip(",")
        if key in {"windows"}:
            out[f"{label}_{key}"] = safe_int(value)
        elif key in {"mean", "p50", "p90", "p95", "p99", "max", "min", "std"}:
            out[f"{label}_{key}"] = safe_float(value)
        elif key == "aggregator":
            out["aggregator"] = value
        else:
            out[f"{label}_{key}"] = value

    return out


def parse_goodvsbad_eval_txt(eval_path: Path) -> dict[str, object]:
    """Parse metrics from one goodvsbad_eval.txt file."""
    text = eval_path.read_text(errors="replace")

    row: dict[str, object] = {
        "eval_txt_path": str(eval_path),
        "returncode": "",
        "aggregator": "",
        "auc": math.nan,
        "threshold": math.nan,
        "threshold_percentile": math.nan,
        "threshold_source": "",
        "precision": math.nan,
        "recall": math.nan,
        "F1": math.nan,
        "FPR": math.nan,
        "TN": "",
        "FP": "",
        "FN": "",
        "TP": "",
        "accuracy": math.nan,
        "good_windows": "",
        "bad_windows": "",
        "good_mean": math.nan,
        "bad_mean": math.nan,
        "good_p95": math.nan,
        "bad_p95": math.nan,
        "good_p99": math.nan,
        "bad_p99": math.nan,
        "good_max": math.nan,
        "bad_max": math.nan,
        "plot_path": "",
    }

    row.update(parse_label_stats(text, "good"))
    row.update(parse_label_stats(text, "bad"))

    auc_match = re.search(
        r"AUC\(bad\s+vs\s+good\)\s*=\s*([-+0-9.eE]+)\s+"
        r"threshold\s*=\s*([-+0-9.eE]+)\s*\(([^)]*)\)",
        text,
    )
    if auc_match:
        row["auc"] = safe_float(auc_match.group(1))
        row["threshold"] = safe_float(auc_match.group(2))
        threshold_note = auc_match.group(3).strip()
        row["threshold_source"] = threshold_note

        pct_match = re.search(r"p\s*([-+0-9.eE]+)", threshold_note)
        if pct_match:
            row["threshold_percentile"] = safe_float(pct_match.group(1))

    # Confusion matrix lines:
    # #   actual good           124          14
    # #   actual bad              0          79
    actual_good_match = re.search(
        r"actual\s+good\s+([0-9]+)\s+([0-9]+)",
        text,
        flags=re.IGNORECASE,
    )
    actual_bad_match = re.search(
        r"actual\s+bad\s+([0-9]+)\s+([0-9]+)",
        text,
        flags=re.IGNORECASE,
    )

    if actual_good_match:
        row["TN"] = safe_int(actual_good_match.group(1))
        row["FP"] = safe_int(actual_good_match.group(2))

    if actual_bad_match:
        row["FN"] = safe_int(actual_bad_match.group(1))
        row["TP"] = safe_int(actual_bad_match.group(2))

    metric_match = re.search(
        r"precision\s*=\s*([-+0-9.eE]+)\s+"
        r"recall(?:\(TPR\))?\s*=\s*([-+0-9.eE]+)\s+"
        r"F1\s*=\s*([-+0-9.eE]+)\s+"
        r"FPR\s*=\s*([-+0-9.eE]+)",
        text,
    )
    if metric_match:
        row["precision"] = safe_float(metric_match.group(1))
        row["recall"] = safe_float(metric_match.group(2))
        row["F1"] = safe_float(metric_match.group(3))
        row["FPR"] = safe_float(metric_match.group(4))

    plot_match = re.search(r"saved\s+plot\s+to\s+(.+)", text, flags=re.IGNORECASE)
    if plot_match:
        row["plot_path"] = plot_match.group(1).strip().lstrip("#").strip()

    returncode_match = re.search(r"returncode\s*=\s*(-?[0-9]+)", text)
    if returncode_match:
        row["returncode"] = safe_int(returncode_match.group(1))

    # Accuracy is not printed by window_score, but can be reconstructed from confusion matrix.
    try:
        tn = int(row["TN"])
        fp = int(row["FP"])
        fn = int(row["FN"])
        tp = int(row["TP"])
        denom = tp + tn + fp + fn
        if denom > 0:
            row["accuracy"] = (tp + tn) / denom
    except (TypeError, ValueError):
        pass

    return row


# ============================================================
# Ranking and SQLite output
# ============================================================

def selected_rank_metric(args: argparse.Namespace) -> str | None:
    if args.precision_rank:
        return "precision"
    if args.recall_rank:
        return "recall"
    if args.f1_rank:
        return "F1"
    if args.auc_rank:
        return "auc"
    if args.accuracy_rank:
        return "accuracy"
    if args.fpr_rank:
        return "FPR"
    return None


def sort_rows_by_metric(rows: list[dict], metric: str | None) -> list[dict]:
    """Sort rows by metric. FPR ranks low-to-high; other metrics rank high-to-low."""
    if metric is None:
        return rows

    low_is_better = metric == "FPR"

    def sort_key(row: dict) -> tuple[int, float, int, str]:
        value = row.get(metric, math.nan)
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            value_float = math.nan

        invalid_flag = 0 if np.isfinite(value_float) else 1

        if invalid_flag:
            sortable_value = 0.0
        elif low_is_better:
            sortable_value = value_float
        else:
            sortable_value = -value_float

        model_index = row.get("model_index", "")
        try:
            model_index_int = int(model_index)
        except (TypeError, ValueError):
            model_index_int = 10**12

        return (
            invalid_flag,
            sortable_value,
            model_index_int,
            str(row.get("model_name", "")),
        )

    return sorted(rows, key=sort_key)


def write_sqlite(
    db_path: Path,
    table_name: str,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
    """Replace a SQLite table with the parsed evaluation summary."""
    if db_path.parent != Path(""):
        db_path.parent.mkdir(parents=True, exist_ok=True)

    cleaned_rows = [
        {field: finite_or_blank(row.get(field, "")) for field in fieldnames}
        for row in rows
    ]

    with sqlite3.connect(db_path) as conn:
        quoted_table = '"' + table_name.replace('"', '""') + '"'
        conn.execute(f"DROP TABLE IF EXISTS {quoted_table}")

        columns_sql = ", ".join(
            '"' + field.replace('"', '""') + '"' + " TEXT"
            for field in fieldnames
        )
        conn.execute(f"CREATE TABLE {quoted_table} ({columns_sql})")

        placeholders = ", ".join("?" for _ in fieldnames)
        quoted_columns = ", ".join(
            '"' + field.replace('"', '""') + '"'
            for field in fieldnames
        )
        insert_sql = (
            f"INSERT INTO {quoted_table} ({quoted_columns}) "
            f"VALUES ({placeholders})"
        )

        conn.executemany(
            insert_sql,
            [[str(row.get(field, "")) for field in fieldnames] for row in cleaned_rows],
        )
        conn.commit()


# ============================================================
# CLI and main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect graph VAE good-vs-bad metrics from "
            "checkpoints/graph_vae/*/inference_result/goodvsbad_eval.txt."
        )
    )

    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help=(
            "Only scan model directories whose leading numeric prefix is greater "
            "than or equal to this value. Overrides MODEL_START_INDEX."
        ),
    )

    parser.add_argument(
        "--eval-name",
        "--eval_name",
        default=EVAL_TXT_NAME,
        help="Evaluation txt filename inside inference_result/. Default: goodvsbad_eval.txt",
    )

    parser.add_argument(
        "--full-summary",
        "--full_summary",
        action="store_true",
        help=(
            "Also write the full detailed summary CSV. "
            "By default, only the compact summary CSV is written."
        ),
    )

    parser.add_argument(
        "--with-sqlite",
        "--with_sqlite",
        action="store_true",
        help=(
            "Write parsed metrics to SQLite. "
            "By default, SQLite output is skipped."
        ),
    )

    parser.add_argument(
        "--db-path",
        "--db_path",
        default=str(SWEEP_DB_PATH),
        help="SQLite DB path. Default: graph_vae_sweep.sqlite3",
    )

    rank_group = parser.add_mutually_exclusive_group()
    rank_group.add_argument("--precision-rank", "--precision_rank", action="store_true")
    rank_group.add_argument("--recall-rank", "--recall_rank", action="store_true")
    rank_group.add_argument("--f1-rank", "--f1_rank", action="store_true")
    rank_group.add_argument("--auc-rank", "--auc_rank", action="store_true")
    rank_group.add_argument("--accuracy-rank", "--accuracy_rank", action="store_true")
    rank_group.add_argument("--fpr-rank", "--fpr_rank", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_start_index = resolve_model_start_index(args)
    db_path = Path(args.db_path)

    eval_txt_files = find_eval_txt_files(
        CHECKPOINTS_GRAPH_VAE_DIR,
        inference_result_dir_name=INFERENCE_RESULT_DIR_NAME,
        eval_txt_name=args.eval_name,
        model_start_index=model_start_index,
    )

    compact_fields = [
        "model_index",
        "model_name",
        "batch_size",
        "learning_rate",
        "beta",
        "window_size",
        "stride",
        "radius",
        "aggregator",
        "threshold",
        "threshold_percentile",
        "auc",
        "precision",
        "recall",
        "F1",
        "FPR",
        "accuracy",
        "TP",
        "TN",
        "FP",
        "FN",
        "good_windows",
        "bad_windows",
        "returncode",
    ]

    full_fields = compact_fields + [
        "threshold_source",
        "good_mean",
        "bad_mean",
        "good_p95",
        "bad_p95",
        "good_p99",
        "bad_p99",
        "good_max",
        "bad_max",
        "plot_path",
        "eval_txt_path",
    ]

    if not eval_txt_files:
        write_csv(COMPACT_SUMMARY_CSV, [], compact_fields)

        if args.full_summary:
            write_csv(FULL_SUMMARY_CSV, [], full_fields)

        print("=" * 80)
        print("WARNING: no good-vs-bad eval txt files matched the requested filter.")
        print(f"CHECKPOINTS_GRAPH_VAE_DIR = {CHECKPOINTS_GRAPH_VAE_DIR}")
        print(
            f"Expected pattern           = "
            f"{CHECKPOINTS_GRAPH_VAE_DIR}/*/{INFERENCE_RESULT_DIR_NAME}/{args.eval_name}"
        )
        print(f"SWEEP_DB_PATH             = {db_path}")
        print(f"MODEL_START_INDEX         = {MODEL_START_INDEX}")
        print(f"--start                   = {args.start}")
        print(f"RESOLVED_START_INDEX      = {model_start_index}")
        print()
        print(f"Wrote empty summary CSV: {COMPACT_SUMMARY_CSV}")

        if args.full_summary:
            print(f"Wrote empty full CSV:    {FULL_SUMMARY_CSV}")
        else:
            print("Skipped full summary CSV. Use --full-summary to write it.")

        print("No metrics were evaluated.")
        print("=" * 80)
        return 0

    print("=" * 80)
    print(f"Found {len(eval_txt_files)} eval txt files under {CHECKPOINTS_GRAPH_VAE_DIR}")
    print(f"Expected filename         = {args.eval_name}")
    print(f"SWEEP_DB_PATH             = {db_path}")
    print(f"MODEL_START_INDEX         = {MODEL_START_INDEX}")
    print(f"--start                   = {args.start}")
    print(f"RESOLVED_START_INDEX      = {model_start_index}")
    print(f"WRITE_FULL_SUMMARY        = {args.full_summary}")
    print(f"WRITE_SQLITE              = {args.with_sqlite}")
    print("=" * 80)

    rows: list[dict] = []

    for eval_path in eval_txt_files:
        model_dir = eval_path.parent.parent
        model_name = model_dir.name
        model_index = get_model_index(model_name)
        params = load_training_params_from_model_dir(model_dir)

        try:
            parsed = parse_goodvsbad_eval_txt(eval_path)
        except Exception as exc:
            print(f"WARNING: failed to parse {eval_path}: {exc}")
            continue

        row = {
            "model_index": model_index if model_index is not None else "",
            "model_name": model_name,
            **params,
            **parsed,
        }
        rows.append(row)

        print(
            f"Parsed {model_name}: "
            f"AUC={row.get('auc', '')} "
            f"threshold={row.get('threshold', '')} "
            f"precision={row.get('precision', '')} "
            f"recall={row.get('recall', '')} "
            f"F1={row.get('F1', '')} "
            f"FPR={row.get('FPR', '')}"
        )

    if not rows:
        print("No valid eval txt files parsed.")
        return 1

    rank_metric = selected_rank_metric(args)
    rows = sort_rows_by_metric(rows, rank_metric)

    compact_rows = [
        {field: finite_or_blank(row.get(field, "")) for field in compact_fields}
        for row in rows
    ]

    full_rows = [
        {field: finite_or_blank(row.get(field, "")) for field in full_fields}
        for row in rows
    ]

    write_csv(COMPACT_SUMMARY_CSV, compact_rows, compact_fields)

    if args.full_summary:
        write_csv(FULL_SUMMARY_CSV, full_rows, full_fields)

    if args.with_sqlite:
        write_sqlite(db_path, SQLITE_TABLE_NAME, full_rows, full_fields)

    print("\n" + "=" * 80)
    print(f"Saved compact summary CSV: {COMPACT_SUMMARY_CSV}")

    if args.full_summary:
        print(f"Saved full summary CSV:    {FULL_SUMMARY_CSV}")
    else:
        print("Skipped full summary CSV. Use --full-summary to write it.")

    if args.with_sqlite:
        print(f"Saved SQLite table:        {db_path}::{SQLITE_TABLE_NAME}")
    else:
        print("Skipped SQLite output. Use --with-sqlite to write it.")

    if rank_metric is not None:
        direction = "low to high" if rank_metric == "FPR" else "high to low"
        print(f"Rows ranked by {rank_metric} from {direction}.")
    else:
        print("Rows written in model-directory order.")

    print("=" * 80)

    top_rows = rows[:10]
    print("\nTop parsed model rows:")
    for row in top_rows:
        try:
            auc_value = float(row.get("auc", math.nan))
        except (TypeError, ValueError):
            auc_value = math.nan

        if np.isfinite(auc_value):
            auc_text = f"AUC={auc_value:.4f}"
        else:
            auc_text = "AUC=nan"

        print(
            f"  {str(row['model_name']):45s} "
            f"{auc_text} "
            f"threshold={row.get('threshold', '')} "
            f"precision={row.get('precision', '')} "
            f"recall={row.get('recall', '')} "
            f"F1={row.get('F1', '')} "
            f"FPR={row.get('FPR', '')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())