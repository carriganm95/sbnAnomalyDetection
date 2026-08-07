from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import os
import re
import runpy
import shlex
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ============================================================
# Default Paths
# ============================================================

# Main project directory.
PROJECT_DIR = Path("/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection")

# Default GraphVAE sweep config directory.
DEFAULT_CONFIG_DIR = PROJECT_DIR / "tuning_configs" / "graph_vae_sweep"

# Store trained GraphVAE models under the usual checkpoint area.
DEFAULT_MODEL_ROOT = PROJECT_DIR / "checkpoints" / "graph_vae"

# Store each sweep run/model under checkpoints/graph_vae/<run_name>/
DEFAULT_RUNS_ROOT = DEFAULT_MODEL_ROOT

# Store database/export summaries directly in the main project directory.
DEFAULT_DB_PATH = PROJECT_DIR / "graph_vae_sweep.sqlite3"

# Default separate GraphVAE inference/evaluation inputs.
# --input in sbn-infer overrides inference.input_path in the YAML.
DEFAULT_GOOD_INPUT = PROJECT_DIR / "data" / "good_events_test.npz"
DEFAULT_BAD_INPUT = PROJECT_DIR / "data" / "bad_events_test.npz"
DEFAULT_TRAIN_INPUT = PROJECT_DIR / "data" / "events_train.npz"

# Training-split outputs used to fit the per-channel z-score baseline.
DEFAULT_BASELINE_SCORES_NAME = "scores_train.npz"
DEFAULT_CHANNEL_BASELINE_NAME = "channel_baseline.npz"
DEFAULT_BASELINE_CMD = "python -m sbn_anomaly.infer.channel_baseline"
DEFAULT_BASELINE_EVAL_PLOT_NAME = "goodvsbad_zscore.png"
DEFAULT_BASELINE_EVAL_LOG_NAME = "goodvsbad_zscore_eval.txt"
DEFAULT_BASELINE_EVAL_JSON_NAME = "goodvsbad_zscore_eval.json"
DEFAULT_BASELINE_EVAL_PERCENTILE = 90.0

# Default good-vs-bad evaluation settings.
DEFAULT_CHANNEL_MAP = PROJECT_DIR / "configs" / "SBNDTPCChannelMap_v2_with_positions.csv"
DEFAULT_EVAL_AGGREGATOR = "group_max_mean"
DEFAULT_EVAL_PLOT_NAME = "goodvsbad.png"
DEFAULT_EVAL_LOG_NAME = "goodvsbad_eval.txt"
DEFAULT_EVAL_JSON_NAME = "goodvsbad_eval.json"
DEFAULT_EVAL_PERCENTILE = 95

# Per-channel node-score plotting script. This is run after each successful
# good-vs-bad evaluation, using that model's own inference_result directory.
PER_CHANNEL_PLOT_SCRIPT = (
    PROJECT_DIR / "graphing" / "plot_per_channel_scores.py"
)

# Selected-window per-channel node-score plotting script. Each selected row in
# node_scores is one window; multiple selected windows are averaged per channel.
PER_CHANNEL_PER_WINDOW_PLOT_SCRIPT = (
    PROJECT_DIR / "graphing" / "plot_per_channel_per_window_scores.py"
)
DEFAULT_PER_CHANNEL_WINDOW_DATASETS = "both"
DEFAULT_PER_CHANNEL_PER_WINDOW_PLOT_NAME = (
    "channel_node_scores_selected_windows.png"
)

# Fallback filename used by --missing-plot if the plotting script does not
# expose its output filename/path as a global variable.
DEFAULT_PER_CHANNEL_PLOT_NAME = "per_channel_scores.png"




# ============================================================
# Ignored model directories
# ============================================================

# Any model directory listed here is skipped completely before any per-model
# sweep activity, including training, inference, evaluation, plotting, and
# rewrite/repair modes. Add full model directory paths here.
IGNORED: list[Path] = [
    Path(DEFAULT_MODEL_ROOT/"0001_time2000s_tstride400s_rad4_bs64_lr0p001_beta0p5"),
    Path(DEFAULT_MODEL_ROOT/"0002_time2000s_tstride1000s_rad4_bs64_lr0p001_beta0p5"),
    Path(DEFAULT_MODEL_ROOT/"0003_time20000s_tstride200s_rad4_bs64_lr0p001_beta0p5"),
    Path(DEFAULT_MODEL_ROOT/"0004_time20000s_tstride400s_rad4_bs64_lr0p001_beta0p5"),
    Path(DEFAULT_MODEL_ROOT/"0005_time20000s_tstride1000s_rad4_bs64_lr0p001_beta0p5"),
    Path(DEFAULT_MODEL_ROOT/"0006_time40000s_tstride200s_rad4_bs64_lr0p001_beta0p5"),
    Path(DEFAULT_MODEL_ROOT/"0007_time40000s_tstride400s_rad4_bs64_lr0p001_beta0p5"),
    Path(DEFAULT_MODEL_ROOT/"0008_time40000s_tstride1000s_rad4_bs64_lr0p001_beta0p5"),
    Path(DEFAULT_MODEL_ROOT/"0009_time80000s_tstride200s_rad4_bs64_lr0p001_beta0p5"),
    Path(DEFAULT_MODEL_ROOT/"0010_time80000s_tstride400s_rad4_bs64_lr0p001_beta0p5"),
    Path(DEFAULT_MODEL_ROOT/"0011_time80000s_tstride1000s_rad4_bs64_lr0p001_beta0p5"),
    Path(DEFAULT_MODEL_ROOT/"All_data_timed"),
]


# ============================================================
# Good / bad classification
# ============================================================

GOOD_RUNS = {18445, 19724, 20141, 20142, 20144}
BAD_RUNS = {19627, 19946, 20104}


# ============================================================
# Small helpers
# ============================================================

def now_str() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name.strip("_")




def is_ignored_run_dir(run_dir: Path) -> bool:
    """Return True if run_dir is listed in IGNORED."""
    resolved_run_dir = run_dir.expanduser().resolve()
    return any(
        resolved_run_dir == Path(path).expanduser().resolve()
        for path in IGNORED
    )


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as fh:
        cfg = yaml.safe_load(fh)
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML config did not load as a dict: {path}")
    return cfg


def save_yaml(path: Path, cfg: dict[str, Any]) -> None:
    with path.open("w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)


def flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            flat.update(flatten_dict(v, key))
        else:
            flat[key] = v
    return flat


def jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def value_to_str(v: Any) -> str:
    try:
        return json.dumps(v, default=jsonable)
    except TypeError:
        return str(v)


# ============================================================
# Database
# ============================================================

SQLITE_TIMEOUT_SEC = 120.0
SQLITE_BUSY_TIMEOUT_MS = 120_000
SQLITE_WRITE_RETRIES = 12
SQLITE_RETRY_BASE_DELAY_SEC = 1.0


def is_database_locked_error(exc: BaseException) -> bool:
    """Return True for SQLite busy/locked errors that are safe to retry."""
    message = str(exc).lower()
    return "database is locked" in message or "database is busy" in message


def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        str(db_path),
        timeout=120.0,
        isolation_level=None,
    )

    # Wait for temporary write locks instead of failing immediately.
    conn.execute("PRAGMA busy_timeout = 120000")

    # Do not change journal_mode here. Changing it requires an exclusive lock,
    # which can fail when the database is on shared storage.
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_name TEXT UNIQUE,
            config_name TEXT,
            original_config_path TEXT,
            run_config_path TEXT,
            run_dir TEXT,
            checkpoint_dir TEXT,
            inference_dir TEXT,
            final_model_path TEXT,
            good_score_npz_path TEXT,
            bad_score_npz_path TEXT,
            status TEXT,
            start_time TEXT,
            end_time TEXT,
            duration_sec REAL,
            train_cmd TEXT,
            infer_good_cmd TEXT,
            infer_bad_cmd TEXT,
            eval_cmd TEXT,
            train_returncode INTEGER,
            infer_good_returncode INTEGER,
            infer_bad_returncode INTEGER,
            eval_returncode INTEGER,
            eval_plot_path TEXT,
            eval_log_path TEXT,
            eval_json_path TEXT,
            error TEXT,
            config_json TEXT,
            metrics_json TEXT
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS params (
            experiment_id INTEGER,
            key TEXT,
            value TEXT,
            PRIMARY KEY (experiment_id, key),
            FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metrics (
            experiment_id INTEGER,
            key TEXT,
            value REAL,
            value_text TEXT,
            PRIMARY KEY (experiment_id, key),
            FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        )
        """
    )

    existing_cols = {
        row[1]
        for row in conn.execute(
            "PRAGMA table_info(experiments)"
        ).fetchall()
    }

    needed_cols = {
        "eval_cmd": "TEXT",
        "eval_returncode": "INTEGER",
        "eval_plot_path": "TEXT",
        "eval_log_path": "TEXT",
        "eval_json_path": "TEXT",
    }

    for col, col_type in needed_cols.items():
        if col not in existing_cols:
            conn.execute(
                f"ALTER TABLE experiments ADD COLUMN {col} {col_type}"
            )

    return conn


def get_existing_experiment(
    conn: sqlite3.Connection,
    run_name: str,
) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT
            id,
            run_name,
            config_name,
            status,
            run_dir,
            checkpoint_dir,
            inference_dir,
            final_model_path,
            good_score_npz_path,
            bad_score_npz_path,
            start_time,
            end_time
        FROM experiments
        WHERE run_name = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (run_name,),
    ).fetchone()

    if row is None:
        return None

    keys = [
        "id",
        "run_name",
        "config_name",
        "status",
        "run_dir",
        "checkpoint_dir",
        "inference_dir",
        "final_model_path",
        "good_score_npz_path",
        "bad_score_npz_path",
        "start_time",
        "end_time",
    ]
    return dict(zip(keys, row))


def delete_existing_experiment(
    conn: sqlite3.Connection,
    run_name: str,
    *,
    reason: str,
) -> None:
    """Atomically delete one run's DB rows, retrying transient SQLite locks."""

    # End any transaction accidentally left open by an earlier operation. With
    # isolation_level=None this is normally a no-op, but it also makes this
    # function safe if the connection settings are changed later.
    if conn.in_transaction:
        conn.commit()

    for attempt in range(1, SQLITE_WRITE_RETRIES + 1):
        try:
            # Acquire the write lock before reading the IDs. This prevents the
            # rows from changing between SELECT and DELETE.
            conn.execute("BEGIN IMMEDIATE")

            rows = conn.execute(
                "SELECT id, status FROM experiments WHERE run_name = ?",
                (run_name,),
            ).fetchall()

            if not rows:
                conn.commit()
                return

            ids = [int(row[0]) for row in rows]
            statuses = [str(row[1]) for row in rows]
            placeholders = ",".join("?" for _ in ids)

            if attempt == 1:
                print(
                    f"Existing DB record(s) for run_name={run_name!r} found "
                    f"with status={statuses}; {reason}, overwriting.",
                    flush=True,
                )

            conn.execute(
                f"DELETE FROM params WHERE experiment_id IN ({placeholders})",
                ids,
            )
            conn.execute(
                f"DELETE FROM metrics WHERE experiment_id IN ({placeholders})",
                ids,
            )
            conn.execute(
                f"DELETE FROM experiments WHERE id IN ({placeholders})",
                ids,
            )
            conn.commit()
            return

        except sqlite3.OperationalError as exc:
            if conn.in_transaction:
                conn.rollback()

            if not is_database_locked_error(exc):
                raise

            if attempt >= SQLITE_WRITE_RETRIES:
                raise RuntimeError(
                    f"SQLite database remained locked after "
                    f"{SQLITE_WRITE_RETRIES} attempts while deleting "
                    f"run_name={run_name!r}."
                ) from exc

            delay = min(
                SQLITE_RETRY_BASE_DELAY_SEC * attempt,
                10.0,
            )
            print(
                f"SQLite is locked while rewriting {run_name!r}; "
                f"retrying in {delay:.1f} s "
                f"({attempt}/{SQLITE_WRITE_RETRIES})...",
                flush=True,
            )
            time.sleep(delay)

        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise


def run_directory_needs_rewrite(run_dir: Path) -> tuple[bool, str]:
    """Return whether a run directory is missing or only contains one YAML file."""
    if not run_dir.exists():
        return True, f"run directory is missing: {run_dir}"

    if not run_dir.is_dir():
        return True, f"expected run directory path is not a directory: {run_dir}"

    entries = [p for p in run_dir.iterdir() if not p.name.startswith(".")]

    if len(entries) == 1 and entries[0].is_file() and entries[0].suffix.lower() in {".yaml", ".yml"}:
        return True, f"run directory only contains one YAML file: {entries[0].name}"

    return False, f"run directory looks non-empty enough: {run_dir}"


def evaluation_outputs_missing(
    *,
    eval_plot_path: Path,
    eval_log_path: Path,
    eval_json_path: Path,
) -> list[Path]:
    """Return evaluation output files that are missing.

    These are the files produced by this sweep script's evaluation stage:
    the window_score plot, the captured evaluation log, and the JSON summary.
    The good/bad score NPZ files are inference outputs, so they are checked
    separately only when evaluation actually needs to run.
    """
    expected_paths = [eval_plot_path, eval_log_path, eval_json_path]
    return [path for path in expected_paths if not path.exists()]


def insert_experiment(
    conn: sqlite3.Connection,
    *,
    run_name: str,
    config_name: str,
    original_config_path: Path,
    run_config_path: Path,
    run_dir: Path,
    checkpoint_dir: Path,
    inference_dir: Path,
    final_model_path: Path,
    good_score_npz_path: Path,
    bad_score_npz_path: Path,
    train_cmd: list[str],
    infer_good_cmd: list[str],
    infer_bad_cmd: list[str],
    eval_cmd: list[str],
    eval_plot_path: Path,
    eval_log_path: Path,
    eval_json_path: Path,
    patched_cfg: dict[str, Any],
) -> int:
    cur = conn.execute(
        """
        INSERT INTO experiments (
            run_name,
            config_name,
            original_config_path,
            run_config_path,
            run_dir,
            checkpoint_dir,
            inference_dir,
            final_model_path,
            good_score_npz_path,
            bad_score_npz_path,
            status,
            start_time,
            train_cmd,
            infer_good_cmd,
            infer_bad_cmd,
            eval_cmd,
            eval_plot_path,
            eval_log_path,
            eval_json_path,
            config_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_name,
            config_name,
            str(original_config_path),
            str(run_config_path),
            str(run_dir),
            str(checkpoint_dir),
            str(inference_dir),
            str(final_model_path),
            str(good_score_npz_path),
            str(bad_score_npz_path),
            "running",
            now_str(),
            " ".join(shlex.quote(x) for x in train_cmd),
            " ".join(shlex.quote(x) for x in infer_good_cmd),
            " ".join(shlex.quote(x) for x in infer_bad_cmd),
            " ".join(shlex.quote(x) for x in eval_cmd),
            str(eval_plot_path),
            str(eval_log_path),
            str(eval_json_path),
            json.dumps(patched_cfg, default=jsonable),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def insert_params(
    conn: sqlite3.Connection,
    experiment_id: int,
    cfg: dict[str, Any],
) -> None:
    flat = flatten_dict(cfg)
    rows = [(experiment_id, k, value_to_str(v)) for k, v in sorted(flat.items())]
    conn.executemany(
        """
        INSERT OR REPLACE INTO params (experiment_id, key, value)
        VALUES (?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def insert_metrics(
    conn: sqlite3.Connection,
    experiment_id: int,
    metrics: dict[str, Any],
) -> None:
    rows = []
    for k, v in sorted(metrics.items()):
        if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(float(v)):
            rows.append((experiment_id, k, float(v), None))
        else:
            rows.append((experiment_id, k, None, value_to_str(v)))

    conn.executemany(
        """
        INSERT OR REPLACE INTO metrics (experiment_id, key, value, value_text)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def update_experiment_done(
    conn: sqlite3.Connection,
    experiment_id: int,
    *,
    status: str,
    duration_sec: float,
    train_returncode: int | None,
    infer_good_returncode: int | None,
    infer_bad_returncode: int | None,
    eval_returncode: int | None,
    good_score_npz_path: Path | None,
    bad_score_npz_path: Path | None,
    error: str | None,
    metrics: dict[str, Any],
) -> None:
    conn.execute(
        """
        UPDATE experiments
        SET
            status = ?,
            end_time = ?,
            duration_sec = ?,
            train_returncode = ?,
            infer_good_returncode = ?,
            infer_bad_returncode = ?,
            eval_returncode = ?,
            good_score_npz_path = ?,
            bad_score_npz_path = ?,
            error = ?,
            metrics_json = ?
        WHERE id = ?
        """,
        (
            status,
            now_str(),
            duration_sec,
            train_returncode,
            infer_good_returncode,
            infer_bad_returncode,
            eval_returncode,
            str(good_score_npz_path) if good_score_npz_path else None,
            str(bad_score_npz_path) if bad_score_npz_path else None,
            error,
            json.dumps(metrics, default=jsonable),
            experiment_id,
        ),
    )
    conn.commit()


def restore_completed_experiment_to_db(
    conn: sqlite3.Connection,
    *,
    config_path: Path,
    run_dir: Path,
    channel_map: Path,
    train_base_cmd: list[str],
    infer_base_cmd: list[str],
    eval_base_cmd: list[str],
    args: argparse.Namespace,
) -> bool:
    """Restore a completed run directory that is missing from SQLite.

    This mode never trains, infers, evaluates, or plots. It only reconstructs
    the database row, params, and metrics from files already on disk.
    """
    run_name = safe_name(config_path.stem)

    if get_existing_experiment(conn, run_name) is not None:
        return False

    run_config_path = run_dir / "config_run.yaml"
    if not run_config_path.exists():
        print(f"Cannot restore {run_name}: missing {run_config_path}", flush=True)
        return False

    stored_cfg = load_yaml(run_config_path)
    (
        patched_cfg,
        checkpoint_dir,
        inference_dir,
        final_model_path,
        good_score_npz_path,
        bad_score_npz_path,
    ) = prepare_run_config(stored_cfg, run_dir)

    required_paths = [
        final_model_path,
        good_score_npz_path,
        bad_score_npz_path,
    ]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        print(f"Cannot restore {run_name}: required file(s) are missing:", flush=True)
        for path in missing_paths:
            print(f"  - {path}", flush=True)
        return False

    eval_plot_path = inference_dir / args.eval_plot_name
    eval_log_path = inference_dir / args.eval_log_name
    eval_json_path = inference_dir / args.eval_json_name

    train_cmd = train_base_cmd + ["--config", str(run_config_path)]
    infer_good_cmd = infer_base_cmd + [
        "--config",
        str(run_config_path),
        "--input",
        str(Path(args.good_input)),
        "--output",
        str(good_score_npz_path),
    ]
    infer_bad_cmd = infer_base_cmd + [
        "--config",
        str(run_config_path),
        "--input",
        str(Path(args.bad_input)),
        "--output",
        str(bad_score_npz_path),
    ]
    eval_cmd = eval_base_cmd + [
        "--scores",
        str(good_score_npz_path),
        "--compare",
        str(bad_score_npz_path),
        "--labels",
        args.good_label,
        args.bad_label,
        "--aggregator",
        args.eval_aggregator,
        "--channel-map",
        str(channel_map),
        "--plot",
        str(eval_plot_path),
    ]
    if args.eval_threshold is not None:
        eval_cmd += ["--threshold", str(args.eval_threshold)]
    elif args.eval_percentile is not None:
        eval_cmd += ["--percentile", str(args.eval_percentile)]
    if args.eval_per_run:
        eval_cmd += ["--per-run"]

    print(f"Restoring missing DB row: {run_name}", flush=True)

    experiment_id = insert_experiment(
        conn,
        run_name=run_name,
        config_name=config_path.name,
        original_config_path=config_path,
        run_config_path=run_config_path,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        inference_dir=inference_dir,
        final_model_path=final_model_path,
        good_score_npz_path=good_score_npz_path,
        bad_score_npz_path=bad_score_npz_path,
        train_cmd=train_cmd,
        infer_good_cmd=infer_good_cmd,
        infer_bad_cmd=infer_bad_cmd,
        eval_cmd=eval_cmd,
        eval_plot_path=eval_plot_path,
        eval_log_path=eval_log_path,
        eval_json_path=eval_json_path,
        patched_cfg=patched_cfg,
    )
    insert_params(conn, experiment_id, patched_cfg)

    metrics = collect_run_metrics(
        checkpoint_dir=checkpoint_dir,
        good_score_npz_path=good_score_npz_path,
        bad_score_npz_path=bad_score_npz_path,
    )
    metrics["eval.skipped"] = 0
    metrics["eval.evaluate_only"] = 0
    metrics["eval.missing_evaluate_only"] = 0
    metrics["eval.returncode"] = 0 if eval_json_path.exists() else -1
    metrics["eval.plot_path"] = str(eval_plot_path)
    metrics["eval.log_path"] = str(eval_log_path)
    metrics["eval.json_path"] = str(eval_json_path)
    metrics["eval.plot.exists"] = int(eval_plot_path.exists())
    metrics["eval.log.exists"] = int(eval_log_path.exists())
    metrics["eval.json.exists"] = int(eval_json_path.exists())
    metrics.update(parse_window_score_eval_log(eval_log_path))

    insert_metrics(conn, experiment_id, metrics)
    update_experiment_done(
        conn,
        experiment_id,
        status="success",
        duration_sec=0.0,
        train_returncode=0,
        infer_good_returncode=0,
        infer_bad_returncode=0,
        eval_returncode=0 if eval_json_path.exists() else None,
        good_score_npz_path=good_score_npz_path,
        bad_score_npz_path=bad_score_npz_path,
        error=None,
        metrics=metrics,
    )

    print(f"Restored database row for {run_name}", flush=True)
    return True


def reorder_experiment_ids(
    conn: sqlite3.Connection,
    config_paths: list[Path],
) -> None:
    """Renumber experiment IDs to match sorted YAML sweep order.

    Foreign-key-like experiment_id values in params and metrics are rewritten
    in the same transaction. Database-only rows are preserved after all
    config-backed rows, retaining their prior ID order.
    """
    ordered_run_names = [safe_name(path.stem) for path in config_paths]

    rows = conn.execute(
        "SELECT id, run_name FROM experiments ORDER BY id"
    ).fetchall()
    if not rows:
        return

    current_ids = {str(run_name): int(experiment_id) for experiment_id, run_name in rows}

    ordered_existing_names = [
        run_name
        for run_name in ordered_run_names
        if run_name in current_ids
    ]
    ordered_name_set = set(ordered_existing_names)

    extra_names = [
        str(run_name)
        for experiment_id, run_name in rows
        if str(run_name) not in ordered_name_set
    ]
    final_order = ordered_existing_names + extra_names

    if len(final_order) != len(rows):
        raise RuntimeError(
            "Experiment ID reorder produced an inconsistent row count: "
            f"final_order={len(final_order)}, rows={len(rows)}"
        )

    print("Reordering experiment IDs by sorted YAML config order...", flush=True)

    conn.execute("BEGIN IMMEDIATE")
    try:
        # Move every ID into a collision-free negative range first.
        for old_id, _run_name in rows:
            temporary_id = -int(old_id) - 1
            conn.execute(
                "UPDATE params SET experiment_id = ? WHERE experiment_id = ?",
                (temporary_id, old_id),
            )
            conn.execute(
                "UPDATE metrics SET experiment_id = ? WHERE experiment_id = ?",
                (temporary_id, old_id),
            )
            conn.execute(
                "UPDATE experiments SET id = ? WHERE id = ?",
                (temporary_id, old_id),
            )

        for new_id, run_name in enumerate(final_order, start=1):
            old_id = current_ids[run_name]
            temporary_id = -old_id - 1
            conn.execute(
                "UPDATE experiments SET id = ? WHERE id = ?",
                (new_id, temporary_id),
            )
            conn.execute(
                "UPDATE params SET experiment_id = ? WHERE experiment_id = ?",
                (new_id, temporary_id),
            )
            conn.execute(
                "UPDATE metrics SET experiment_id = ? WHERE experiment_id = ?",
                (new_id, temporary_id),
            )

        conn.execute("DELETE FROM sqlite_sequence WHERE name = 'experiments'")
        conn.execute(
            """
            INSERT INTO sqlite_sequence(name, seq)
            VALUES(
                'experiments',
                (SELECT COALESCE(MAX(id), 0) FROM experiments)
            )
            """
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    print(
        f"Experiment IDs reordered: 1 through {len(final_order)}",
        flush=True,
    )


# ============================================================
# Config patching
# ============================================================

def prepare_run_config(
    original_cfg: dict[str, Any],
    run_dir: Path,
) -> tuple[dict[str, Any], Path, Path, Path, Path, Path]:
    """
    Make a per-run GraphVAE config so YAML jobs do not overwrite each other.

    Returns
    -------
    patched_cfg, checkpoint_dir, inference_dir, final_model_path,
    good_score_npz_path, bad_score_npz_path
    """
    cfg = copy.deepcopy(original_cfg)

    training = cfg.setdefault("training", {})
    inference = cfg.setdefault("inference", {})

    cfg["model_type"] = "graph_vae"

    checkpoint_dir = run_dir
    inference_dir = run_dir / "inference_result"
    final_model_path = checkpoint_dir / "graph_vae_final.pt"
    good_score_npz_path = inference_dir / "scores_good.npz"
    bad_score_npz_path = inference_dir / "scores_bad.npz"

    training["checkpoint_dir"] = str(checkpoint_dir)
    training["output_path"] = str(final_model_path)

    # Inference uses the checkpoint just trained. Input/output are passed on the
    # command line because GraphVAE needs separate good and bad evaluations.
    inference["checkpoint_path"] = str(final_model_path)
    inference["output_path"] = str(good_score_npz_path)

    return (
        cfg,
        checkpoint_dir,
        inference_dir,
        final_model_path,
        good_score_npz_path,
        bad_score_npz_path,
    )


# ============================================================
# Subprocess execution
# ============================================================

def format_bytes(n: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n)
    for unit in units:
        if value < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PB"


def get_process_usage_text(
    proc: subprocess.Popen,
    label: str = "resource",
    *,
    sample_seconds: float = 0.5,
) -> str:
    lines = [f"\n[{now_str()}] {label} usage"]

    try:
        import psutil

        parent = psutil.Process(proc.pid)
        processes = [parent] + parent.children(recursive=True)

        live_processes = []
        for p in processes:
            try:
                p.cpu_percent(interval=None)
                live_processes.append(p)
            except psutil.NoSuchProcess:
                pass

        time.sleep(sample_seconds)

        total_cpu = 0.0
        total_rss = 0
        total_pss = 0
        have_pss = False
        still_live = 0

        for p in live_processes:
            try:
                still_live += 1
                total_cpu += p.cpu_percent(interval=None)
                mem = p.memory_info()
                total_rss += mem.rss

                try:
                    full_mem = p.memory_full_info()
                    pss = getattr(full_mem, "pss", None)
                    if pss is not None:
                        total_pss += pss
                        have_pss = True
                except Exception:
                    pass

            except psutil.NoSuchProcess:
                pass

        if have_pss:
            lines.append(
                f"  Process tree: {still_live} process(es) | "
                f"CPU: {total_cpu:.1f}% over {sample_seconds:.1f}s sample | "
                f"PSS: {format_bytes(total_pss)} | "
                f"RSS: {format_bytes(total_rss)}"
            )
        else:
            lines.append(
                f"  Process tree: {still_live} process(es) | "
                f"CPU: {total_cpu:.1f}% over {sample_seconds:.1f}s sample | "
                f"RSS: {format_bytes(total_rss)} "
                f"(PSS unavailable; RSS may over-count shared memory)"
            )

    except Exception as exc:
        lines.append(f"  Process usage unavailable: {exc}")

    return "\n".join(lines) + "\n"


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    timeout: int | None = None,
    monitor_interval: int = 30,
    batch: bool = False,
    log_path: Path | None = None,
) -> int:
    """Run command, show its output, and periodically inject resource usage.

    If log_path is provided, stdout/stderr from the subprocess plus the
    command header/footer are also written to that file. This is used for
    per-model evaluation logs.
    """
    import selectors

    header = (
        f"$ {' '.join(shlex.quote(x) for x in cmd)}\n"
        f"cwd={cwd}\n"
        + "=" * 80
        + "\n"
    )
    log_fh = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = log_path.open("w", encoding="utf-8")
        log_fh.write(header)
        log_fh.flush()

    print(header, end="", flush=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    if batch:
        env["SBN_BATCH"] = "1"
        env["TQDM_DISABLE"] = "1"
        env["DISABLE_TQDM"] = "1"

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=None,
        bufsize=0,
        env=env,
    )

    start_time = time.perf_counter()
    last_monitor = start_time

    selector = selectors.DefaultSelector()
    assert proc.stdout is not None
    selector.register(proc.stdout, selectors.EVENT_READ)
    stdout_open = True
    returncode: int | None = None

    try:
        while True:
            now = time.perf_counter()

            if timeout is not None and now - start_time > timeout:
                proc.kill()
                proc.wait()
                print(f"\nCommand timed out after {timeout} seconds.", flush=True)
                raise subprocess.TimeoutExpired(cmd, timeout)

            if stdout_open:
                events = selector.select(timeout=0.2)
                for key, _ in events:
                    chunk = os.read(key.fileobj.fileno(), 4096)
                    if chunk:
                        sys.stdout.buffer.write(chunk)
                        sys.stdout.buffer.flush()
                        if log_fh is not None:
                            log_fh.write(chunk.decode("utf-8", errors="replace"))
                            log_fh.flush()
                    else:
                        try:
                            selector.unregister(key.fileobj)
                        except Exception:
                            pass
                        stdout_open = False

            returncode = proc.poll()
            now = time.perf_counter()

            if monitor_interval > 0 and returncode is None and now - last_monitor >= monitor_interval:
                usage_text = get_process_usage_text(proc, "during command")
                print(usage_text, end="", flush=True)
                if log_fh is not None:
                    log_fh.write(usage_text)
                    log_fh.flush()
                last_monitor = time.perf_counter()

            if returncode is not None and not stdout_open:
                break

            if returncode is not None and stdout_open:
                continue

    except KeyboardInterrupt:
        proc.terminate()
        raise

    finally:
        try:
            selector.close()
        except Exception:
            pass

    footer = "\n" + "=" * 80 + "\n" + f"returncode={returncode}\n"
    print(footer, end="", flush=True)
    if log_fh is not None:
        log_fh.write(footer)
        log_fh.close()

    return int(returncode)


# ============================================================
# Per-channel score plotting
# ============================================================

def load_per_channel_plot_namespace(
    *,
    plot_script_path: Path,
    inference_dir: Path,
) -> dict[str, Any]:
    """Load the plotting script without executing its __main__ block."""
    plot_script_path = Path(plot_script_path).expanduser().resolve()
    inference_dir = Path(inference_dir).expanduser().resolve()

    if not plot_script_path.exists():
        raise FileNotFoundError(
            f"Per-channel plotting script does not exist: {plot_script_path}"
        )

    if not plot_script_path.is_file():
        raise FileNotFoundError(
            f"Per-channel plotting script path is not a file: {plot_script_path}"
        )

    namespace = runpy.run_path(
        str(plot_script_path),
        run_name="plot_per_channel_scores",
    )
    namespace["INFERENCE_RESULT_DIR"] = inference_dir
    return namespace


def get_per_channel_plot_path(
    *,
    plot_script_path: Path,
    inference_dir: Path,
) -> Path:
    """Best-effort discovery of the PNG produced by plot_per_channel_scores.py.

    The plotting script is loaded without running main(). Common global output
    variable names are checked first. If none are defined, fall back to
    inference_result/per_channel_scores.png.
    """
    inference_dir = Path(inference_dir).expanduser().resolve()
    namespace = load_per_channel_plot_namespace(
        plot_script_path=plot_script_path,
        inference_dir=inference_dir,
    )

    candidate_names = (
        "OUTPUT_PATH",
        "OUTPUT_PLOT_PATH",
        "PLOT_PATH",
        "OUTPUT_FILE",
        "OUTPUT_FILENAME",
        "PLOT_FILENAME",
    )

    for name in candidate_names:
        value = namespace.get(name)
        if value is None:
            continue

        path = Path(value).expanduser()

        # The plotting script may construct an absolute output path at import
        # time from its own hard-coded INFERENCE_RESULT_DIR. For sweep use, keep
        # only the filename and always check inside the current model's
        # inference_result directory.
        if path.is_absolute():
            path = inference_dir / path.name
        else:
            path = inference_dir / path

        return path.resolve()

    return (inference_dir / DEFAULT_PER_CHANNEL_PLOT_NAME).resolve()


def get_per_channel_per_window_plot_paths(
    *,
    plot_script_path: Path,
    inference_dir: Path,
    output_name: str,
) -> list[Path]:
    """Discover all plane PNGs made by the selected-window plotter."""
    inference_dir = Path(inference_dir).expanduser().resolve()
    namespace = load_per_channel_plot_namespace(
        plot_script_path=plot_script_path,
        inference_dir=inference_dir,
    )

    get_output_paths = namespace.get("get_output_paths")
    if not callable(get_output_paths):
        raise RuntimeError(
            "Selected-window plotting script does not define callable "
            f"get_output_paths(): {plot_script_path}"
        )

    paths = [
        Path(path).expanduser().resolve()
        for path in get_output_paths(inference_dir, output_name)
    ]
    if not paths:
        raise RuntimeError(
            f"Selected-window plotting script returned no output paths: {plot_script_path}"
        )
    return paths


def run_per_channel_score_plot(
    *,
    plot_script_path: Path,
    inference_dir: Path,
) -> None:
    """
    Run graphing/plot_per_channel_scores.py for one model.

    The plotting script currently selects its input directory through the
    global INFERENCE_RESULT_DIR variable rather than a command-line argument.
    To avoid editing the plotting script for every model, load it without
    executing its __main__ block, replace INFERENCE_RESULT_DIR with this
    model's inference_result directory, and then call its main() function.

    Expected input files inside inference_dir:
        - scores_good.npz
        - scores_bad.npz
    """
    plot_script_path = Path(plot_script_path).expanduser().resolve()
    inference_dir = Path(inference_dir).expanduser().resolve()

    good_score_path = inference_dir / "scores_good.npz"
    bad_score_path = inference_dir / "scores_bad.npz"

    missing_inputs = [
        path
        for path in (good_score_path, bad_score_path)
        if not path.exists()
    ]
    if missing_inputs:
        raise FileNotFoundError(
            "Cannot run per-channel score plotting because required score "
            "file(s) are missing: "
            + ", ".join(str(path) for path in missing_inputs)
        )

    print("Per-channel node-score plotting...")
    print(f"  Plot script:      {plot_script_path}")
    print(f"  Inference result: {inference_dir}")

    # Load without executing the script's __main__ block, then override the
    # model-specific inference directory before calling main().
    namespace = load_per_channel_plot_namespace(
        plot_script_path=plot_script_path,
        inference_dir=inference_dir,
    )

    if "main" not in namespace or not callable(namespace["main"]):
        raise RuntimeError(
            f"Plotting script does not define a callable main(): {plot_script_path}"
        )

    old_argv = sys.argv.copy()

    try:
        sys.argv = [
            str(plot_script_path),
            str(inference_dir),
        ]
        namespace["main"]()
    finally:
        sys.argv = old_argv

    print("Per-channel node-score plot finished.")


def run_per_channel_per_window_score_plot(
    *,
    plot_script_path: Path,
    inference_dir: Path,
    window_spec: str | None,
    datasets: str,
    output_name: str,
) -> None:
    """Run the selected-window per-channel plotter for one model."""
    plot_script_path = Path(plot_script_path).expanduser().resolve()
    inference_dir = Path(inference_dir).expanduser().resolve()

    required_score_paths = []
    if datasets in {"good", "both"}:
        required_score_paths.append(inference_dir / "scores_good.npz")
    if datasets in {"bad", "both"}:
        required_score_paths.append(inference_dir / "scores_bad.npz")

    missing_inputs = [path for path in required_score_paths if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(
            "Cannot run selected-window per-channel plotting because required "
            "score file(s) are missing: "
            + ", ".join(str(path) for path in missing_inputs)
        )

    print("Selected-window per-channel node-score plotting...")
    print(f"  Plot script:      {plot_script_path}")
    print(f"  Inference result: {inference_dir}")
    if window_spec is None:
        print("  Window indices:   plotter default")
    else:
        print(f"  Window indices:   {window_spec}")
    print(f"  Dataset(s):       {datasets}")

    namespace = load_per_channel_plot_namespace(
        plot_script_path=plot_script_path,
        inference_dir=inference_dir,
    )
    plot_main = namespace.get("main")
    if not callable(plot_main):
        raise RuntimeError(
            f"Plotting script does not define a callable main(): {plot_script_path}"
        )

    output_paths = plot_main(
        inference_dir,
        window_spec=window_spec,
        datasets=datasets,
        output=output_name,
    )
    print("Selected-window per-channel plots finished:")
    if isinstance(output_paths, (list, tuple)):
        for output_path in output_paths:
            print(f"  - {output_path}")
    else:
        print(f"  - {output_paths}")


# ============================================================
# Result parsing
# ============================================================

def read_last_training_history_row(history_csv: Path) -> dict[str, Any]:
    if not history_csv.exists():
        return {}

    with history_csv.open("r", newline="") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        return {}

    last = rows[-1]
    metrics: dict[str, Any] = {}

    for key, value in last.items():
        if value is None or value == "":
            continue
        try:
            metrics[f"train_history.final_{key}"] = float(value)
        except ValueError:
            metrics[f"train_history.final_{key}"] = value

    for wanted in ["loss", "val_loss", "recon", "kl", "score_p95", "score_p99"]:
        vals = []
        for row in rows:
            if wanted in row and row[wanted] not in ("", None):
                try:
                    vals.append(float(row[wanted]))
                except ValueError:
                    pass
        if vals:
            metrics[f"train_history.best_{wanted}"] = min(vals)

    return metrics


def summarize_array(prefix: str, arr: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return {f"{prefix}.n": 0}

    return {
        f"{prefix}.n": int(arr.size),
        f"{prefix}.mean": float(np.mean(arr)),
        f"{prefix}.std": float(np.std(arr)),
        f"{prefix}.min": float(np.min(arr)),
        f"{prefix}.median": float(np.median(arr)),
        f"{prefix}.p90": float(np.percentile(arr, 90)),
        f"{prefix}.p95": float(np.percentile(arr, 95)),
        f"{prefix}.p99": float(np.percentile(arr, 99)),
        f"{prefix}.max": float(np.max(arr)),
    }


def summarize_graph_vae_scores_npz(prefix: str, score_npz_path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    if not score_npz_path.exists():
        metrics[f"{prefix}.score_npz.exists"] = 0
        metrics[f"{prefix}.score_npz.path"] = str(score_npz_path)
        return metrics

    with np.load(score_npz_path, allow_pickle=True) as data:
        metrics[f"{prefix}.score_npz.exists"] = 1
        metrics[f"{prefix}.score_npz.path"] = str(score_npz_path)

        if "scores" in data:
            metrics.update(summarize_array(f"{prefix}.scores", np.asarray(data["scores"])))

        if "node_scores" in data:
            node_scores = np.asarray(data["node_scores"], dtype=np.float64)
            metrics[f"{prefix}.node_scores.shape"] = "x".join(str(x) for x in node_scores.shape)
            finite_node_scores = node_scores[np.isfinite(node_scores)]
            metrics.update(summarize_array(f"{prefix}.node_scores.finite", finite_node_scores))

        if "channel_mean_error" in data:
            metrics.update(
                summarize_array(
                    f"{prefix}.channel_mean_error",
                    np.asarray(data["channel_mean_error"]),
                )
            )

        if "channel_max_error" in data:
            metrics.update(
                summarize_array(
                    f"{prefix}.channel_max_error",
                    np.asarray(data["channel_max_error"]),
                )
            )

        if "channel_active_frac" in data:
            metrics.update(
                summarize_array(
                    f"{prefix}.channel_active_frac",
                    np.asarray(data["channel_active_frac"]),
                )
            )

        if "provenance" in data:
            provenance = np.asarray(data["provenance"])
            metrics[f"{prefix}.provenance.shape"] = "x".join(str(x) for x in provenance.shape)
            if provenance.ndim == 2 and provenance.shape[1] >= 1:
                runs = provenance[:, 0]
                runs = runs[np.isfinite(runs)].astype(int)
                unique_runs = sorted(np.unique(runs))
                metrics[f"{prefix}.runs.n_unique"] = int(len(unique_runs))
                metrics[f"{prefix}.runs.list"] = ",".join(str(r) for r in unique_runs)

        if "is_anomaly" in data:
            is_anomaly = np.asarray(data["is_anomaly"]).astype(bool)
            metrics[f"{prefix}.is_anomaly.n"] = int(is_anomaly.size)
            metrics[f"{prefix}.is_anomaly.count"] = int(np.count_nonzero(is_anomaly))
            metrics[f"{prefix}.is_anomaly.frac"] = (
                float(np.count_nonzero(is_anomaly) / is_anomaly.size)
                if is_anomaly.size > 0
                else np.nan
            )

    return metrics


def load_scores(score_npz_path: Path) -> np.ndarray:
    with np.load(score_npz_path, allow_pickle=True) as data:
        if "scores" not in data:
            return np.asarray([], dtype=np.float64)
        scores = np.asarray(data["scores"], dtype=np.float64)
    return scores[np.isfinite(scores)]


def mann_whitney_auc(good_scores: np.ndarray, bad_scores: np.ndarray) -> float:
    """Tie-correct separation AUC = P(bad > good) + 0.5 P(tie)."""
    good_scores = np.asarray(good_scores, dtype=np.float64)
    bad_scores = np.asarray(bad_scores, dtype=np.float64)
    good_scores = good_scores[np.isfinite(good_scores)]
    bad_scores = bad_scores[np.isfinite(bad_scores)]

    if good_scores.size == 0 or bad_scores.size == 0:
        return float("nan")

    values = np.concatenate([good_scores, bad_scores])
    labels = np.concatenate([
        np.zeros(good_scores.size, dtype=np.int8),
        np.ones(bad_scores.size, dtype=np.int8),
    ])

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_labels = labels[order]

    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = avg_rank
        start = end

    rank_sum_bad = float(np.sum(ranks[labels == 1]))
    n_good = float(good_scores.size)
    n_bad = float(bad_scores.size)
    u_bad = rank_sum_bad - n_bad * (n_bad + 1.0) / 2.0
    return float(u_bad / (n_good * n_bad))


def parse_window_score_eval_log(eval_log_path: Path) -> dict[str, Any]:
    """Best-effort parser for window_score textual output.

    The exact print format may change, so this stores the full log separately and
    only extracts common scalar fields when their names appear in the output.
    """
    metrics: dict[str, Any] = {}
    if not eval_log_path.exists():
        return metrics

    text = eval_log_path.read_text(errors="replace")
    patterns = {
        "eval.auc": r"(?i)\bAUC\b[^0-9+\-.eE]*(?P<value>[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?)",
        "eval.threshold": r"(?i)\bthreshold\b[^0-9+\-.eE]*(?P<value>[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?)",
        "eval.precision": r"(?i)\bprecision\b[^0-9+\-.eE]*(?P<value>[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?)",
        "eval.recall": r"(?i)\brecall\b[^0-9+\-.eE]*(?P<value>[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?)",
        "eval.f1": r"(?i)\bF1\b[^0-9+\-.eE]*(?P<value>[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?)",
        "eval.fpr": r"(?i)\bFPR\b[^0-9+\-.eE]*(?P<value>[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            try:
                metrics[key] = float(match.group("value"))
            except ValueError:
                pass

    return metrics


def save_eval_summary_json(
    eval_json_path: Path,
    *,
    eval_cmd: list[str],
    eval_returncode: int | None,
    eval_plot_path: Path,
    eval_log_path: Path,
    good_score_npz_path: Path,
    bad_score_npz_path: Path,
    metrics: dict[str, Any],
) -> None:
    eval_json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": now_str(),
        "command": " ".join(shlex.quote(x) for x in eval_cmd),
        "returncode": eval_returncode,
        "plot_path": str(eval_plot_path),
        "log_path": str(eval_log_path),
        "good_score_npz_path": str(good_score_npz_path),
        "bad_score_npz_path": str(bad_score_npz_path),
        "metrics": metrics,
    }
    with eval_json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=jsonable)
        fh.write("\n")


def collect_run_metrics(
    checkpoint_dir: Path,
    good_score_npz_path: Path,
    bad_score_npz_path: Path,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    history_csv = checkpoint_dir / "training_history.csv"
    metrics.update(read_last_training_history_row(history_csv))

    metrics.update(summarize_graph_vae_scores_npz("good", good_score_npz_path))
    metrics.update(summarize_graph_vae_scores_npz("bad", bad_score_npz_path))

    if good_score_npz_path.exists() and bad_score_npz_path.exists():
        good_scores = load_scores(good_score_npz_path)
        bad_scores = load_scores(bad_score_npz_path)

        if good_scores.size > 0 and bad_scores.size > 0:
            metrics["separation.bad_minus_good.mean"] = float(
                np.mean(bad_scores) - np.mean(good_scores)
            )
            metrics["separation.bad_minus_good.median"] = float(
                np.median(bad_scores) - np.median(good_scores)
            )
            metrics["separation.bad_minus_good.p95"] = float(
                np.percentile(bad_scores, 95) - np.percentile(good_scores, 95)
            )
            metrics["separation.bad_minus_good.p99"] = float(
                np.percentile(bad_scores, 99) - np.percentile(good_scores, 99)
            )
            metrics["separation.auc"] = mann_whitney_auc(good_scores, bad_scores)

    return metrics


# ============================================================
# Export
# ============================================================

def export_summary(conn: sqlite3.Connection, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "graph_vae_sweep_summary.csv"
    xlsx_path = output_dir / "graph_vae_sweep_summary.xlsx"

    query = """
    SELECT
        e.id,
        e.run_name,
        e.config_name,
        e.status,
        e.start_time,
        e.end_time,
        e.duration_sec,
        e.run_dir,
        e.final_model_path,
        e.good_score_npz_path,
        e.bad_score_npz_path,
        e.error,
        m.key AS metric_key,
        COALESCE(CAST(m.value AS TEXT), m.value_text) AS metric_value
    FROM experiments e
    LEFT JOIN metrics m ON e.id = m.experiment_id
    ORDER BY e.id, m.key
    """

    rows = conn.execute(query).fetchall()
    cols = [desc[0] for desc in conn.execute(query).description]

    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(cols)
        writer.writerows(rows)

    try:
        import pandas as pd

        experiments = pd.read_sql_query("SELECT * FROM experiments ORDER BY id", conn)
        params = pd.read_sql_query("SELECT * FROM params ORDER BY experiment_id, key", conn)
        metrics = pd.read_sql_query("SELECT * FROM metrics ORDER BY experiment_id, key", conn)

        with pd.ExcelWriter(xlsx_path) as writer:
            experiments.to_excel(writer, sheet_name="experiments", index=False)
            params.to_excel(writer, sheet_name="params", index=False)
            metrics.to_excel(writer, sheet_name="metrics", index=False)

        print(f"Exported Excel summary: {xlsx_path}")

    except Exception as exc:
        print(f"WARNING: could not export Excel summary: {exc}")

    print(f"Exported CSV summary: {csv_path}")


# ============================================================
# Main sweep
# ============================================================

def run_sweep(args: argparse.Namespace) -> int:
    config_dir = Path(args.config_dir)
    runs_root = Path(args.runs_root)
    db_path = Path(args.db_path)
    good_input = Path(args.good_input)
    bad_input = Path(args.bad_input)
    train_input = Path(args.train_input)
    channel_map = Path(args.channel_map)

    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory does not exist: {config_dir}")

    if args.missing_evaluate_only:
        args.evaluate_only = True

    if args.baseline_infer:
        if not 0.0 <= float(args.baseline_eval_percentile) <= 100.0:
            raise ValueError("--baseline-eval-percentile must be between 0 and 100.")
        if args.evaluate_only or args.missing_evaluate_only:
            raise ValueError(
                "--baseline-infer cannot be combined with evaluation-only modes."
            )
        if args.skip_infer:
            raise ValueError("--baseline-infer and --skip-infer cannot be used together.")
        if args.restore_missing_db:
            raise ValueError(
                "--baseline-infer cannot be combined with --restore-missing-db."
            )
        if args.per_channel_plot or args.per_channel_per_window_plot:
            raise ValueError(
                "--baseline-infer cannot create good/bad score plots; remove the "
                "per-channel plotting flags."
            )
    # --missing-plot and --force-replot are standalone plot-only modes.
    # --missing-plot plots only when the expected plot is absent.
    # --force-replot always reruns plotting, even when the plot already exists.
    if args.missing_plot or args.force_replot:
        args.per_channel_plot = True
        args.per_channel_per_window_plot = True

    if args.missing_plot and args.force_replot:
        raise ValueError(
            "--missing-plot and --force-replot cannot be used together."
        )

    if (args.missing_plot or args.force_replot) and (
        args.evaluate_only
        or args.missing_evaluate_only
        or args.infer_only
        or args.missing_infer_only
        or args.baseline_infer
        or args.skip_infer
        or args.force_rewrite
        or args.missing_rewrite
    ):
        raise ValueError(
            "--missing-plot/--force-replot are standalone plot-only modes and cannot "
            "be combined with training/inference/evaluation-only rewrite modes."
        )

    if args.evaluate_only and args.skip_eval:
        raise ValueError(
            "--evaluate-only/--missing-evaluate-only and --skip-eval cannot be used together."
        )

    if args.evaluate_only and (args.infer_only or args.missing_infer_only):
        raise ValueError(
            "--evaluate-only/--missing-evaluate-only cannot be combined with "
            "--infer-only or --missing-infer-only."
        )

    if not args.evaluate_only and not args.skip_infer:
        if args.baseline_infer:
            if not train_input.exists():
                raise FileNotFoundError(
                    f"Training inference input does not exist: {train_input}"
                )
        else:
            if not good_input.exists():
                raise FileNotFoundError(f"Good inference input does not exist: {good_input}")
            if not bad_input.exists():
                raise FileNotFoundError(f"Bad inference input does not exist: {bad_input}")

    if not args.skip_eval and not args.baseline_infer and not channel_map.exists():
        raise FileNotFoundError(f"Channel map for evaluation does not exist: {channel_map}")

    config_paths = sorted(config_dir.glob(args.pattern))
    if not config_paths:
        raise FileNotFoundError(f"No configs matching {args.pattern!r} found in {config_dir}")

    runs_root.mkdir(parents=True, exist_ok=True)

    train_base_cmd = shlex.split(args.train_cmd)
    infer_base_cmd = shlex.split(args.infer_cmd)
    eval_base_cmd = shlex.split(args.eval_cmd)
    baseline_base_cmd = shlex.split(args.baseline_cmd)

    if args.missing_infer_only:
        args.infer_only = True

    if args.infer_only and args.skip_infer:
        raise ValueError("--infer-only/--missing-infer-only and --skip-infer cannot be used together.")

    conn = init_db(db_path)

    if args.restore_missing_db:
        print("=" * 80)
        print("Restoring completed models missing from the database...")
        print("Ignored model directories are included in restore mode.")
        print("=" * 80)

        restored_count = 0

        try:
            for idx, config_path in enumerate(config_paths, start=1):
                run_name = safe_name(config_path.stem)
                run_dir = runs_root / run_name

                print(f"[{idx}/{len(config_paths)}] checking {run_name}")

                if not run_dir.exists():
                    print(f"Skipping: run directory does not exist: {run_dir}")
                    continue

                restored = restore_completed_experiment_to_db(
                    conn,
                    config_path=config_path,
                    run_dir=run_dir,
                    channel_map=channel_map,
                    train_base_cmd=train_base_cmd,
                    infer_base_cmd=infer_base_cmd,
                    eval_base_cmd=eval_base_cmd,
                    args=args,
                )
                if restored:
                    restored_count += 1

            reorder_experiment_ids(conn, config_paths)

            print("=" * 80)
            print(f"Restored database rows: {restored_count}")
            print("=" * 80)

            if args.export_summary:
                export_summary(conn, db_path.parent)

            if conn.in_transaction:
                conn.commit()
            conn.execute("PRAGMA optimize")
            return 0
        finally:
            conn.close()

    print("=" * 80)
    print(f"GraphVAE config directory: {config_dir}")
    print(f"Number of configs: {len(config_paths)}")
    print(f"Runs root: {runs_root}")
    print(f"Database: {db_path}")
    print(f"Good inference input: {good_input}")
    print(f"Bad inference input:  {bad_input}")
    print(f"Baseline inference mode: {args.baseline_infer}")
    if args.baseline_infer:
        print(f"Training inference input: {train_input}")
    print(f"Batch mode: {args.batch}")
    print(f"Evaluate-only mode: {args.evaluate_only}")
    print(f"Missing-evaluate-only mode: {args.missing_evaluate_only}")
    print(f"Infer-only mode: {args.infer_only}")
    print(f"Missing-infer-only mode: {args.missing_infer_only}")
    print(f"Per-channel plotting: {args.per_channel_plot}")
    print(
        "Selected-window per-channel plotting: "
        f"{args.per_channel_per_window_plot}"
    )
    if args.per_channel_per_window_plot:
        if args.per_channel_window_indices is None:
            print("Selected window indices: plotter default")
        else:
            print(f"Selected window indices: {args.per_channel_window_indices}")
        print(f"Selected window datasets: {args.per_channel_window_datasets}")
    print(f"Missing-plot mode: {args.missing_plot}")
    print(f"Force-replot mode: {args.force_replot}")
    print(f"Skip evaluation: {args.skip_eval}")
    if not args.skip_eval:
        print(f"Evaluation command: {args.eval_cmd}")
        if args.baseline_infer:
            print("Evaluation aggregator: zscore_mean")
            print("Evaluation baseline: inference_result/channel_baseline.npz")
            print(f"Evaluation plot: {args.baseline_eval_plot_name}")
        else:
            print(f"Evaluation aggregator: {args.eval_aggregator}")
            print(f"Evaluation channel map: {channel_map}")
        if args.eval_threshold is not None:
            print(f"Evaluation threshold: {args.eval_threshold} (explicit; percentile ignored by window_score)")
        elif args.baseline_infer:
            print(f"Evaluation percentile: {args.baseline_eval_percentile}")
        else:
            print(f"Evaluation percentile: {args.eval_percentile}")
    if args.batch:
        print("Progress-bar suppression env enabled: SBN_BATCH=1, TQDM_DISABLE=1")
    print("=" * 80)

    for idx, config_path in enumerate(config_paths, start=1):
        t0 = time.perf_counter()

        config_stem = safe_name(config_path.stem)
        run_name = f"{config_stem}"
        run_dir = runs_root / run_name

        # Highest-priority exclusion: ignored model directories are skipped
        # before any training, inference, evaluation, plotting, rewrite,
        # repair, or database activity for that model.
        if is_ignored_run_dir(run_dir):
            print("\n" + "=" * 80)
            print(f"[{idx}/{len(config_paths)}] {config_path}")
            print(f"IGNORED: skipping model directory completely: {run_dir}")
            print("=" * 80)
            continue

        inference_dir = run_dir / "inference_result"
        good_score_npz_path = inference_dir / "scores_good.npz"
        bad_score_npz_path = inference_dir / "scores_bad.npz"
        baseline_score_npz_path = inference_dir / DEFAULT_BASELINE_SCORES_NAME
        channel_baseline_npz_path = inference_dir / DEFAULT_CHANNEL_BASELINE_NAME
        baseline_eval_plot_path = inference_dir / args.baseline_eval_plot_name
        baseline_eval_log_path = inference_dir / args.baseline_eval_log_name
        baseline_eval_json_path = inference_dir / args.baseline_eval_json_name
        eval_plot_path = inference_dir / args.eval_plot_name
        eval_log_path = inference_dir / args.eval_log_name
        eval_json_path = inference_dir / args.eval_json_name

        if args.missing_plot or args.force_replot:
            print("\n" + "=" * 80)
            print(f"[{idx}/{len(config_paths)}] {config_path}")
            if args.force_replot:
                print(f"Force-replotting plots for model directory: {run_dir}")
            else:
                print(f"Checking plots for model directory: {run_dir}")

            missing_score_paths = [
                path
                for path in (good_score_npz_path, bad_score_npz_path)
                if not path.exists()
            ]
            if missing_score_paths:
                print(
                    "Skipping: required inference score file(s) are missing, so "
                    "the plots cannot be produced."
                )
                for path in missing_score_paths:
                    print(f"  - {path}")
                print("=" * 80)
                continue

            per_channel_plot_path = get_per_channel_plot_path(
                plot_script_path=PER_CHANNEL_PLOT_SCRIPT,
                inference_dir=inference_dir,
            )
            per_channel_per_window_plot_paths = (
                get_per_channel_per_window_plot_paths(
                    plot_script_path=PER_CHANNEL_PER_WINDOW_PLOT_SCRIPT,
                    inference_dir=inference_dir,
                    output_name=args.per_channel_per_window_plot_name,
                )
            )
            missing_per_channel_per_window_plot_paths = [
                path
                for path in per_channel_per_window_plot_paths
                if not path.exists()
            ]

            need_per_channel_plot = (
                args.force_replot or not per_channel_plot_path.exists()
            )
            need_per_channel_per_window_plot = (
                args.force_replot
                or bool(missing_per_channel_per_window_plot_paths)
            )
            need_goodvsbad_plot = (
                args.force_replot or not eval_plot_path.exists()
            )

            if need_per_channel_plot:
                if args.force_replot and per_channel_plot_path.exists():
                    print(
                        "Existing per-channel plot found; rerunning because "
                        "--force-replot was set."
                    )
                else:
                    print("Per-channel plot is missing; generating it now.")
                print(f"Expected per-channel plot: {per_channel_plot_path}")

                run_per_channel_score_plot(
                    plot_script_path=PER_CHANNEL_PLOT_SCRIPT,
                    inference_dir=inference_dir,
                )

                if per_channel_plot_path.exists():
                    print(f"Created per-channel plot: {per_channel_plot_path}")
                else:
                    print(
                        "WARNING: plotting finished, but the expected per-channel "
                        f"plot was not found: {per_channel_plot_path}"
                    )
            else:
                print("Per-channel plot already exists; no replot needed.")
                print(f"Existing per-channel plot: {per_channel_plot_path}")

            if need_per_channel_per_window_plot:
                if (
                    args.force_replot
                    and not missing_per_channel_per_window_plot_paths
                ):
                    print(
                        "Existing selected-window per-channel plane plots found; "
                        "rerunning because --force-replot was set."
                    )
                else:
                    print(
                        "One or more selected-window per-channel plane plots "
                        "are missing; generating all six now."
                    )
                print("Expected selected-window per-channel plane plots:")
                for path in per_channel_per_window_plot_paths:
                    marker = "missing" if not path.exists() else "exists"
                    print(f"  - {path} [{marker}]")

                run_per_channel_per_window_score_plot(
                    plot_script_path=PER_CHANNEL_PER_WINDOW_PLOT_SCRIPT,
                    inference_dir=inference_dir,
                    window_spec=args.per_channel_window_indices,
                    datasets=args.per_channel_window_datasets,
                    output_name=args.per_channel_per_window_plot_name,
                )

                remaining_missing_paths = [
                    path
                    for path in per_channel_per_window_plot_paths
                    if not path.exists()
                ]
                if not remaining_missing_paths:
                    print("Created all selected-window per-channel plane plots.")
                else:
                    print(
                        "WARNING: plotting finished, but these expected plane "
                        "plots were not found:"
                    )
                    for path in remaining_missing_paths:
                        print(f"  - {path}")
            else:
                print(
                    "All selected-window per-channel plane plots already exist; "
                    "no replot needed."
                )
                for path in per_channel_per_window_plot_paths:
                    print(f"  - {path}")

            if need_goodvsbad_plot:
                if args.force_replot and eval_plot_path.exists():
                    print(
                        "Existing good-vs-bad plot found; rerunning because "
                        "--force-replot was set."
                    )
                else:
                    print("Good-vs-bad plot is missing; generating it now.")
                print(f"Expected good-vs-bad plot: {eval_plot_path}")

                plot_eval_cmd = eval_base_cmd + [
                    "--scores",
                    str(good_score_npz_path),
                    "--compare",
                    str(bad_score_npz_path),
                    "--labels",
                    args.good_label,
                    args.bad_label,
                    "--aggregator",
                    args.eval_aggregator,
                    "--channel-map",
                    str(channel_map),
                    "--plot",
                    str(eval_plot_path),
                ]
                if args.eval_threshold is not None:
                    plot_eval_cmd += ["--threshold", str(args.eval_threshold)]
                elif args.eval_percentile is not None:
                    plot_eval_cmd += ["--percentile", str(args.eval_percentile)]
                if args.eval_per_run:
                    plot_eval_cmd += ["--per-run"]

                plot_returncode = run_command(
                    plot_eval_cmd,
                    cwd=PROJECT_DIR,
                    timeout=args.timeout,
                    monitor_interval=args.monitor_interval,
                    batch=args.batch,
                )

                if plot_returncode != 0:
                    print(
                        "ERROR: good-vs-bad plotting failed with return code "
                        f"{plot_returncode}."
                    )
                    print("=" * 80)
                    if args.stop_on_error:
                        return plot_returncode
                    continue

                if eval_plot_path.exists():
                    print(f"Created good-vs-bad plot: {eval_plot_path}")
                else:
                    print(
                        "WARNING: window_score finished successfully, but the expected "
                        f"good-vs-bad plot was not found: {eval_plot_path}"
                    )
            else:
                print("Good-vs-bad plot already exists; no replot needed.")
                print(f"Existing good-vs-bad plot: {eval_plot_path}")

            print("=" * 80)
            continue

        if (
            args.missing_infer_only
            and (
                (
                    args.baseline_infer
                    and baseline_score_npz_path.exists()
                    and channel_baseline_npz_path.exists()
                    and (
                        args.skip_eval
                        or (
                            baseline_eval_plot_path.exists()
                            and baseline_eval_log_path.exists()
                            and baseline_eval_json_path.exists()
                        )
                    )
                )
                or (
                    not args.baseline_infer
                    and good_score_npz_path.exists()
                    and bad_score_npz_path.exists()
                    and (args.skip_eval or eval_json_path.exists())
                )
            )
        ):
            print("\n" + "=" * 80)
            print(f"[{idx}/{len(config_paths)}] {config_path}")
            if args.baseline_infer:
                print(
                    "Skipping because --missing-infer was set and both baseline "
                    "inference outputs already exist."
                )
                print(f"Existing training scores: {baseline_score_npz_path}")
                print(f"Existing channel baseline: {channel_baseline_npz_path}")
                if not args.skip_eval:
                    print(f"Existing z-score plot: {baseline_eval_plot_path}")
                    print(f"Existing z-score evaluation log: {baseline_eval_log_path}")
                    print(f"Existing z-score evaluation JSON: {baseline_eval_json_path}")
            else:
                print(
                    "Skipping because --missing-infer-only was set, both inference "
                    "outputs already exist, and evaluation is not requested or already exists."
                )
                print(f"Existing good inference output: {good_score_npz_path}")
                print(f"Existing bad inference output:  {bad_score_npz_path}")
                if not args.skip_eval:
                    print(f"Existing evaluation summary: {eval_json_path}")
            print("=" * 80)
            continue

        missing_eval_output_paths = evaluation_outputs_missing(
            eval_plot_path=eval_plot_path,
            eval_log_path=eval_log_path,
            eval_json_path=eval_json_path,
        )

        if args.missing_evaluate_only and not missing_eval_output_paths:
            print("\n" + "=" * 80)
            print(f"[{idx}/{len(config_paths)}] {config_path}")
            print(
                "Skipping because --missing-evaluate-only was set and all evaluation "
                "outputs already exist."
            )
            print(f"Existing evaluation plot: {eval_plot_path}")
            print(f"Existing evaluation log:  {eval_log_path}")
            print(f"Existing evaluation JSON: {eval_json_path}")
            print("=" * 80)
            continue

        if args.missing_evaluate_only:
            print("\n" + "=" * 80)
            print(f"[{idx}/{len(config_paths)}] {config_path}")
            print(
                "--missing-evaluate-only was set and at least one evaluation output "
                "is missing, so evaluation will be rerun from the saved scores."
            )
            print("Missing evaluation output(s):")
            for path in missing_eval_output_paths:
                print(f"  - {path}")
            print("=" * 80)

        existing_experiment = get_existing_experiment(conn, run_name)
        existing_status = None
        if existing_experiment is not None:
            existing_status = str(existing_experiment.get("status") or "")

        if existing_experiment is not None:
            missing_rewrite_needed = False
            missing_rewrite_reason = ""
            if args.missing_rewrite:
                missing_rewrite_needed, missing_rewrite_reason = run_directory_needs_rewrite(run_dir)

            if args.evaluate_only:
                print("\n" + "=" * 80)
                print(f"[{idx}/{len(config_paths)}] {config_path}")
                print(f"Run name conflict: {run_name!r}")
                if args.missing_evaluate_only:
                    print(
                        "missing_evaluate_only=True and evaluation output(s) are missing, "
                        "so the old database record will be replaced and only the saved "
                        "good/bad score files will be re-evaluated."
                    )
                else:
                    print(
                        "evaluate_only=True, so the old database record will be replaced "
                        "and only the saved good/bad score files will be re-evaluated."
                    )
                print(f"Existing status: {existing_experiment.get('status')}")
                print(f"Existing good scores: {existing_experiment.get('good_score_npz_path')}")
                print(f"Existing bad scores:  {existing_experiment.get('bad_score_npz_path')}")
                print("=" * 80)
                delete_existing_experiment(
                    conn,
                    run_name,
                    reason=(
                        "missing_evaluate_only=True"
                        if args.missing_evaluate_only
                        else "evaluate_only=True"
                    ),
                )
            elif args.infer_only:
                print("\n" + "=" * 80)
                print(f"[{idx}/{len(config_paths)}] {config_path}")
                print(f"Run name conflict: {run_name!r}")
                if args.baseline_infer:
                    print(
                        "infer_only=True with baseline_infer=True, so the old database "
                        "record will be replaced and baseline inference will be rerun "
                        "using the existing checkpoint."
                    )
                else:
                    print(
                        "infer_only=True, so the old database record will be replaced "
                        "and both good/bad inference jobs will be rerun using the existing checkpoint."
                    )
                print(f"Existing status: {existing_experiment.get('status')}")
                print(f"Existing final model: {existing_experiment.get('final_model_path')}")
                print("=" * 80)
                delete_existing_experiment(conn, run_name, reason="infer_only=True")
            elif args.force_rewrite:
                delete_existing_experiment(conn, run_name, reason="force_rewrite=True")
            elif existing_status in {"failed", "running", "missing_weights", "missing_scores"}:
                print("\n" + "=" * 80)
                print(f"[{idx}/{len(config_paths)}] {config_path}")
                print(f"Run name conflict: {run_name!r}")
                print(
                    f"Existing record has status={existing_status!r}, so it will be rewritten."
                )
                print("=" * 80)
                delete_existing_experiment(conn, run_name, reason=f"existing status={existing_status!r}")
            elif missing_rewrite_needed:
                print("\n" + "=" * 80)
                print(f"[{idx}/{len(config_paths)}] {config_path}")
                print(f"Run name conflict: {run_name!r}")
                print("missing_rewrite=True and run directory is missing/incomplete; rerunning.")
                print(f"Reason: {missing_rewrite_reason}")
                print("=" * 80)
                delete_existing_experiment(
                    conn,
                    run_name,
                    reason=f"missing_rewrite=True ({missing_rewrite_reason})",
                )
            else:
                print("\n" + "=" * 80)
                print(f"[{idx}/{len(config_paths)}] {config_path}")
                print(f"Run name conflict: {run_name!r}")
                print(
                    "force_rewrite=False, so training/inference will be skipped "
                    "and the existing model/database record will be kept."
                )
                if args.missing_rewrite:
                    print(f"missing_rewrite check: {missing_rewrite_reason}")
                print(f"Existing status: {existing_experiment.get('status')}")
                print(f"Existing run directory: {existing_experiment.get('run_dir')}")
                print(f"Existing final model: {existing_experiment.get('final_model_path')}")
                print("=" * 80)
                continue

        run_dir.mkdir(parents=True, exist_ok=True)
        inference_dir.mkdir(parents=True, exist_ok=True)

        run_config_path = run_dir / "config_run.yaml"

        print("\n" + "=" * 80)
        print(f"[{idx}/{len(config_paths)}] {config_path}")
        print(f"Run directory: {run_dir}")
        print(f"Inference directory: {inference_dir}")
        print("=" * 80)

        train_returncode: int | None = None
        infer_good_returncode: int | None = None
        infer_bad_returncode: int | None = None
        eval_returncode: int | None = None
        experiment_id: int | None = None
        metrics: dict[str, Any] = {}
        error: str | None = None
        status = "failed"
        final_model_path: Path | None = None

        try:
            original_cfg = load_yaml(config_path)
            (
                patched_cfg,
                checkpoint_dir,
                inference_dir,
                final_model_path,
                good_score_npz_path,
                bad_score_npz_path,
            ) = prepare_run_config(original_cfg, run_dir)

            baseline_score_npz_path = inference_dir / DEFAULT_BASELINE_SCORES_NAME
            channel_baseline_npz_path = inference_dir / DEFAULT_CHANNEL_BASELINE_NAME
            baseline_eval_plot_path = inference_dir / args.baseline_eval_plot_name
            baseline_eval_log_path = inference_dir / args.baseline_eval_log_name
            baseline_eval_json_path = inference_dir / args.baseline_eval_json_name
            if args.baseline_infer:
                patched_cfg.setdefault("inference", {})["output_path"] = str(
                    baseline_score_npz_path
                )

            save_yaml(run_config_path, patched_cfg)

            train_cmd = train_base_cmd + ["--config", str(run_config_path)]
            infer_good_cmd = infer_base_cmd + [
                "--config",
                str(run_config_path),
                "--input",
                str(good_input),
                "--output",
                str(good_score_npz_path),
            ]
            infer_bad_cmd = infer_base_cmd + [
                "--config",
                str(run_config_path),
                "--input",
                str(bad_input),
                "--output",
                str(bad_score_npz_path),
            ]
            baseline_infer_cmd = infer_base_cmd + [
                "--config",
                str(run_config_path),
                "--input",
                str(train_input),
                "--output",
                str(baseline_score_npz_path),
            ]
            baseline_build_cmd = baseline_base_cmd + [
                "--scores",
                str(baseline_score_npz_path),
                "--output",
                str(channel_baseline_npz_path),
            ]
            eval_cmd = eval_base_cmd + [
                "--scores",
                str(good_score_npz_path),
                "--compare",
                str(bad_score_npz_path),
                "--labels",
                args.good_label,
                args.bad_label,
                "--aggregator",
                args.eval_aggregator,
                "--channel-map",
                str(channel_map),
                "--plot",
                str(eval_plot_path),
            ]
            if args.eval_threshold is not None:
                eval_cmd += ["--threshold", str(args.eval_threshold)]
            elif args.eval_percentile is not None:
                eval_cmd += ["--percentile", str(args.eval_percentile)]
            if args.eval_per_run:
                eval_cmd += ["--per-run"]

            baseline_eval_cmd = eval_base_cmd + [
                "--scores",
                str(good_score_npz_path),
                "--compare",
                str(bad_score_npz_path),
                "--labels",
                args.good_label,
                args.bad_label,
                "--aggregator",
                "zscore_mean",
                "--baseline",
                str(channel_baseline_npz_path),
                "--plot",
                str(baseline_eval_plot_path),
            ]
            if args.eval_threshold is not None:
                baseline_eval_cmd += ["--threshold", str(args.eval_threshold)]
            else:
                baseline_eval_cmd += [
                    "--percentile",
                    str(args.baseline_eval_percentile),
                ]
            if args.eval_per_run:
                baseline_eval_cmd += ["--per-run"]

            if args.baseline_infer:
                recorded_score_path_1 = baseline_score_npz_path
                recorded_score_path_2 = channel_baseline_npz_path
                recorded_infer_cmd_1 = baseline_infer_cmd
                recorded_infer_cmd_2 = baseline_build_cmd
                recorded_eval_cmd = baseline_eval_cmd
                active_eval_cmd = baseline_eval_cmd
                active_eval_plot_path = baseline_eval_plot_path
                active_eval_log_path = baseline_eval_log_path
                active_eval_json_path = baseline_eval_json_path
            else:
                recorded_score_path_1 = good_score_npz_path
                recorded_score_path_2 = bad_score_npz_path
                recorded_infer_cmd_1 = infer_good_cmd
                recorded_infer_cmd_2 = infer_bad_cmd
                recorded_eval_cmd = eval_cmd
                active_eval_cmd = eval_cmd
                active_eval_plot_path = eval_plot_path
                active_eval_log_path = eval_log_path
                active_eval_json_path = eval_json_path

            experiment_id = insert_experiment(
                conn,
                run_name=run_name,
                config_name=config_path.name,
                original_config_path=config_path,
                run_config_path=run_config_path,
                run_dir=run_dir,
                checkpoint_dir=checkpoint_dir,
                inference_dir=inference_dir,
                final_model_path=final_model_path,
                good_score_npz_path=recorded_score_path_1,
                bad_score_npz_path=recorded_score_path_2,
                train_cmd=train_cmd,
                infer_good_cmd=recorded_infer_cmd_1,
                infer_bad_cmd=recorded_infer_cmd_2,
                eval_cmd=recorded_eval_cmd,
                eval_plot_path=active_eval_plot_path,
                eval_log_path=active_eval_log_path,
                eval_json_path=active_eval_json_path,
                patched_cfg=patched_cfg,
            )
            insert_params(conn, experiment_id, patched_cfg)

            if args.evaluate_only:
                if args.missing_evaluate_only:
                    print(
                        "Skipping training and inference because --missing-evaluate-only "
                        "reruns only missing evaluation outputs."
                    )
                else:
                    print("Skipping training and inference because --evaluate-only was set.")
                print(f"Using existing good score file: {good_score_npz_path}")
                print(f"Using existing bad score file:  {bad_score_npz_path}")

                missing_score_paths = [
                    path
                    for path in (good_score_npz_path, bad_score_npz_path)
                    if not path.exists()
                ]
                if missing_score_paths:
                    error = (
                        "Saved score file(s) missing for evaluate-only mode: "
                        + ", ".join(str(path) for path in missing_score_paths)
                    )
                    print(f"ERROR: {error}")
                    status = "missing_scores"
                    metrics = {
                        "checkpoint.exists": int(final_model_path.exists()),
                        "checkpoint.expected_path": str(final_model_path),
                        "good.score_npz.exists": int(good_score_npz_path.exists()),
                        "good.score_npz.path": str(good_score_npz_path),
                        "bad.score_npz.exists": int(bad_score_npz_path.exists()),
                        "bad.score_npz.path": str(bad_score_npz_path),
                    }
                else:
                    if not final_model_path.exists():
                        print(
                            "WARNING: expected checkpoint is missing, but evaluation only "
                            "uses saved score files, so evaluation will still run."
                        )
                        print(f"Expected checkpoint: {final_model_path}")

                    print("Good-vs-bad evaluation...")
                    eval_returncode = run_command(
                        eval_cmd,
                        cwd=PROJECT_DIR,
                        timeout=args.timeout,
                        monitor_interval=args.monitor_interval,
                        batch=args.batch,
                        log_path=eval_log_path,
                    )
                    if eval_returncode != 0:
                        raise RuntimeError(f"Evaluation failed with return code {eval_returncode}")

                    status = "success"

            elif args.infer_only:
                if not final_model_path.exists():
                    error = (
                        f"Expected trained weight file is missing for run_name={run_name!r}: "
                        f"{final_model_path}"
                    )
                    print(f"ERROR: {error}")
                    status = "missing_weights"
                else:
                    if args.missing_infer_only:
                        print("Skipping training because --missing-infer-only was set.")
                    else:
                        print("Skipping training because --infer-only was set.")
                    print(f"Using existing checkpoint: {final_model_path}")

                    if args.baseline_infer:
                        print("Training-set inference for channel baseline...")
                        infer_good_returncode = run_command(
                            baseline_infer_cmd,
                            cwd=PROJECT_DIR,
                            timeout=args.timeout,
                            monitor_interval=args.monitor_interval,
                            batch=args.batch,
                        )
                        if infer_good_returncode != 0:
                            raise RuntimeError(
                                "Baseline training-set inference failed with return "
                                f"code {infer_good_returncode}"
                            )

                        print("Building per-channel baseline...")
                        infer_bad_returncode = run_command(
                            baseline_build_cmd,
                            cwd=PROJECT_DIR,
                            timeout=args.timeout,
                            monitor_interval=args.monitor_interval,
                            batch=args.batch,
                        )
                        if infer_bad_returncode != 0:
                            raise RuntimeError(
                                "Channel-baseline construction failed with return "
                                f"code {infer_bad_returncode}"
                            )
                    else:
                        print("Good inference/scoring...")
                        infer_good_returncode = run_command(
                            infer_good_cmd,
                            cwd=PROJECT_DIR,
                            timeout=args.timeout,
                            monitor_interval=args.monitor_interval,
                            batch=args.batch,
                        )
                        if infer_good_returncode != 0:
                            raise RuntimeError(f"Good inference failed with return code {infer_good_returncode}")

                        print("Bad inference/scoring...")
                        infer_bad_returncode = run_command(
                            infer_bad_cmd,
                            cwd=PROJECT_DIR,
                            timeout=args.timeout,
                            monitor_interval=args.monitor_interval,
                            batch=args.batch,
                        )
                        if infer_bad_returncode != 0:
                            raise RuntimeError(f"Bad inference failed with return code {infer_bad_returncode}")

                    if args.per_channel_plot:
                        run_per_channel_score_plot(
                            plot_script_path=PER_CHANNEL_PLOT_SCRIPT,
                            inference_dir=inference_dir,
                        )

                    if args.per_channel_per_window_plot:
                        run_per_channel_per_window_score_plot(
                            plot_script_path=PER_CHANNEL_PER_WINDOW_PLOT_SCRIPT,
                            inference_dir=inference_dir,
                            window_spec=args.per_channel_window_indices,
                            datasets=args.per_channel_window_datasets,
                            output_name=args.per_channel_per_window_plot_name,
                        )

                    if args.skip_eval:
                        print("Skipping good-vs-bad evaluation because --skip-eval was set.")
                    else:
                        if args.baseline_infer:
                            missing_test_scores = [
                                path
                                for path in (good_score_npz_path, bad_score_npz_path)
                                if not path.exists()
                            ]
                            if missing_test_scores:
                                raise FileNotFoundError(
                                    "Baseline z-score evaluation requires existing "
                                    "scores_good.npz and scores_bad.npz; missing: "
                                    + ", ".join(str(path) for path in missing_test_scores)
                                )
                            print("Good-vs-bad zscore_mean evaluation...")
                        else:
                            print("Good-vs-bad evaluation...")
                        eval_returncode = run_command(
                            active_eval_cmd,
                            cwd=PROJECT_DIR,
                            timeout=args.timeout,
                            monitor_interval=args.monitor_interval,
                            batch=args.batch,
                            log_path=active_eval_log_path,
                        )
                        if eval_returncode != 0:
                            raise RuntimeError(f"Evaluation failed with return code {eval_returncode}")

                    status = "success"

            else:
                print("Training...")
                train_returncode = run_command(
                    train_cmd,
                    cwd=PROJECT_DIR,
                    timeout=args.timeout,
                    monitor_interval=args.monitor_interval,
                    batch=args.batch,
                )
                if train_returncode != 0:
                    raise RuntimeError(f"Training failed with return code {train_returncode}")

                if args.skip_infer:
                    print("Skipping both good/bad inference jobs because --skip-infer was set.")
                    status = "trained_no_infer"
                else:
                    if args.baseline_infer:
                        print("Training-set inference for channel baseline...")
                        infer_good_returncode = run_command(
                            baseline_infer_cmd,
                            cwd=PROJECT_DIR,
                            timeout=args.timeout,
                            monitor_interval=args.monitor_interval,
                            batch=args.batch,
                        )
                        if infer_good_returncode != 0:
                            raise RuntimeError(
                                "Baseline training-set inference failed with return "
                                f"code {infer_good_returncode}"
                            )

                        print("Building per-channel baseline...")
                        infer_bad_returncode = run_command(
                            baseline_build_cmd,
                            cwd=PROJECT_DIR,
                            timeout=args.timeout,
                            monitor_interval=args.monitor_interval,
                            batch=args.batch,
                        )
                        if infer_bad_returncode != 0:
                            raise RuntimeError(
                                "Channel-baseline construction failed with return "
                                f"code {infer_bad_returncode}"
                            )
                    else:
                        print("Good inference/scoring...")
                        infer_good_returncode = run_command(
                            infer_good_cmd,
                            cwd=PROJECT_DIR,
                            timeout=args.timeout,
                            monitor_interval=args.monitor_interval,
                            batch=args.batch,
                        )
                        if infer_good_returncode != 0:
                            raise RuntimeError(f"Good inference failed with return code {infer_good_returncode}")

                        print("Bad inference/scoring...")
                        infer_bad_returncode = run_command(
                            infer_bad_cmd,
                            cwd=PROJECT_DIR,
                            timeout=args.timeout,
                            monitor_interval=args.monitor_interval,
                            batch=args.batch,
                        )
                        if infer_bad_returncode != 0:
                            raise RuntimeError(f"Bad inference failed with return code {infer_bad_returncode}")

                    if args.per_channel_plot:
                        run_per_channel_score_plot(
                            plot_script_path=PER_CHANNEL_PLOT_SCRIPT,
                            inference_dir=inference_dir,
                        )

                    if args.per_channel_per_window_plot:
                        run_per_channel_per_window_score_plot(
                            plot_script_path=PER_CHANNEL_PER_WINDOW_PLOT_SCRIPT,
                            inference_dir=inference_dir,
                            window_spec=args.per_channel_window_indices,
                            datasets=args.per_channel_window_datasets,
                            output_name=args.per_channel_per_window_plot_name,
                        )

                    if args.skip_eval:
                        print("Skipping good-vs-bad evaluation because --skip-eval was set.")
                    else:
                        if args.baseline_infer:
                            missing_test_scores = [
                                path
                                for path in (good_score_npz_path, bad_score_npz_path)
                                if not path.exists()
                            ]
                            if missing_test_scores:
                                raise FileNotFoundError(
                                    "Baseline z-score evaluation requires existing "
                                    "scores_good.npz and scores_bad.npz; missing: "
                                    + ", ".join(str(path) for path in missing_test_scores)
                                )
                            print("Good-vs-bad zscore_mean evaluation...")
                        else:
                            print("Good-vs-bad evaluation...")
                        eval_returncode = run_command(
                            active_eval_cmd,
                            cwd=PROJECT_DIR,
                            timeout=args.timeout,
                            monitor_interval=args.monitor_interval,
                            batch=args.batch,
                            log_path=active_eval_log_path,
                        )
                        if eval_returncode != 0:
                            raise RuntimeError(f"Evaluation failed with return code {eval_returncode}")

                    status = "success"

            if status == "missing_weights":
                metrics = {
                    "checkpoint.exists": 0,
                    "checkpoint.expected_path": str(final_model_path),
                }
                if args.baseline_infer:
                    metrics.update(
                        {
                            "baseline.scores.exists": int(baseline_score_npz_path.exists()),
                            "baseline.scores.path": str(baseline_score_npz_path),
                            "baseline.channel.exists": int(channel_baseline_npz_path.exists()),
                            "baseline.channel.path": str(channel_baseline_npz_path),
                        }
                    )
                else:
                    metrics.update(
                        {
                            "good.score_npz.exists": 0,
                            "bad.score_npz.exists": 0,
                        }
                    )
            elif status == "missing_scores":
                pass
            elif args.baseline_infer:
                metrics = {
                    "checkpoint.exists": int(final_model_path.exists()),
                    "checkpoint.expected_path": str(final_model_path),
                    "baseline.infer_mode": 1,
                    "baseline.scores.exists": int(baseline_score_npz_path.exists()),
                    "baseline.scores.path": str(baseline_score_npz_path),
                    "baseline.channel.exists": int(channel_baseline_npz_path.exists()),
                    "baseline.channel.path": str(channel_baseline_npz_path),
                    "baseline.infer_returncode": (
                        infer_good_returncode if infer_good_returncode is not None else -1
                    ),
                    "baseline.build_returncode": (
                        infer_bad_returncode if infer_bad_returncode is not None else -1
                    ),
                    "eval.skipped": int(args.skip_eval),
                }
                if not args.skip_eval:
                    metrics["eval.returncode"] = (
                        eval_returncode if eval_returncode is not None else -1
                    )
                    metrics["eval.aggregator"] = "zscore_mean"
                    metrics["eval.baseline_path"] = str(channel_baseline_npz_path)
                    metrics["eval.plot_path"] = str(active_eval_plot_path)
                    metrics["eval.log_path"] = str(active_eval_log_path)
                    metrics["eval.json_path"] = str(active_eval_json_path)
                    metrics["eval.plot.exists"] = int(active_eval_plot_path.exists())
                    metrics["eval.log.exists"] = int(active_eval_log_path.exists())
                    metrics.update(parse_window_score_eval_log(active_eval_log_path))
                    save_eval_summary_json(
                        active_eval_json_path,
                        eval_cmd=active_eval_cmd,
                        eval_returncode=eval_returncode,
                        eval_plot_path=active_eval_plot_path,
                        eval_log_path=active_eval_log_path,
                        good_score_npz_path=good_score_npz_path,
                        bad_score_npz_path=bad_score_npz_path,
                        metrics=metrics,
                    )
            else:
                metrics = collect_run_metrics(
                    checkpoint_dir=checkpoint_dir,
                    good_score_npz_path=recorded_score_path_1,
                    bad_score_npz_path=recorded_score_path_2,
                )
                metrics["eval.skipped"] = int(args.skip_eval)
                metrics["eval.evaluate_only"] = int(args.evaluate_only)
                metrics["eval.missing_evaluate_only"] = int(args.missing_evaluate_only)
                if not args.skip_eval:
                    metrics["eval.returncode"] = eval_returncode if eval_returncode is not None else -1
                    metrics["eval.plot_path"] = str(active_eval_plot_path)
                    metrics["eval.log_path"] = str(active_eval_log_path)
                    metrics["eval.json_path"] = str(active_eval_json_path)
                    metrics["eval.plot.exists"] = int(active_eval_plot_path.exists())
                    metrics["eval.log.exists"] = int(active_eval_log_path.exists())
                    metrics.update(parse_window_score_eval_log(active_eval_log_path))
                    save_eval_summary_json(
                        active_eval_json_path,
                        eval_cmd=active_eval_cmd,
                        eval_returncode=eval_returncode,
                        eval_plot_path=active_eval_plot_path,
                        eval_log_path=active_eval_log_path,
                        good_score_npz_path=good_score_npz_path,
                        bad_score_npz_path=bad_score_npz_path,
                        metrics=metrics,
                    )

            insert_metrics(conn, experiment_id, metrics)

        except Exception as exc:
            error = repr(exc)
            print(f"ERROR: {error}")

        finally:
            duration_sec = time.perf_counter() - t0

            if experiment_id is not None:
                update_experiment_done(
                    conn,
                    experiment_id,
                    status=status,
                    duration_sec=duration_sec,
                    train_returncode=train_returncode,
                    infer_good_returncode=infer_good_returncode,
                    infer_bad_returncode=infer_bad_returncode,
                    eval_returncode=eval_returncode,
                    good_score_npz_path=recorded_score_path_1,
                    bad_score_npz_path=recorded_score_path_2,
                    error=error,
                    metrics=metrics,
                )

            print(f"Status: {status}")
            print(f"Duration: {duration_sec:.1f} s")

            if error and args.stop_on_error:
                print("Stopping because --stop-on-error was set.")
                break

    if args.export_summary:
        export_summary(conn, db_path.parent)
    else:
        print("Skipping CSV/XLSX summary export because --export-summary was not set.")

    print("Finalizing SQLite database before closing...", flush=True)

    if conn.in_transaction:
        conn.commit()

    # The fixed connection uses journal_mode=DELETE, so there is no WAL to
    # checkpoint. optimize is safe and updates SQLite's query-planner metadata.
    conn.execute("PRAGMA optimize")
    conn.close()
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a GraphVAE YAML sweep: train, infer good/bad separately, and record results in SQLite."
    )

    parser.add_argument(
        "--config-dir",
        default=str(DEFAULT_CONFIG_DIR),
        help="Directory containing GraphVAE YAML configs.",
    )
    parser.add_argument(
        "--runs-root",
        default=str(DEFAULT_RUNS_ROOT),
        help="Directory where per-run outputs will be written.",
    )
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help="SQLite database path.",
    )
    parser.add_argument(
        "--good-input",
        "--good_input",
        dest="good_input",
        default=str(DEFAULT_GOOD_INPUT),
        help="Good-test sparse events npz passed to sbn-infer --input.",
    )
    parser.add_argument(
        "--bad-input",
        "--bad_input",
        dest="bad_input",
        default=str(DEFAULT_BAD_INPUT),
        help="Bad-test sparse events npz passed to sbn-infer --input.",
    )
    parser.add_argument(
        "--train-input",
        "--train_input",
        dest="train_input",
        default=str(DEFAULT_TRAIN_INPUT),
        help=(
            "Training sparse-events NPZ used by --baseline-infer. The trained "
            "model infers this file to create scores_train.npz."
        ),
    )
    parser.add_argument(
        "--pattern",
        default="*.yaml",
        help="Glob pattern for config files.",
    )
    parser.add_argument(
        "--train-cmd",
        default="sbn-train",
        help='Training command, e.g. "sbn-train" or "python -m sbn_anomaly.train.cli".',
    )
    parser.add_argument(
        "--infer-cmd",
        default="sbn-infer",
        help='Inference command, e.g. "sbn-infer" or "python -m sbn_anomaly.infer.cli".',
    )
    parser.add_argument(
        "--baseline-cmd",
        "--baseline_cmd",
        dest="baseline_cmd",
        default=DEFAULT_BASELINE_CMD,
        help=(
            "Command that converts scores_train.npz into channel_baseline.npz "
            "when --baseline-infer is active."
        ),
    )
    parser.add_argument(
        "--eval-cmd",
        default="python -m sbn_anomaly.infer.window_score",
        help=(
            "Evaluation command run after good/bad inference. The script appends "
            "--scores, --compare, --labels, --aggregator, --channel-map, and --plot."
        ),
    )
    parser.add_argument(
        "--evaluate-only",
        "--evaluate_only",
        "--eval-only",
        "--eval_only",
        dest="evaluate_only",
        action="store_true",
        help=(
            "Do not train and do not rerun inference. For each YAML, use the existing "
            "<runs-root>/<config_stem>/inference_result/scores_good.npz and "
            "scores_bad.npz files and rerun only the good-vs-bad window_score evaluation. "
            "This lets you re-aggregate and re-threshold saved scores."
        ),
    )
    parser.add_argument(
        "--missing-evaluate-only",
        "--missing_evaluate_only",
        "--missing-eval-only",
        "--missing_eval_only",
        dest="missing_evaluate_only",
        action="store_true",
        help=(
            "Only rerun evaluation for runs missing at least one evaluation output file: "
            "the evaluation plot, evaluation log, or evaluation JSON summary. This implies "
            "--evaluate-only and uses existing scores_good.npz and scores_bad.npz files."
        ),
    )
    parser.add_argument(
        "--skip-eval",
        "--skip_eval",
        dest="skip_eval",
        action="store_true",
        help="Run train/inference only; do not run good-vs-bad window_score evaluation.",
    )
    parser.add_argument(
        "--eval-aggregator",
        "--eval_aggregator",
        dest="eval_aggregator",
        default=DEFAULT_EVAL_AGGREGATOR,
        help="Aggregator passed to sbn_anomaly.infer.window_score.",
    )
    parser.add_argument(
        "--channel-map",
        "--channel_map",
        dest="channel_map",
        default=str(DEFAULT_CHANNEL_MAP),
        help="Channel map CSV passed to sbn_anomaly.infer.window_score --channel-map.",
    )
    parser.add_argument(
        "--eval-threshold",
        "--eval_threshold",
        dest="eval_threshold",
        type=float,
        default=None,
        help="Optional explicit operating threshold passed to window_score. If set, this overrides --eval-percentile.",
    )
    parser.add_argument(
        "--eval-percentile",
        "--eval_percentile",
        dest="eval_percentile",
        type=float,
        default=DEFAULT_EVAL_PERCENTILE,
        help="Good-set percentile passed to window_score --percentile when --eval-threshold is not set. Default: %(default)s.",
    )
    parser.add_argument(
        "--eval-per-run",
        "--eval_per_run",
        dest="eval_per_run",
        action="store_true",
        help="Also pass --per-run to window_score.",
    )
    parser.add_argument(
        "--good-label",
        "--good_label",
        dest="good_label",
        default="good",
        help="First label passed to window_score --labels.",
    )
    parser.add_argument(
        "--bad-label",
        "--bad_label",
        dest="bad_label",
        default="bad",
        help="Second label passed to window_score --labels.",
    )
    parser.add_argument(
        "--eval-plot-name",
        "--eval_plot_name",
        dest="eval_plot_name",
        default=DEFAULT_EVAL_PLOT_NAME,
        help="Evaluation plot filename inside each inference_result directory.",
    )
    parser.add_argument(
        "--eval-log-name",
        "--eval_log_name",
        dest="eval_log_name",
        default=DEFAULT_EVAL_LOG_NAME,
        help="Evaluation stdout/stderr log filename inside each inference_result directory.",
    )
    parser.add_argument(
        "--eval-json-name",
        "--eval_json_name",
        dest="eval_json_name",
        default=DEFAULT_EVAL_JSON_NAME,
        help="Evaluation JSON summary filename inside each inference_result directory.",
    )
    parser.add_argument(
        "--baseline-eval-percentile",
        "--baseline_eval_percentile",
        dest="baseline_eval_percentile",
        type=float,
        default=DEFAULT_BASELINE_EVAL_PERCENTILE,
        help=(
            "Good-score percentile used by the zscore_mean evaluation after "
            "--baseline-infer. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--baseline-eval-plot-name",
        "--baseline_eval_plot_name",
        dest="baseline_eval_plot_name",
        default=DEFAULT_BASELINE_EVAL_PLOT_NAME,
        help="Z-score evaluation plot filename inside inference_result.",
    )
    parser.add_argument(
        "--baseline-eval-log-name",
        "--baseline_eval_log_name",
        dest="baseline_eval_log_name",
        default=DEFAULT_BASELINE_EVAL_LOG_NAME,
        help="Z-score evaluation log filename inside inference_result.",
    )
    parser.add_argument(
        "--baseline-eval-json-name",
        "--baseline_eval_json_name",
        dest="baseline_eval_json_name",
        default=DEFAULT_BASELINE_EVAL_JSON_NAME,
        help="Z-score evaluation JSON filename inside inference_result.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optional timeout in seconds for each train/infer subprocess.",
    )
    parser.add_argument(
        "--skip-infer",
        action="store_true",
        help="Only train models; do not run good/bad inference.",
    )
    parser.add_argument(
        "--infer-only",
        "--infer_only",
        "--inference-only",
        "--inference_only",
        dest="infer_only",
        action="store_true",
        help=(
            "Assume models are already trained and rerun both good/bad inference jobs only. "
            "For each YAML, the expected checkpoint is "
            "<runs-root>/<config_stem>/graph_vae_final.pt."
        ),
    )
    parser.add_argument(
        "--baseline-infer",
        "--baseline_infer",
        dest="baseline_infer",
        action="store_true",
        help=(
            "Switch every inference stage to channel-baseline mode. Instead of "
            "inferring good/bad test inputs, infer --train-input to "
            "inference_result/scores_train.npz, then build "
            "inference_result/channel_baseline.npz. Existing scores_good.npz and "
            "scores_bad.npz are then evaluated with zscore_mean and plotted. "
            "Compatible with --infer-only and --missing-infer."
        ),
    )
    parser.add_argument(
        "--missing-infer-only",
        "--missing-infer",
        "--missing_infer_only",
        "--missing_infer",
        "--missing-infer_only",
        "--missing_infer-only",
        dest="missing_infer_only",
        action="store_true",
        help=(
            "Only rerun inference for models missing required inference outputs. "
            "Normally these are scores_good.npz and scores_bad.npz; with "
            "--baseline-infer they are scores_train.npz, channel_baseline.npz, "
            "and (unless --skip-eval is set) the z-score plot/log/JSON outputs. "
            "This implies --infer-only."
        ),
    )
    parser.add_argument(
        "--per-channel-plot",
        "--per_channel_plot",
        "--per_channel-plot",
        "--per-channel_plot",
        "--per-channel-plots",
        "--per_channel_plots",
        "--per-channel_plots",
        "--per_channel-plots",
        dest="per_channel_plot",
        action="store_true",
        help=(
            "Run graphing/plot_per_channel_scores.py after every successful pair "
            "of good/bad inference jobs. This applies whenever inference is actually "
            "carried out, including normal train+infer, --infer-only, and "
            "--missing-infer-only reruns. It does not run in --evaluate-only mode "
            "because no inference is performed there."
        ),
    )
    parser.add_argument(
        "--per-channel-per-window-plot",
        "--per_channel_per_window_plot",
        dest="per_channel_per_window_plot",
        action="store_true",
        help=(
            "Run graphing/plot_per_channel_scores_per_window.py after every "
            "successful pair of good/bad inference jobs. The selected rows of "
            "node_scores are controlled by --per-channel-window-indices."
        ),
    )
    parser.add_argument(
        "--per-channel-window-indices",
        "--per_channel_window_indices",
        dest="per_channel_window_indices",
        default=None,
        help=(
            "Zero-based window indices/ranges for "
            "--per-channel-per-window-plot, e.g. '1200', '1,4,8', or "
            "'1200-1210'. Ranges are inclusive. If omitted, the plotter's "
            "DEFAULT_WINDOW_SPEC variable is used."
        ),
    )
    parser.add_argument(
        "--per-channel-window-datasets",
        "--per_channel_window_datasets",
        dest="per_channel_window_datasets",
        choices=("good", "bad", "both"),
        default=DEFAULT_PER_CHANNEL_WINDOW_DATASETS,
        help=(
            "Score file(s) used by --per-channel-per-window-plot. "
            "Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--per-channel-per-window-plot-name",
        "--per_channel_per_window_plot_name",
        dest="per_channel_per_window_plot_name",
        default=DEFAULT_PER_CHANNEL_PER_WINDOW_PLOT_NAME,
        help=(
            "Selected-window plot filename inside each inference_result "
            "directory. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--missing-plot",
        "--missing_plot",
        dest="missing_plot",
        action="store_true",
        help=(
            "Plot-only repair mode. For each model directory under --runs-root, "
            "independently check whether the all-window per-channel plot, the "
            "six selected-window per-plane boxplots, and goodvsbad.png exist. "
            "Recreate any missing plot set from scores_good.npz and "
            "scores_bad.npz. No training or inference is run."
        ),
    )
    parser.add_argument(
        "--force-replot",
        "--force_replot",
        dest="force_replot",
        action="store_true",
        help=(
            "Standalone plot-only mode. For each model directory under --runs-root, "
            "rerun the all-window per-channel plot, all six selected-window "
            "per-plane boxplots, and good-vs-bad plot whenever scores_good.npz and "
            "scores_bad.npz exist, even if the plots already exist. No training or "
            "inference is run."
        ),
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the sweep after the first failed config.",
    )
    parser.add_argument(
        "--monitor-interval",
        "--monitor_interval",
        dest="monitor_interval",
        type=int,
        default=30,
        help="Print CPU/RAM usage every N seconds during train/infer. Set 0 to disable.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help=(
            "Run in batch/log-file mode. Sets SBN_BATCH=1 and TQDM_DISABLE=1 "
            "for train/infer subprocesses so progress bars can be suppressed."
        ),
    )
    parser.add_argument(
        "--force-rewrite",
        "--force_rewrite",
        dest="force_rewrite",
        action="store_true",
        help=(
            "If a config/run name already exists in the database, delete the old "
            "database record and rerun training/inference, overwriting files in that run directory."
        ),
    )
    parser.add_argument(
        "--missing-rewrite",
        "--missing_rewrite",
        dest="missing_rewrite",
        action="store_true",
        help=(
            "If a config/run name already exists in the database, check whether "
            "checkpoints/graph_vae/<config_stem>/ exists and contains more than just one YAML file. "
            "If incomplete, delete the old database record and rerun."
        ),
    )
    parser.add_argument(
        "--restore-missing-db",
        "--restore_missing_db",
        "--restore_missing-db",
        "--restore_missing_db",
        "--restore-missing-database",
        "--restore_missing_database",
        "--restore-missing_database",
        "--restore_missing-database",
        dest="restore_missing_db",
        action="store_true",
        help=(
            "Restore completed model directories that are missing from the SQLite "
            "database using files already on disk. No training, inference, evaluation, "
            "or plotting is run. Ignored model directories are still eligible for "
            "restoration. Afterward, experiment IDs are renumbered to follow the "
            "sorted YAML config sweep order."
        ),
    )
    parser.add_argument(
        "--export-summary",
        "--export_summary",
        "--export-csv-xlsx",
        "--export_csv_xlsx",
        dest="export_summary",
        action="store_true",
        help=(
            "Write graph_vae_sweep_summary.csv and graph_vae_sweep_summary.xlsx after the sweep. "
            "By default, the SQLite database is updated but CSV/XLSX summary files are not produced."
        ),
    )

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run_sweep(args)


if __name__ == "__main__":
    sys.exit(main())
