"""Plotting helpers used only by timed-window inference.

The existing functions in ``sbn_anomaly.utils.plotting`` remain unchanged.
This module is imported only when ``data.timed_window`` is true.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def save_timed_score_over_time_plot(
    scores: Sequence[float] | np.ndarray,
    max_scores: Sequence[float] | np.ndarray,
    elapsed_time: Sequence[float] | np.ndarray,
    output_dir: str | Path,
    filename: str = "score_over_time_timed.png",
    threshold: float | None = None,
    title: str | None = None,
    time_unit: str = "minutes",
) -> Path:
    """Plot per-window scores against elapsed stitched time.

    Parameters
    ----------
    scores:
        Aggregated anomaly score for every window.
    max_scores:
        Maximum node reconstruction error for every window.
    elapsed_time:
        Elapsed window-start time in ``time_unit``.
    output_dir:
        Directory in which the PNG is written.
    filename:
        Output PNG filename.
    threshold:
        Optional anomaly threshold shown as a horizontal dashed line.
    title:
        Optional figure title.
    time_unit:
        Label for the elapsed-time values, normally seconds, minutes, or hours.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    scores_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    max_scores_arr = np.asarray(max_scores, dtype=np.float64).reshape(-1)
    elapsed_arr = np.asarray(elapsed_time, dtype=np.float64).reshape(-1)

    if scores_arr.size != max_scores_arr.size:
        raise ValueError(
            "scores and max_scores must have the same length: "
            f"{scores_arr.size} != {max_scores_arr.size}"
        )
    if scores_arr.size != elapsed_arr.size:
        raise ValueError(
            "scores and elapsed_time must have the same length: "
            f"{scores_arr.size} != {elapsed_arr.size}"
        )
    if elapsed_arr.size and not np.all(np.isfinite(elapsed_arr)):
        raise ValueError("elapsed_time contains NaN or infinite values")
    if elapsed_arr.size > 1 and np.any(np.diff(elapsed_arr) < 0):
        raise ValueError("elapsed_time must be nondecreasing")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(elapsed_arr, scores_arr, label="Window score")
    ax.plot(elapsed_arr, max_scores_arr, label="Maximum node score")

    if threshold is not None:
        threshold_value = float(threshold)
        if np.isfinite(threshold_value):
            ax.axhline(
                threshold_value,
                linestyle="--",
                label="Threshold",
            )

    unit = str(time_unit).strip().lower() or "time units"
    ax.set_xlabel(f"Elapsed stitched time [{unit}]")
    ax.set_ylabel("Anomaly score")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
