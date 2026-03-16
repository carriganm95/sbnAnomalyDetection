"""Anomaly detection metrics.

Provides helper functions for evaluating model performance when ground-truth
labels are available, and for computing summary statistics on raw scores.
"""

from __future__ import annotations

import numpy as np


def compute_roc_auc(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC AUC and the full ROC curve.

    Parameters
    ----------
    scores:
        1-D array of anomaly scores (higher = more anomalous).
    labels:
        1-D binary array (1 = anomaly, 0 = normal).

    Returns
    -------
    auc:
        Area under the ROC curve.
    fpr, tpr:
        False-positive and true-positive rates at each threshold.
    thresholds:
        Corresponding score thresholds.
    """
    try:
        from sklearn.metrics import roc_auc_score, roc_curve
    except ImportError as exc:
        raise ImportError("scikit-learn is required for ROC metrics.") from exc

    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc = float(roc_auc_score(labels, scores))
    return auc, fpr, tpr, thresholds


def anomaly_score_stats(scores: np.ndarray) -> dict[str, float]:
    """Return descriptive statistics for an array of anomaly scores.

    Parameters
    ----------
    scores:
        1-D array of anomaly scores.

    Returns
    -------
    stats:
        Dictionary with keys ``mean``, ``std``, ``min``, ``max``, ``p50``,
        ``p95``, ``p99``.
    """
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "p50": float(np.percentile(scores, 50)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
    }


def find_threshold_at_fpr(
    scores: np.ndarray,
    labels: np.ndarray,
    target_fpr: float = 0.01,
) -> float:
    """Return the score threshold that achieves ``target_fpr`` on labelled data.

    Parameters
    ----------
    scores:
        Anomaly scores.
    labels:
        Binary ground-truth labels.
    target_fpr:
        Desired false-positive rate (default 1 %).

    Returns
    -------
    threshold:
        Score cutoff closest to ``target_fpr``.
    """
    _, fpr, _, thresholds = compute_roc_auc(scores, labels)
    # Find the threshold where FPR is closest to target.
    idx = int(np.argmin(np.abs(fpr - target_fpr)))
    return float(thresholds[idx])
