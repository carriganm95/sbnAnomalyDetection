"""Plotting utilities.

This module is intentionally lightweight and headless-safe (uses the Agg
backend) so plots can be generated during CLI runs and in CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import matplotlib.colors as colors

def _as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _resolve_hist2d_spec(
    spec: Any | None,
    idx: int,
    feature_names: Sequence[str] | None,
) -> dict[str, Any]:
    """Return a matplotlib.hist2d-compatible spec for one feature.

    Supported top-level forms:
    - None: use the historical default bins and auto bounds
    - int: same bin count for all features
    - sequence: per-feature bin counts by index
        - dict: per-feature spec by name or index; each value may be an int or a
            dict containing `bins`, `range`, `underflow`, and/or `overflow`
    """

    def _as_positive_int(value: Any, default: int = 10) -> int:
        try:
            int_value = int(value)
        except Exception:
            return default
        return int_value if int_value > 0 else default

    def _normalize_range(value_range: Any) -> list[list[float]] | None:
        if not isinstance(value_range, (list, tuple)) or len(value_range) != 2:
            return None
        try:
            low = float(value_range[0])
            high = float(value_range[1])
        except Exception:
            return None
        return [[low, high], [low, high]]

    default_spec: dict[str, Any] = {"bins": 10}
    if spec is None:
        return default_spec

    feature_spec: Any = spec
    if isinstance(spec, dict):
        # Allow explicit per-feature overrides by flattened index.
        if idx in spec:
            feature_spec = spec[idx]
        elif str(idx) in spec:
            feature_spec = spec[str(idx)]
        elif feature_names and len(feature_names) > 0:
            # Hit-mode features are laid out as repeated blocks of variables.
            # Map flattened index back to variable index so one feature-name
            # setting applies consistently across all hits.
            feature_name = feature_names[idx % len(feature_names)]
            if feature_name in spec:
                feature_spec = spec[feature_name]
            else:
                feature_spec = None
        else:
            feature_spec = None

    if feature_spec is None:
        return default_spec

    if isinstance(feature_spec, int):
        return {"bins": _as_positive_int(feature_spec)}

    if isinstance(feature_spec, (list, tuple)):
        if idx < len(feature_spec):
            return {"bins": _as_positive_int(feature_spec[idx])}
        return default_spec

    if isinstance(feature_spec, dict):
        out: dict[str, Any] = {"bins": _as_positive_int(feature_spec.get("bins", 10))}
        normalized_range = _normalize_range(feature_spec.get("range"))
        if normalized_range is not None:
            out["range"] = normalized_range
        if feature_spec.get("underflow") is not None:
            try:
                out["underflow"] = float(feature_spec["underflow"])
            except Exception:
                pass
        if feature_spec.get("overflow") is not None:
            try:
                out["overflow"] = float(feature_spec["overflow"])
            except Exception:
                pass
        return out

    return default_spec


def save_training_curves(
    history: Mapping[str, Sequence[Any]],
    output_dir: str | Path,
    filename: str = "training_curves.png",
) -> Path:
    """Save a PNG with loss and (optional) metric curves.

        Expected keys in *history* (all optional except loss):
      - epoch (1-based ints)
      - loss (floats)
      - precision, recall, f1, auc (floats)
            - score_p95, score_p99, anomaly_fraction_above_threshold
            - epoch_time_sec, events_per_sec

    Returns the written PNG path.
    """

    # Headless-safe backend.
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loss = np.asarray(_as_list(history.get("loss")), dtype=float)
    if loss.size == 0:
        # No loss history available; nothing to plot. Return without error
        # so callers can continue with other artifact generation.
        return out_dir / filename

    epochs_raw = _as_list(history.get("epoch"))
    if epochs_raw:
        epochs = np.asarray(epochs_raw, dtype=int)
    else:
        epochs = np.arange(1, loss.size + 1, dtype=int)

    # Pad/truncate to match loss length.
    n = int(loss.size)
    if epochs.size != n:
        epochs = np.arange(1, n + 1, dtype=int)

    def _metric(name: str) -> np.ndarray:
        arr = np.asarray(_as_list(history.get(name)), dtype=float)
        if arr.size == 0:
            return arr
        if arr.size < n:
            arr = np.pad(arr, (0, n - arr.size), constant_values=np.nan)
        if arr.size > n:
            arr = arr[:n]
        # Replace all-NaN arrays with an empty array to indicate 'no data'.
        if not np.isfinite(arr).any():
            return np.asarray([], dtype=float)
        # Mask non-finite values to NaN so plotting functions can ignore them.
        arr = np.where(np.isfinite(arr), arr, np.nan)
        return arr

    precision = _metric("precision")
    recall = _metric("recall")
    f1 = _metric("f1")
    auc = _metric("auc")
    val_loss = _metric("val_loss")
    score_p95 = _metric("score_p95")
    score_p99 = _metric("score_p99")
    anomaly_frac = _metric("anomaly_fraction_above_threshold")
    epoch_time_sec = _metric("epoch_time_sec")
    events_per_sec = _metric("events_per_sec")

    has_cls_metrics = any(x.size for x in (precision, recall, f1, auc))
    has_score_metrics = any(x.size for x in (score_p95, score_p99, anomaly_frac))
    has_perf_metrics = any(x.size for x in (epoch_time_sec, events_per_sec))

    n_panels = 1 + int(has_score_metrics or has_cls_metrics) + int(has_perf_metrics)

    fig, axes = plt.subplots(
        n_panels,
        1,
        sharex=True,
        figsize=(9, 3 * n_panels + 1),
    )
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]

    ax0 = axes[0]
    ax0.plot(epochs, loss, label="loss")
    if val_loss.size:
        ax0.plot(epochs, val_loss, label="val_loss")
        ax0.legend(loc="best")
    ax0.set_ylabel("Loss")
    ax0.grid(True, alpha=0.3)

    panel_idx = 1
    if has_score_metrics or has_cls_metrics:
        ax1 = axes[panel_idx]
        plotted_any = False
        def _maybe_plot(arr: np.ndarray, label: str) -> None:
            nonlocal plotted_any
            if arr.size:
                ax1.plot(epochs, arr, label=label)
                plotted_any = True

        _maybe_plot(score_p95, "score_p95")
        _maybe_plot(score_p99, "score_p99")
        _maybe_plot(anomaly_frac, "anomaly_frac")
        _maybe_plot(precision, "precision")
        _maybe_plot(recall, "recall")
        _maybe_plot(f1, "f1")
        _maybe_plot(auc, "auc")

        ax1.set_ylabel("Scores / Metrics")
        ax1.grid(True, alpha=0.3)
        if plotted_any:
            ax1.legend(loc="best")
            # Only use log scale when there are positive, finite values.
            try:
                combined = np.concatenate([arr for arr in (score_p95, score_p99, anomaly_frac, precision, recall, f1, auc) if arr.size])
                if np.isfinite(combined).any() and (combined > 0).any():
                    ax1.set_yscale("log")
            except Exception:
                # Fall back to linear scale on any unexpected issue.
                pass
        panel_idx += 1

    if has_perf_metrics:
        axp = axes[panel_idx]
        if epoch_time_sec.size:
            axp.plot(epochs, epoch_time_sec, label="epoch_time_sec")
        if events_per_sec.size:
            axp.plot(epochs, events_per_sec, label="events_per_sec")
        axp.set_ylabel("Performance")
        axp.grid(True, alpha=0.3)
        axp.legend(loc="best")

    axes[-1].set_xlabel("Epoch")

    fig.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _to_2d_matrix(data: Sequence[Any]) -> "np.ndarray":
    """Convert sequence of scalars/vectors/matrices into a 2D float array."""
    import numpy as np

    chunks = list(data)
    if not chunks:
        return np.zeros((0, 0), dtype=float)

    rows = []
    for item in chunks:
        if hasattr(item, "detach"):
            array = item.detach().to("cpu").numpy()
        else:
            array = np.asarray(item)
        array = np.asarray(array, dtype=float)
        if array.ndim == 0:
            array = array.reshape(1, 1)
        elif array.ndim == 1:
            array = array.reshape(1, -1)
        else:
            array = array.reshape(array.shape[0], -1)
        rows.append(array)

    return np.concatenate(rows, axis=0)


def _feature_label(
    feature_idx: int,
    feature_names: Sequence[str] | None,
    n_variables: int | None,
) -> str:
    """Return a descriptive feature label including variable/hit when possible."""
    if n_variables is not None and n_variables > 0:
        var_idx = feature_idx % n_variables
        hit_idx = feature_idx // n_variables
        if feature_names and var_idx < len(feature_names):
            var_name = str(feature_names[var_idx])
        else:
            var_name = f"var_{var_idx}"
        return f"feature {feature_idx} ({var_name}, hit {hit_idx})"
    return f"feature {feature_idx}"


def _draw_feature_hist2d(
    x_feature: "np.ndarray",
    y_feature: "np.ndarray",
    path: Path,
    title: str,
    bins: int | None = None,
    value_range: list[list[float]] | None = None,
    underflow: float | None = None,
    overflow: float | None = None,
) -> None:
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 5))
    # Allow callers to override binning per-feature. Fall back to the
    # historical default of 10 bins when unspecified or invalid.
    bins_arg = int(bins) if (bins is not None and int(bins) > 0) else 10

    # Determine histogram range BEFORE clipping or drawing.
    # This ensures both x and y axes use identical bins.
    if value_range is not None:
        # Explicit range from config takes precedence
        hist_range = value_range
        lo, hi = value_range[0][0], value_range[0][1]
    elif underflow is not None or overflow is not None:
        # Use underflow/overflow to define bounds
        lo = underflow if underflow is not None else float(np.nanmin([np.min(x_feature), np.min(y_feature)]))
        hi = overflow if overflow is not None else float(np.nanmax([np.max(x_feature), np.max(y_feature)]))
        hist_range = [[lo, hi], [lo, hi]]
    else:
        # Auto-compute bounds from data
        lo = min(float(np.min(x_feature)), float(np.min(y_feature)))
        hi = max(float(np.max(x_feature)), float(np.max(y_feature)))
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo, hi = -1.0, 1.0
        elif hi <= lo:
            delta = 1.0 if lo == 0.0 else max(abs(lo) * 0.05, 1e-6)
            lo -= delta
            hi += delta
        hist_range = [[lo, hi], [lo, hi]]

    # Clip data to bounds if underflow/overflow are configured
    if underflow is not None or overflow is not None:
        x_feature = np.clip(x_feature, lo, hi)
        y_feature = np.clip(y_feature, lo, hi)

    hist_kwargs: dict[str, Any] = {
        "bins": bins_arg,
        "norm": colors.LogNorm(vmin=1),
        "cmap": "Blues",
        "range": hist_range,
    }

    h = ax.hist2d(x_feature, y_feature, **hist_kwargs)
    fig.colorbar(h[3], ax=ax, label="count")

    # Force identical axis bounds so the y=x diagonal is geometrically correct.
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y=x")

    ax.set_xlabel("original value")
    ax.set_ylabel("reconstructed value")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_reconstruction_hist2d(
    original: Sequence[Any],
    reconstruction: Sequence[Any],
    output_dir: str | Path,
    filename: str = "reconstruction_hist2d.png",
    feature_names: Sequence[str] | None = None,
    n_variables: int | None = None,
    bins: Any | None = None,
) -> Path:
    """Save per-feature 2D histograms of original vs reconstructed values."""
    import matplotlib

    matplotlib.use("Agg", force=True)

    x = _to_2d_matrix(original)
    y = _to_2d_matrix(reconstruction)
    if x.size == 0 or y.size == 0:
        raise ValueError("original and reconstruction inputs must be non-empty")

    n_rows = min(x.shape[0], y.shape[0])
    n_features = min(x.shape[1], y.shape[1])
    x = x[:n_rows, :n_features]
    y = y[:n_rows, :n_features]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(filename).stem
    suffix = Path(filename).suffix or ".png"

    first_path = out_dir / filename
    for feature_idx in range(n_features):
        if feature_idx == 0:
            out_path = first_path
        else:
            out_path = out_dir / f"{stem}_feature_{feature_idx:03d}{suffix}"

        feature_label = _feature_label(feature_idx, feature_names, n_variables)
        title = f"Original vs Reconstruction - {feature_label}"
        feature_hist_spec = _resolve_hist2d_spec(bins, feature_idx, feature_names)
        _draw_feature_hist2d(
            x[:, feature_idx],
            y[:, feature_idx],
            out_path,
            title,
            bins=feature_hist_spec.get("bins"),
            value_range=feature_hist_spec.get("range"),
            underflow=feature_hist_spec.get("underflow"),
            overflow=feature_hist_spec.get("overflow"),
        )

    return first_path


def save_epoch_score_hist2d(
    original: Sequence[Any],
    reconstruction: Sequence[Any],
    output_dir: str | Path,
    epoch: int,
    best: bool = False,
    feature_names: Sequence[str] | None = None,
    n_variables: int | None = None,
    bins: Any | None = None,
) -> Path:
    """Save per-epoch per-feature 2D histograms of original vs reconstructed values."""
    import matplotlib

    matplotlib.use("Agg", force=True)

    x = _to_2d_matrix(original)
    y = _to_2d_matrix(reconstruction)
    if x.size == 0 or y.size == 0:
        raise ValueError("original and reconstruction inputs must be non-empty")

    n_rows = min(x.shape[0], y.shape[0])
    n_features = min(x.shape[1], y.shape[1])
    x = x[:n_rows, :n_features]
    y = y[:n_rows, :n_features]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_str = "_best" if best else ""
    base_name = f"epoch_{epoch:04d}{best_str}_hist2d.png"
    stem = Path(base_name).stem
    suffix = Path(base_name).suffix or ".png"

    first_path = out_dir / base_name
    epoch_label = f"Epoch {epoch}" + (" (BEST)" if best else "")

    for feature_idx in range(n_features):
        if feature_idx == 0:
            out_path = first_path
        else:
            out_path = out_dir / f"{stem}_feature_{feature_idx:03d}{suffix}"

        feature_label = _feature_label(feature_idx, feature_names, n_variables)
        title = f"Original vs Reconstruction - {epoch_label} - {feature_label}"
        feature_hist_spec = _resolve_hist2d_spec(bins, feature_idx, feature_names)
        _draw_feature_hist2d(
            x[:, feature_idx],
            y[:, feature_idx],
            out_path,
            title,
            bins=feature_hist_spec.get("bins"),
            value_range=feature_hist_spec.get("range"),
            underflow=feature_hist_spec.get("underflow"),
            overflow=feature_hist_spec.get("overflow"),
        )

    return first_path
