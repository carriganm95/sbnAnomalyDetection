import os
os.environ["MPLBACKEND"] = "Agg"

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# Input / output
# ============================================================

npz_path = "/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection/checkpoints/gnn/0018_win100_stride100_hist5_bs64_lr0p003/inference_scores.npz"
out_dir = "/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection/inference_result_plots/0018_win100_stride100_hist5_bs64_lr0p003"
os.makedirs(out_dir, exist_ok=True)

# ============================================================
# Plot settings
# ============================================================

bins = 70

# x-axis cutoff. Only show up to this percentile of the selected runs.
# The visible cutoff is still based on this percentile, but the x-axis
# is normalized by the true maximum value in the whole array, not by this percentile.
percentile = 60

# X-axis transform mode.
#
# Options:
#   "none"       : keep original raw score scale
#   "global_max" : x / max(x) over the whole finite array
#   "tanh"       : tanh(x / scale), then normalized so the whole-data max maps to 1
#   "sigmoid"    : sigmoid(x / scale), shifted so 0 maps to 0, then normalized so
#                  the whole-data max maps to 1
#
# For your current case, "tanh" or "sigmoid" is better than "global_max", because
# the true max is a huge outlier and global_max squeezes ordinary values to x ~ 0.
x_axis_transform = "none"

# Scale for tanh/sigmoid x-axis transforms, in raw score units.
# If None, the script uses the current visible cutoff x_max, i.e. the selected
# percentile cutoff. This does NOT normalize the cutoff to 1; it only controls
# how strongly tanh/sigmoid stretch the low-score region.
#
# Smaller scale  -> stronger stretching of small scores
# Larger scale   -> closer to linear/global-max behavior
x_transform_scale = None

# Do NOT use density for this mode.
# Instead, each bin is normalized by the total number of windows in that run.
normalize_to_windows = True

# Variable binning: more bins near low scores, fewer bins in the high-score tail.
binning_mode = "variable"   # options: "linear", "variable"

# For variable binning:
# [0, tail_start_percentile] gets many bins;
# [tail_start_percentile, percentile] gets fewer bins.
tail_start_percentile = 60
low_score_bin_fraction = 0.80   # the percentage of bins used in the low-score peak region

# ============================================================
# Good / bad classification
# Same as C++ ROOT plotter
# ============================================================

good_runs = {
    18445, 19724, 20141, 20142, 20144
}

bad_runs = {
    19627, 19946, 20104
}

# ============================================================
# Optional run selection
# ============================================================

# If True, ignore runs_to_plot and plot every run found in the NPZ,
# including runs not listed in good_runs or bad_runs.
plot_all_runs = True

# If plot_all_runs=False, only these runs are plotted.
# Missing runs are skipped with a warning.
runs_to_plot = [
    18445,
    19627,
    19724,
    19946,
    20104,
    20141,
    20142,
    20144,
]

# Apply switch
if plot_all_runs:
    runs_to_plot = None

# Use this instead if you want all runs:
# runs_to_plot = None


# ============================================================
# Unknown-run handling
# ============================================================

# Controls how runs not listed in good_runs or bad_runs are displayed.
#
# Options:
#   "unknown" : keep unknown runs labeled/colored as Unknown
#   "good"    : treat unknown runs as Good for labels/colors
#   "bad"     : treat unknown runs as Bad for labels/colors
unknown_run_default = "good"

allowed_unknown_run_defaults = {"unknown", "good", "bad"}
if unknown_run_default not in allowed_unknown_run_defaults:
    raise ValueError(
        f"Bad unknown_run_default={unknown_run_default!r}. "
        f"Use one of {sorted(allowed_unknown_run_defaults)}."
    )


# ============================================================
# Color palettes
# Good runs use cold colors.
# Bad runs use hot colors.
# Unknown runs use sequential colors.
# ============================================================

good_colors = [
    "#2483c8",  # blue
    "#063d6b",  # medium blue
    "#17becf",  # cyan
    "#00a087",  # teal
    "#2ca02c",  # green
    "#4daf4a",  # medium green
    "#66c2a5",  # pale teal
]

bad_colors = [
    "#d62728",  # red
    "#e41a1c",  # bright red
    "#b2182b",  # dark red
    "#ff7f0e",  # orange
    "#a65628",  # brown-orange
    "#613807",  # dark brown-orange
]

unknown_colors = [
    "#7f7f7f",  # gray
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#bcbd22",  # olive
    "#1f77b4",  # blue
    "#ff9896",  # light red
    "#c5b0d5",  # light purple
    "#c49c94",  # light brown
    "#dbdb8d",  # light olive
]


# ============================================================
# Helpers
# ============================================================

def get_effective_run_class(run):
    """Return the display class for one run: 'good', 'bad', or 'unknown'."""
    if run in good_runs:
        return "good"

    if run in bad_runs:
        return "bad"

    return unknown_run_default


def get_run_color(run, good_idx, bad_idx, unknown_idx):
    run_class = get_effective_run_class(run)

    if run_class == "good":
        color = good_colors[good_idx % len(good_colors)]
        good_idx += 1
        return color, good_idx, bad_idx, unknown_idx

    if run_class == "bad":
        color = bad_colors[bad_idx % len(bad_colors)]
        bad_idx += 1
        return color, good_idx, bad_idx, unknown_idx

    if run_class == "unknown":
        color = unknown_colors[unknown_idx % len(unknown_colors)]
        unknown_idx += 1
        return color, good_idx, bad_idx, unknown_idx

    raise ValueError(
        f"Bad effective run class {run_class!r} for run {run}. "
        "Expected 'good', 'bad', or 'unknown'."
    )


def get_run_label(run):
    if run in good_runs:
        return f"Run {run} (Good)"

    if run in bad_runs:
        return f"Run {run} (Bad)"

    if unknown_run_default == "good":
        return f"Run {run} (Good, defaulted from Unknown)"

    if unknown_run_default == "bad":
        return f"Run {run} (Bad, defaulted from Unknown)"

    if unknown_run_default == "unknown":
        return f"Run {run} (Unknown)"

    raise ValueError(
        f"Bad unknown_run_default={unknown_run_default!r}. "
        "Use 'unknown', 'good', or 'bad'."
    )


def resolve_runs_to_plot(all_runs, requested_runs):
    """
    Decide which runs to plot.

    Parameters
    ----------
    all_runs : array-like
        Runs actually present in the NPZ file.
    requested_runs : list[int] or None
        User-selected runs. If None or empty, plot all runs.

    Returns
    -------
    selected_runs : list[int]
        Runs that exist and should be plotted.
    """
    all_runs_set = set(int(r) for r in all_runs)

    if requested_runs is None or len(requested_runs) == 0:
        return sorted(all_runs_set)

    selected_runs = []
    for run in requested_runs:
        run = int(run)
        if run in all_runs_set:
            selected_runs.append(run)
        else:
            print(f"WARNING: requested run {run} does not exist in this NPZ file; skipping.")

    if len(selected_runs) == 0:
        print("WARNING: none of the requested runs exist in this NPZ file.")

    return selected_runs



def _stable_sigmoid(z):
    """Numerically stable sigmoid for numpy arrays/scalars."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)

    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))

    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)

    return out


def transform_x_axis(raw_x, mode, global_x_max, transform_scale):
    """
    Transform raw score values for display on the x-axis.

    The histogram counts are still computed in raw score space. This function
    only changes the displayed x coordinate of the bin edges.
    """
    raw_x = np.asarray(raw_x, dtype=float)
    mode = str(mode).lower()

    if mode == "none":
        return raw_x

    if not np.isfinite(global_x_max) or global_x_max <= 0.0:
        raise ValueError(f"Bad global_x_max={global_x_max}; cannot transform x-axis.")

    if mode == "global_max":
        return raw_x / global_x_max

    if transform_scale is None:
        raise ValueError("transform_scale must be set for tanh/sigmoid transforms.")

    if not np.isfinite(transform_scale) or transform_scale <= 0.0:
        raise ValueError(f"Bad transform_scale={transform_scale}; cannot transform x-axis.")

    if mode == "tanh":
        denominator = np.tanh(global_x_max / transform_scale)
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError(f"Bad tanh denominator={denominator}; cannot transform x-axis.")
        return np.tanh(raw_x / transform_scale) / denominator

    if mode == "sigmoid":
        # Shift so x=0 maps to 0 instead of 0.5.
        numerator = _stable_sigmoid(raw_x / transform_scale) - 0.5
        denominator = _stable_sigmoid(global_x_max / transform_scale) - 0.5
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError(f"Bad sigmoid denominator={denominator}; cannot transform x-axis.")
        return numerator / denominator

    raise ValueError(
        f"Unknown x_axis_transform={mode!r}. Use 'none', 'global_max', 'tanh', or 'sigmoid'."
    )


def make_bin_edges(
    selected_values,
    bins=80,
    percentile=99.5,
    binning_mode="linear",
    tail_start_percentile=70,
    low_score_bin_fraction=0.75,
):
    """
    Make common bin edges for all runs.

    linear:
        Uniform bin width from 0 to x_max.

    variable:
        More bins in the low-score region, fewer bins in the tail.
        This helps resolve the peak without using too many total bins.
    """

    selected_values = np.asarray(selected_values)
    selected_values = selected_values[np.isfinite(selected_values)]

    if selected_values.size == 0:
        return None, None, None

    x_min = 0.0
    x_max = np.percentile(selected_values, percentile)

    if not np.isfinite(x_max) or x_max <= x_min:
        print(f"WARNING: bad x_max={x_max}; using max selected value instead.")
        x_max = np.nanmax(selected_values)

    if not np.isfinite(x_max) or x_max <= x_min:
        return None, None, None

    if binning_mode == "linear":
        bin_edges = np.linspace(x_min, x_max, bins + 1)
        return bin_edges, x_min, x_max

    if binning_mode != "variable":
        raise ValueError(f"Unknown binning_mode={binning_mode!r}. Use 'linear' or 'variable'.")

    # Boundary between dense low-score bins and coarse tail bins
    x_split = np.percentile(selected_values, tail_start_percentile)

    # Make sure x_split is useful
    if not np.isfinite(x_split) or x_split <= x_min or x_split >= x_max:
        print(
            f"WARNING: bad x_split={x_split}; falling back to linear binning."
        )
        bin_edges = np.linspace(x_min, x_max, bins + 1)
        return bin_edges, x_min, x_max

    low_bins = int(round(bins * low_score_bin_fraction))
    tail_bins = bins - low_bins

    # Avoid degenerate cases
    low_bins = max(low_bins, 1)
    tail_bins = max(tail_bins, 1)

    low_edges = np.linspace(x_min, x_split, low_bins + 1)
    tail_edges = np.linspace(x_split, x_max, tail_bins + 1)

    # Combine, removing duplicate x_split
    bin_edges = np.concatenate([low_edges, tail_edges[1:]])

    # Remove accidental duplicate edges
    bin_edges = np.unique(bin_edges)

    if bin_edges.size < 2:
        return None, None, None

    return bin_edges, x_min, x_max


def plot_hist_by_run(
    values,
    runs,
    xlabel,
    title,
    output_path,
    bins=80,
    percentile=99.5,
    runs_to_plot=None,
    normalize_to_windows=True,
    binning_mode="variable",
    tail_start_percentile=70,
    low_score_bin_fraction=0.75,
    x_axis_transform="none",
    x_transform_scale=None,
):
    """
    Plot per-run histograms as ROOT-style line histograms.

    If normalize_to_windows=True:
        y = bin_count / total_number_of_windows_for_that run

    This makes runs with many windows and few windows comparable in shape.

    This debug version prints progress messages so you can see whether the script
    is slow during:
      - finite filtering
      - percentile/bin-edge creation
      - histogramming each run
      - tight_layout / savefig
    """

    print("\n" + "=" * 80, flush=True)
    print(f"Entered plot_hist_by_run: {title}", flush=True)
    print("=" * 80, flush=True)

    # Convert first, so shapes/finite checks are reliable.
    values = np.asarray(values)
    runs = np.asarray(runs)

    print(f"Initial values.shape = {values.shape}", flush=True)
    print(f"Initial runs.shape   = {runs.shape}", flush=True)

    # Keep only finite entries
    print("Making finite mask...", flush=True)
    finite_mask = np.isfinite(values) & np.isfinite(runs)
    print("Finished finite mask.", flush=True)

    values = values[finite_mask]
    runs = runs[finite_mask].astype(int)

    print(f"After finite filtering: values.size = {values.size}", flush=True)

    if values.size == 0:
        print(f"No valid values found for {title}", flush=True)
        return

    # True maximum over the whole finite array, not the visible percentile.
    # Used by global_max/tanh/sigmoid transforms so the whole-data maximum maps to 1.
    print("Computing global_x_max...", flush=True)
    global_x_max = np.nanmax(values)
    print(f"global_x_max = {global_x_max}", flush=True)

    x_axis_transform = str(x_axis_transform).lower()

    if x_axis_transform != "none":
        if not np.isfinite(global_x_max) or global_x_max <= 0.0:
            print(
                f"WARNING: cannot transform x-axis for {title}; "
                f"bad global_x_max={global_x_max}; skipping.",
                flush=True,
            )
            return

    print("Finding available runs...", flush=True)
    all_runs = sorted(np.unique(runs))
    print(f"all_runs = {all_runs}", flush=True)

    selected_runs = resolve_runs_to_plot(all_runs, runs_to_plot)
    print(f"selected_runs = {selected_runs}", flush=True)

    if len(selected_runs) == 0:
        print(f"No runs to plot for {title}", flush=True)
        return

    # Use selected runs only when computing percentile x range
    print("Selecting values for selected runs...", flush=True)
    selected_mask = np.isin(runs, selected_runs)
    selected_values = values[selected_mask]
    print(f"selected_values.size = {selected_values.size}", flush=True)

    if selected_values.size == 0:
        print(f"No values found for selected runs in {title}", flush=True)
        return

    print("Making bin edges...", flush=True)
    bin_edges, x_min, x_max = make_bin_edges(
        selected_values=selected_values,
        bins=bins,
        percentile=percentile,
        binning_mode=binning_mode,
        tail_start_percentile=tail_start_percentile,
        low_score_bin_fraction=low_score_bin_fraction,
    )
    print("Finished making bin edges.", flush=True)

    if bin_edges is None:
        print(f"WARNING: cannot determine valid bin edges for {title}; skipping.", flush=True)
        return

    print(f"x_min = {x_min}", flush=True)
    print(f"x_max = {x_max}", flush=True)
    print(f"number of bins = {len(bin_edges) - 1}", flush=True)

    # For tanh/sigmoid, None means use the visible raw cutoff as the transform scale.
    # This makes the visible percentile region readable even when the true max is huge.
    transform_scale_used = x_transform_scale
    if x_axis_transform in {"tanh", "sigmoid"} and transform_scale_used is None:
        transform_scale_used = x_max

    print(f"x_axis_transform = {x_axis_transform}", flush=True)
    print(f"transform_scale_used = {transform_scale_used}", flush=True)

    print("Computing transformed x-axis limits...", flush=True)
    try:
        x_min_plot = float(
            transform_x_axis(
                x_min,
                x_axis_transform,
                global_x_max,
                transform_scale_used,
            )
        )
        x_max_plot = float(
            transform_x_axis(
                global_x_max,
                x_axis_transform,
                global_x_max,
                transform_scale_used,
            )
        )
        x_visible_cutoff_plot = float(
            transform_x_axis(
                x_max,
                x_axis_transform,
                global_x_max,
                transform_scale_used,
            )
        )
    except ValueError as exc:
        print(f"WARNING: {exc}; skipping {title}.", flush=True)
        return

    print(f"x_min_plot = {x_min_plot}", flush=True)
    print(f"x_max_plot = {x_max_plot}", flush=True)
    print(f"x_visible_cutoff_plot = {x_visible_cutoff_plot}", flush=True)

    print("Creating figure...", flush=True)
    plt.figure(figsize=(12, 7))
    print("Created figure.", flush=True)

    good_idx = 0
    bad_idx = 0
    unknown_idx = 0

    plotted_any = False
    ylabel = "Fraction of windows" if normalize_to_windows else "Count"

    print("Starting per-run histogram loop...", flush=True)

    for run in selected_runs:
        print(f"  Plotting run {run}...", flush=True)

        mask = runs == run
        run_values_all = values[mask]
        n_total_windows = run_values_all.size

        print(f"    n_total_windows = {n_total_windows}", flush=True)

        if n_total_windows == 0:
            print(f"    Skipping run {run}: no windows.", flush=True)
            continue

        # Only histogram the visible range.
        run_values_visible = run_values_all[
            (run_values_all >= x_min) & (run_values_all <= x_max)
        ]

        print(f"    run_values_visible.size = {run_values_visible.size}", flush=True)

        if run_values_visible.size == 0:
            print(
                f"WARNING: run {run} exists, but has no entries within "
                f"[{x_min}, {x_max}] for {title}; skipping.",
                flush=True,
            )
            continue

        color, good_idx, bad_idx, unknown_idx = get_run_color(
            run,
            good_idx,
            bad_idx,
            unknown_idx,
        )

        print(f"    Computing histogram for run {run}...", flush=True)
        raw_counts, edges = np.histogram(
            run_values_visible,
            bins=bin_edges,
            density=False,
        )
        print(f"    Finished histogram for run {run}.", flush=True)

        if normalize_to_windows:
            # Normalize by total number of windows in this run, not just visible windows.
            # This means bins above x_max are simply not shown, but the normalization
            # still represents fraction of the full run.
            y_values = raw_counts / n_total_windows
        else:
            y_values = raw_counts

        label = (
            f"{get_run_label(run)}  "
            f"(N={n_total_windows}, shown={run_values_visible.size})"
        )

        # For a true step histogram, x has length nbins+1 and y is extended by one.
        y_step = np.r_[y_values, y_values[-1]]

        print(f"    Transforming bin edges for run {run}...", flush=True)
        edges_to_plot = transform_x_axis(
            edges,
            mode=x_axis_transform,
            global_x_max=global_x_max,
            transform_scale=transform_scale_used,
        )

        print(f"    Drawing step plot for run {run}...", flush=True)
        plt.step(
            edges_to_plot,
            y_step,
            where="post",
            color=color,
            linewidth=2,
            label=label,
        )

        print(f"  Finished plotting run {run}.", flush=True)
        plotted_any = True

    print("Finished per-run histogram loop.", flush=True)

    if not plotted_any:
        print(f"WARNING: no runs were actually plotted for {title}.", flush=True)
        plt.close()
        return

    print("Setting labels/title...", flush=True)

    if x_axis_transform == "none":
        plt.xlabel(xlabel)
    elif x_axis_transform == "global_max":
        plt.xlabel(f"Normalized {xlabel} ({xlabel} / whole-data max)")
    elif x_axis_transform == "tanh":
        plt.xlabel(f"Tanh-transformed {xlabel}")
    elif x_axis_transform == "sigmoid":
        plt.xlabel(f"Sigmoid-transformed {xlabel}")

    plt.ylabel(ylabel)

    if binning_mode == "variable":
        title_extra = (
            f"Shown up to {percentile}th percentile; "
            f"variable bins, split at {tail_start_percentile}th percentile"
        )
    else:
        title_extra = f"Shown up to {percentile}th percentile; linear bins"

    if x_axis_transform == "none":
        plt.title(f"{title}\n{title_extra} (raw x_max = {x_max:.4g})")
        plt.xlim(x_min, x_max)
    else:
        scale_text = ""
        if x_axis_transform in {"tanh", "sigmoid"}:
            scale_text = f", scale = {transform_scale_used:.4g}"

        plt.title(
            f"{title}\n"
            f"{title_extra}; x transform = {x_axis_transform}{scale_text} "
            f"(raw cutoff = {x_max:.4g}, transformed cutoff = {x_visible_cutoff_plot:.4g})"
        )
        plt.xlim(x_min_plot, x_max_plot)

    print("Adding legend...", flush=True)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)

    # Use subplots_adjust instead of tight_layout + bbox_inches='tight'.
    # This is usually much faster and avoids hanging on some network filesystems.
    print("Adjusting subplot layout...", flush=True)
    plt.subplots_adjust(right=0.72)

    # Make sure output directory exists.
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Avoid bbox_inches='tight' for debugging. It can be slow with outside legends.
    print(f"Saving figure to: {output_path}", flush=True)
    plt.savefig(output_path, dpi=100)
    print(f"Finished saving figure to: {output_path}", flush=True)

    plt.close()
    print("Closed figure.", flush=True)

    print(f"Saved: {output_path}", flush=True)
    print(f"  plotted runs = {selected_runs}", flush=True)
    print(f"  x_max ({percentile}th percentile of selected runs, raw scale) = {x_max}", flush=True)
    print(f"  global_x_max (whole finite array) = {global_x_max}", flush=True)
    print(f"  x_axis_transform = {x_axis_transform}", flush=True)

    if x_axis_transform in {"tanh", "sigmoid"}:
        print(f"  x_transform_scale used = {transform_scale_used}", flush=True)

    if x_axis_transform != "none":
        print(f"  transformed visible x cutoff = {x_visible_cutoff_plot}", flush=True)

    print(f"  binning_mode = {binning_mode}", flush=True)
    print(f"  number of bins = {len(bin_edges) - 1}", flush=True)
    print("=" * 80, flush=True)

# ============================================================
# Main
# ============================================================

data = np.load(npz_path, allow_pickle=True)

print("=" * 80)
print(f"Reading NPZ file: {npz_path}")
print("=" * 80)
print("\nKeys:")
for key in data.keys():
    print(f"  - {key}")

scores = data["scores"]
scores_max = data["scores_max"]
first_run = data["first_run"]

print("\nArray shapes:")
print(f"  scores.shape     = {scores.shape}")
print(f"  scores_max.shape = {scores_max.shape}")
print(f"  first_run.shape  = {first_run.shape}")

if not (scores.shape == scores_max.shape == first_run.shape):
    raise ValueError("scores, scores_max, and first_run must have the same shape")

available_runs = sorted(np.unique(first_run[np.isfinite(first_run)]).astype(int))
print("\nAvailable runs in NPZ:")
print(available_runs)
print(f"\nplot_all_runs = {plot_all_runs}")
print(f"unknown_run_default = {unknown_run_default!r}")

# ============================================================
# Histogram 1: scores
# ============================================================

plot_hist_by_run(
    values=scores,
    runs=first_run,
    xlabel="Score",
    title="Histogram of window scores by run",
    output_path=os.path.join(out_dir, "scores_hist_by_run.png"),
    bins=bins,
    percentile=percentile,
    runs_to_plot=runs_to_plot,
    normalize_to_windows=normalize_to_windows,
    binning_mode=binning_mode,
    tail_start_percentile=tail_start_percentile,
    low_score_bin_fraction=low_score_bin_fraction,
    x_axis_transform=x_axis_transform,
    x_transform_scale=x_transform_scale,
)

# ============================================================
# Histogram 2: scores_max
# ============================================================

plot_hist_by_run(
    values=scores_max,
    runs=first_run,
    xlabel="Score max",
    title="Histogram of window max scores by run",
    output_path=os.path.join(out_dir, "scores_max_hist_by_run.png"),
    bins=bins,
    percentile=percentile,
    runs_to_plot=runs_to_plot,
    normalize_to_windows=normalize_to_windows,
    binning_mode=binning_mode,
    tail_start_percentile=tail_start_percentile,
    low_score_bin_fraction=low_score_bin_fraction,
    x_axis_transform=x_axis_transform,
    x_transform_scale=x_transform_scale,
)