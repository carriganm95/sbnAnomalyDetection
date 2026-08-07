# Graphing utilities

This directory contains plotting and data-diagnostic scripts for the GraphVAE
workflow. Some scripts plot saved GraphVAE inference scores, while others inspect
the sparse event files or ROOT waveforms used before training and inference.

Two scripts are integrated directly with
[`run_graph_vae_sweep.py`](../run_graph_vae_sweep.py):

- [`plot_per_channel_scores.py`](plot_per_channel_scores.py)
- [`plot_per_channel_per_window_scores.py`](plot_per_channel_per_window_scores.py)

The other scripts are standalone utilities. They currently use editable constants
near the top of each file instead of command-line arguments.

## Requirements and conventions

Run commands from the repository root unless an absolute path is shown:

```bash
python graphing/<script_name>.py [arguments]
```

A scripts require NumPy and Matplotlib. `plot_pulse.py` additionally requires
PyROOT and access to the referenced ROOT files.

The score plotters expect GraphVAE score files with:

| Array | Expected shape | Meaning |
| --- | ---: | --- |
| `node_scores` | `(W, C)` | Per-window, per-channel reconstruction/anomaly scores. `W` is the number of windows and `C` is the number of channels. Inactive channels may be `NaN`. |

Note: If model training and inference complete successfully, the correctly formatted GraphVAE score files will be generated automatically.

The sparse-event utilities use some or all of these arrays:

| Array | Meaning |
| --- | --- |
| `channels_flat` | Channel ID for every stored hit. |
| `integrals_flat` | Integral value for every stored hit. |
| `offsets` | CSR-style event boundaries; event `i` occupies flat entries `offsets[i]:offsets[i+1]`. |
| `n_channels` | Total number of channel columns represented by the file. |
| `evt_time` | Event timestamps in nanoseconds. |
| `evt_run` | Run number for each event. |

## Script summary

| Script | Main result | Command-line interface | Sweep integration |
| --- | --- | --- | --- |
| `plot_per_channel_scores.py` | Good/bad mean node score versus channel | One optional positional argument | Yes |
| `plot_per_channel_per_window_scores.py` | Six per-plane channel-distribution plots for selected windows | Positional input plus six optional flags | Yes |
| `plot_debug.py` | Target-good-run versus other-good-run versus bad-run integral distributions | No flags; edit constants | No |
| `plot_event_hist.py` | Between-event-time histogram and printed window-duration diagnostics | No flags; edit constants | No |
| `plot_mean_median_hist.py` | Histograms of per-window/per-channel integral mean and standard deviation | No flags; edit constants | No |
| `plot_pulse.py` | Overlay of four ROOT waveform histograms | No flags; edit constants | No |
| `plot_time_window_box_whisker.py` | Per-run boxplots of event counts in fixed-duration windows | No flags; edit constants | No |

## `plot_per_channel_scores.py`

### What it does

Loads `scores_good.npz` and `scores_bad.npz` from one model's
`inference_result` directory, calculates the finite-value mean of `node_scores`
over all windows for every channel, and overlays the good and bad channel curves.

If `SMOOTHING` is enabled, the script removes isolated extreme points using a
local median and median absolute deviation (MAD). Removed points become `NaN`.
Despite the setting name, normal points are not averaged or smoothed.

### Inputs and output

- Required files: `scores_good.npz` and `scores_bad.npz`.
- Required array: two-dimensional `node_scores` in both files.
- The two files must contain the same number of channels.
- Output: `<inference_result_dir>/channel_mean_node_scores.png`.

### Command

```bash
python graphing/plot_per_channel_scores.py \
  checkpoints/graph_vae/<model_name>/inference_result
```

### Arguments and flags

| Argument or flag | Default | Description |
| --- | --- | --- |
| `inference_result_dir` | `DEFAULT_INFERENCE_RESULT_DIR` | Optional positional directory containing `scores_good.npz` and `scores_bad.npz`. |
| `-h`, `--help` | — | Display the generated help message. |

There are no command-line flags for the channel range, spike removal, or figure
style. Change these constants near the top of the script when running it
directly:

| Constant | Current default | Effect |
| --- | ---: | --- |
| `CHANNEL_START` | `0` | First plotted channel, inclusive. |
| `CHANNEL_END` | `11276` | Final channel bound, exclusive. An out-of-range end falls back to the detected channel count. |
| `SMOOTHING` | `True` | Enable local-MAD spike removal. |
| `SMOOTHING_WINDOW` | `1001` | Local channel neighborhood. An even value is increased by one. |
| `SMOOTHING_THRESHOLD` | `5` | A point is removed when its deviation exceeds this many robust local standard deviations. |
| `OUTPUT_FILENAME` | `channel_mean_node_scores.png` | Filename written inside the inference directory. The sweep runner also uses this value to check whether the plot exists. |
| `FIGSIZE` | `(16, 7)` | Matplotlib figure size in inches. |
| `DPI` | `200` | Saved image resolution. |

### Callable functions

| Function | Purpose |
| --- | --- |
| `load_mean_node_scores(npz_path)` | Load `node_scores` and calculate a NaN-aware mean per channel. |
| `resolve_channel_range(start, end, num_channels)` | Validate or repair the requested half-open channel interval. |
| `remove_large_spikes(scores, window_size, threshold, label)` | Replace isolated local-MAD outliers with `NaN`. |
| `main(inference_result_dir=None)` | Validate both score files and create the combined plot. This is the entry point used by the sweep runner. |

## `plot_per_channel_per_window_scores.py`

### What it does

Selects rows of `node_scores` by zero-based window index and shows the score
distribution separately for each physical detector plane. The channel-map CSV
must resolve into exactly six non-overlapping groups, so the script writes six
plots.

Two display styles are available:

- `mean_std`: a per-channel mean line with a shaded
  `mean +/- STD_BAND_SIGMAS * population standard deviation` band.
- `box`: per-channel Q1, median, Q3, 1.5-IQR whiskers, and optional outliers.

If `scores_train.npz` exists, its mean and standard-deviation band is included in
green by default, even when the selected good/bad curves use boxplots. Use
`--good-bad-only` to suppress it when running the plotter directly.

### Inputs and outputs

- `scores_good.npz` is required when `--datasets good` or `both` is selected.
- `scores_bad.npz` is required when `--datasets bad` or `both` is selected.
- `scores_train.npz` is optional and is automatically added unless
  `--good-bad-only` is set.
- Every loaded file must have the same number of channel columns.
- The selected numeric window indices must exist in every loaded file. With
  `--windows all`, each file uses all of its own windows.
- The channel-map channel IDs must directly index the `node_scores` columns.
  A full-detector map will fail for a score file whose subset channels were
  reindexed to `0...C-1` unless a matching subset map is supplied.
- Output names are formed by inserting `_plane_1` through `_plane_6` before the
  suffix. With the default name, the files are
  `channel_node_scores_selected_windows_plane_1.png` through
  `channel_node_scores_selected_windows_plane_6.png`.

### Example commands

Plot all good and bad windows with the default mean/std style:

```bash
python graphing/plot_per_channel_per_window_scores.py \
  checkpoints/graph_vae/<model_name>/inference_result \
  --windows all \
  --datasets both \
  --channel-map configs/SBNDTPCChannelMap_v2_with_positions.csv
```

Plot selected windows as boxplots without the training reference:

```bash
python graphing/plot_per_channel_per_window_scores.py \
  checkpoints/graph_vae/<model_name>/inference_result \
  --windows '0,4,8,20-30' \
  --datasets both \
  --plot-style box \
  --good-bad-only \
  --output selected_boxplots.png
```

### Arguments and flags

| Argument or flag | Default | Description |
| --- | --- | --- |
| `inference_result_dir` | `DEFAULT_INFERENCE_RESULT_DIR` | Optional positional score directory. |
| `--windows SPEC` | `all` | Zero-based indices and inclusive ranges, for example `1200`, `1,4,8`, `1200-1210`, or `all`. Repeated indices are deduplicated and sorted. |
| `--datasets {good,bad,both}` | `both` | Select the good score file, bad score file, or both. |
| `--channel-map PATH` | `CHANNEL_MAP_PATH` | CSV used to assign score columns to exactly six detector-plane groups. |
| `--output PATH` | `channel_node_scores_selected_windows.png` | Base output name or path. Relative paths are placed inside `inference_result_dir`; `.png` is added when no suffix is supplied. |
| `--plot-style {mean_std,box}` | `mean_std` | Select the mean/std-band or box-and-whisker representation. |
| `--good-bad-only` | off | Do not add `scores_train.npz`, even if it exists. |
| `-h`, `--help` | — | Display the generated help message. |

Additional editable constants include `STD_BAND_SIGMAS=1.0`,
`WHISKER_IQR=1.5`, `SHOW_FLIERS=True`, `FIGSIZE=(18, 7)`, `DPI=180`, and the
good, bad, and training colors.

### Callable functions

| Function | Purpose |
| --- | --- |
| `parse_window_spec(spec, num_windows)` | Parse, deduplicate, sort, and validate the requested window rows. |
| `load_plane_channel_groups(channel_map_path, num_channels)` | Detect channel/plane/TPC CSV columns and construct exactly six channel groups. |
| `load_selected_node_scores(npz_path, window_spec)` | Load `node_scores` and return only the requested rows. |
| `get_output_paths(inference_result_dir, output)` | Return the six expected plane-specific output paths without creating plots. The sweep runner uses this for missing-file checks. |
| `calculate_box_statistics(...)` / `draw_boxplots(...)` | Calculate and draw exact per-channel box-and-whisker statistics. |
| `calculate_mean_std(...)` / `draw_mean_std_band(...)` | Calculate finite-value per-channel means/population standard deviations and draw the line and band. |
| `main(...)` | Create and return the six output paths. The sweep runner calls this function directly. |

## Sweep-runner integration

The sweep runner loads the two integrated plotting scripts with `runpy`, then
calls their `main()` functions for each model's
`<runs-root>/<config_stem>/inference_result` directory. It does not launch the
plotters as independent subprocesses.

### Plot-related sweep flags

The table lists the canonical hyphenated spellings. Underscore aliases are also
accepted where they are defined by the sweep parser.

| Sweep flag | Default | Effect |
| --- | --- | --- |
| `--per-channel-plot` | off | Run `plot_per_channel_scores.py` after a successful good/bad inference pair. Accepted aliases also include singular/plural combinations of hyphens and underscores. |
| `--per-channel-per-window-plot` | off | Run `plot_per_channel_per_window_scores.py` after a successful good/bad inference pair. Alias: `--per_channel_per_window_plot`. |
| `--per-channel-window-indices SPEC` | plotter default (`all`) | Window selection forwarded to the selected-window plotter. Alias: `--per_channel_window_indices`. |
| `--per-channel-window-datasets {good,bad,both}` | `both` | Good/bad score files used by the selected-window plotter. Alias: `--per_channel_window_datasets`. |
| `--per-channel-per-window-plot-name NAME` | `channel_node_scores_selected_windows.png` | Base filename for the six plane plots. Alias: `--per_channel_per_window_plot_name`. |
| `--missing-plot` | off | Standalone repair mode. Recreate a missing all-window per-channel plot, recreate all six plane plots if any one is absent, and recreate a missing `goodvsbad.png`. Alias: `--missing_plot`. |
| `--force-replot` | off | Standalone mode that regenerates the same three plot sets even when they already exist. Alias: `--force_replot`. |
| `--eval-plot-name NAME` | `goodvsbad.png` | Name checked/created for the good-vs-bad evaluation plot in plot-only modes. |
| `--eval-aggregator NAME` | `group_max_mean` | Aggregator used when a plot-only mode recreates the good-vs-bad plot. |
| `--eval-threshold VALUE` | unset | Explicit good-vs-bad operating threshold; overrides `--eval-percentile`. |
| `--eval-percentile VALUE` | `95` | Good-set percentile used when no explicit threshold is supplied. |
| `--eval-per-run` | off | Add per-run evaluation when recreating `goodvsbad.png`. |
| `--channel-map PATH` | project channel map | Channel map used by `window_score` for `goodvsbad.png`. It is not forwarded to the six-plane graphing script. |
| `--good-label LABEL` / `--bad-label LABEL` | `good` / `bad` | Labels used by the recreated good-vs-bad plot. |

`goodvsbad.png` is not produced by a script in this directory. The sweep runner
creates it with `python -m sbn_anomaly.infer.window_score`.

The sweep parser's help text currently calls the selected-window script
`plot_per_channel_scores_per_window.py`; this is an outdated name. The path
actually loaded by the runner is
`graphing/plot_per_channel_per_window_scores.py`.

### When integrated plotting runs

- The two direct plotting flags run only when the sweep actually performs
  good/bad inference: normal train+infer, `--infer-only`, or a
  `--missing-infer-only` run that really needs inference.
- `--skip-eval` may be combined with the two plotting flags. The per-channel
  plots are created after inference and before evaluation is skipped.
- In the standalone `--missing-plot` and `--force-replot` branches,
  `--skip-eval` does not suppress regeneration of a missing or forced
  `goodvsbad.png`; the plot-only branch calls `window_score` directly.
- `--evaluate-only` does not run the graphing scripts because it does not perform
  inference. Use `--missing-plot` or `--force-replot` to plot existing scores.
- `--missing-infer-only` can skip a model as soon as its inference/evaluation
  outputs exist, even if a per-channel plot is absent. Use `--missing-plot` to
  repair plots without rerunning inference.
- `--missing-plot` and `--force-replot` automatically enable both integrated
  plot types, but they are mutually exclusive standalone modes. They cannot be
  combined with training, inference-only, evaluation-only, baseline-inference,
  skip-inference, or rewrite modes.
- `--baseline-infer` cannot be combined with either per-channel plotting flag.
  Baseline mode replaces the good/bad inference stage with training-set scoring.
- Plot-only modes require existing `scores_good.npz` and `scores_bad.npz`. A model
  is skipped if either file is missing.
- Model directories listed in the sweep runner's `IGNORED` constant are skipped,
  including during plot-only modes.

### Settings not forwarded by the sweep runner

The selected-window plotter supports `--channel-map`, `--plot-style`, and
`--good-bad-only` when run directly, but the sweep runner does not expose or
forward those three settings. Therefore, during a sweep:

- the plotter uses its `CHANNEL_MAP_PATH` constant;
- the plotter uses its `PLOT_STYLE` constant;
- `scores_train.npz` is automatically included when present.

Run the plotter directly for per-run overrides, or change its constants before a
full sweep. In particular, the sweep's `--channel-map` controls `window_score`
evaluation and `goodvsbad.png`; it does not control the channel map used by the
six plane plots.

### Sweep examples

Create both integrated plot types after inference:

```bash
python run_graph_vae_sweep.py \
  --infer-only \
  --per-channel-plot \
  --per-channel-per-window-plot \
  --per-channel-window-indices '0-99' \
  --per-channel-window-datasets both
```

Create the plots during normal train/infer while skipping `window_score`
evaluation:

```bash
python run_graph_vae_sweep.py \
  --per-channel-plot \
  --per-channel-per-window-plot \
  --skip-eval
```

Repair only missing plots from existing score files:

```bash
python run_graph_vae_sweep.py --missing-plot
```

Regenerate all managed plots from existing score files:

```bash
python run_graph_vae_sweep.py --force-replot
```

## `plot_debug.py`

### What it does

Compares hit-level `integrals_flat` distributions for three groups:

1. one selected `TARGET_RUN` taken from the good-run NPZ;
2. every other run in the same good-run NPZ;
3. every entry in the bad-run NPZ.

It expands event-level `evt_run` values through the CSR `offsets` so every flat
integral receives its event's run label. The three density-normalized step
histograms share common bins and a common x-axis limit.

### Inputs, output, and settings

- Required arrays in both files: `offsets`, `evt_run`, and `integrals_flat`.
- Output: beside the good NPZ, named
  `<good_stem>_run_<TARGET_RUN>_good_bad_integral_histogram.png`.
- No command-line flags are defined. Edit `GOOD_NPZ_PATH`, `BAD_NPZ_PATH`,
  `TARGET_RUN`, `N_BINS`, `X_MAX_PERCENTILE`, `LOG_Y`, and `OUTPUT_PATH`.
- `X_MAX_PERCENTILE=None` shows the complete combined range; otherwise it must
  be in `(0, 100]`.

```bash
python graphing/plot_debug.py
```

This script performs its loading and plotting at module import time rather than
inside `main()`, so importing it also executes the plot.

## `plot_event_hist.py`

### What it does

Treats `evt_time` as nanoseconds and:

- prints counts and statistics for negative, zero, and positive differences
  between consecutive stored events;
- constructs event-count windows for every `(window_size, stride)` pair in
  `WINDOW_SETTINGS` and prints their elapsed-time statistics;
- identifies windows longer than `LONG_WINDOW_THRESHOLD_DAYS`, checks whether
  they cross run numbers, and prints a second duration summary excluding only
  long multi-run windows;
- plots the positive between-event intervals in minutes with a logarithmic
  y-axis and percentile-limited x-axis.

The long-window exclusion affects the printed filtered summary, not the
between-event histogram.

### Inputs, output, and settings

- Required arrays: `evt_time` and `evt_run`, with equal lengths.
- Output: `<OUTPUT_PATH>/<OUTPUT_NAME>`; the current default name is
  `between_event_time_histogram.png`.
- No command-line flags are defined. Edit `NPZ_PATH`, `OUTPUT_PATH`,
  `WINDOW_SETTINGS`, `LONG_WINDOW_THRESHOLD_DAYS`, `HIST_BINS`,
  `X_AXIS_PERCENTILE`, and `OUTPUT_NAME`.

```bash
python graphing/plot_event_hist.py
```

Important callable helpers are `calculate_window_information()`,
`print_duration_summary()`, `inspect_long_windows()`, and `main()`.

## `plot_mean_median_hist.py`

### What it does

For every event-count window and every channel, calculates the population mean
and population standard deviation of the hit integral values. Inactive channels
are represented by zero. The script then writes two log-y histograms; the bar
containing zero is colored red.

Despite the filename, the script does not currently make a median histogram. Its
two outputs are:

- `integral_mean_histogram.png`
- `integral_stdev_histogram.png`

### Inputs, output, and settings

- Required arrays: `channels_flat`, `integrals_flat`, `offsets`, and
  scalar `n_channels`.
- Every histogram contains `number_of_windows * n_channels` entries.
- `OUTPUT_DIR` is currently relative to the process's working directory.
- No command-line flags are defined. Edit `NPZ_PATH`, `WINDOW_SIZE`, `STRIDE`,
  `HIST_BINS`, `X_AXIS_PERCENTILE`, and `OUTPUT_DIR`.

```bash
python graphing/plot_mean_median_hist.py
```

The principal functions are `calculate_window_channel_stats()`,
`plot_histogram()`, and `main()`.

## `plot_pulse.py`

### What it does

Opens four ROOT files, retrieves one canvas from each, extracts the `hRaw`
histogram from each canvas, and overlays the four waveforms in Matplotlib. ROOT
files remain open until plotting is complete.

### Inputs, output, and settings

- `ROOT_FILES`, `CANVAS_PATHS`, `LABELS`, `COLORS`, and `LINESTYLES` describe the
  overlaid curves in matching order.
- Each canvas must contain a primitive named `hRaw` unless the default argument
  of `hist_to_arrays(canvas, hist_name)` is changed.
- Output: `OUTPUT_PATH`, currently `pulse_comparison.png` in the working
  directory.
- No command-line flags are defined.

```bash
python graphing/plot_pulse.py
```

Like `plot_debug.py`, this script executes at import time.

## `plot_time_window_box_whisker.py`

### What it does

Counts events in fixed-duration windows independently for every run, then makes
one box-and-whisker plot per requested duration. A window never crosses a run
boundary. A backward timestamp jump within one run also starts a new independent
segment.

For each segment, timestamps are sorted and windows start at the first event,
advance by `TIME_WINDOW_STRIDE_SECONDS`, and continue through the final event.
The final starts therefore produce trailing partial-duration windows. Counts use
the half-open interval `[start, start + duration)`. A dotted horizontal target
line is drawn when `EVENT_NUM_TARGET` is not `None`.

### Inputs, outputs, and settings

- `INPUT_PATH` may be one `.npz` file or a directory. Directory discovery is
  non-recursive and uses only directly contained `*.npz` files.
- Required arrays: `evt_time` in nanoseconds and `evt_run`.
- One file named `events_per_window_by_run_boxplot_<duration>.png` is written
  under `OUTPUT_DIR` for every entry in `TIME_WINDOW_SIZES_SECONDS`.
- No command-line flags are defined. Edit `INPUT_PATH`, `OUTPUT_DIR`,
  `TIME_WINDOW_SIZES_SECONDS`, `TIME_WINDOW_STRIDE_SECONDS`,
  `EVENT_NUM_TARGET`, `SKIP_EMPTY_WINDOWS`, `FIGURE_WIDTH`, `FIGURE_HEIGHT`, and
  `DPI`.
- `POINT_SIZE`, `POINT_ALPHA`, `USE_HORIZONTAL_JITTER`, `JITTER_WIDTH`, and
  `RANDOM_SEED` are defined but are not currently used by the plotting function.

```bash
python graphing/plot_time_window_box_whisker.py
```

The principal functions are `discover_npz_files()`, `load_event_metadata()`,
`split_at_backward_jumps()`, `count_timed_windows_by_run()`,
`plot_events_per_window_by_run()`, and `main()`.

## Troubleshooting

- **`'node_scores' not found`**: the selected file is not a GraphVAE inference
  score file, or inference did not save per-channel scores.
- **Different channel counts**: good, bad, and optional training scores must use
  the same channel layout.
- **Channel-map ID outside `0...C-1`**: the scores probably use a reindexed
  channel subset. Supply a channel map matching that indexing.
- **Expected six planes but derived another number**: verify the detected channel,
  plane, and optional TPC columns, or set the `CHANNEL_MAP_*_COLUMN` constants to
  exact CSV header names.
- **Selected window outside range**: make the numeric selection valid for every
  loaded score file, including `scores_train.npz`, or use `--good-bad-only`.
- **`--missing-infer` did not restore a plot**: it repairs inference outputs, not
  plotting outputs. Use `--missing-plot`.
- **No plot during `--evaluate-only`**: the integrated graphing hooks run after
  inference, so use a plot-only mode or invoke the plotter directly.
