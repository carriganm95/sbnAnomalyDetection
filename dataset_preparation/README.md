# Dataset-preparation scripts

This directory contains scripts for combining, splitting, filtering, and
checking the NPZ datasets used by the anomaly-detection workflow. This README
documents only the scripts in `dataset_preparation/`.

Run the commands below from the repository root:

```bash
python dataset_preparation/<script>.py [arguments]
```

## Scripts at a glance

| Script | Purpose | Modifies or creates data? |
| --- | --- | --- |
| [`train_test_from_npz.py`](train_test_from_npz.py) | Combine source NPZ files, exclude configured bad files from the nominal data, and make run-preserving good/training splits. | Writes `good_runs.npz`, `windows_train.npz`, and optionally `bad_runs.npz`. |
| [`filter.py`](filter.py) | Slice three sparse NPZ datasets to one channel range, reindex the selected channels, and optionally filter runs or adjust empty-event retention. | Writes one filtered NPZ beside each input. |
| [`inspect_run.py`](inspect_run.py) | Analyze timestamp order, run segments, timed-window event counts, padding, and fixed-event-window durations. | Read-only unless `--output-csv` is used. |
| [`check_time_sequence.py`](check_time_sequence.py) | Find places where `evt_time` decreases in one NPZ file. | Read-only. |

## `train_test_from_npz.py`

### Purpose

This script discovers source files matching `tpc_data_v3_*.npz`, separates them
by configured **file number**, combines complete events, sorts them by
`evt_time`, and produces nominal good and training datasets. It can also combine
the configured bad source files.

The file number is the numeric suffix in a filename such as
`tpc_data_v3_20516.npz`. Membership in `BAD_FILE_NUMBERS` is determined from
this suffix—not from `evt_run` inside the file.

### Processing steps

1. Search `INPUT_DIR` non-recursively for `FILE_PATTERN`.
2. Put files whose numeric suffix is in `BAD_FILE_NUMBERS` into the bad-file
   group; all remaining files are eligible for the nominal split.
3. Count all eligible events in a first pass. `TARGET_RUN` is reported for
   diagnostics only and does not affect the split.
4. Combine all eligible files while keeping every event's flat hit slice and
   event metadata synchronized.
5. Sort complete events by `evt_time`, using run, subrun, and event number as
   deterministic tie-breakers.
6. Order runs by their earliest timestamp and choose the complete-run boundary
   nearest the requested `GOOD_RUNS_FRACTION`. A run is never divided between
   `good_runs.npz` and `windows_train.npz`, so the actual fraction may differ
   from the target.
7. If `--with-bad` is supplied, combine all events from the configured bad
   source files and sort them by `evt_time`.

### User settings

These values are constants near the top of the script; they do not have CLI
overrides.

| Constant | Current value | Meaning |
| --- | --- | --- |
| `INPUT_DIR` | `/exp/sbnd/data/users/micarrig/DQM/` | Directory searched for source NPZ files. |
| `OUTPUT_DIR` | `/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection/dataset_preparation` | Destination directory. |
| `WINDOWS_TRAIN_PATH` | `<OUTPUT_DIR>/windows_train.npz` | Training output. |
| `GOOD_RUNS_PATH` | `<OUTPUT_DIR>/good_runs.npz` | Nominal good-test output. |
| `BAD_RUNS_PATH` | `<OUTPUT_DIR>/bad_runs.npz` | Optional bad-data output. |
| `GOOD_RUNS_FRACTION` | `0.75` | Target fraction of eligible events for `good_runs.npz`; the rest go to `windows_train.npz`. |
| `BAD_FILE_NUMBERS` | `{830, 20173, 20614, 20615, 20620, 20621}` | Source-file numbers excluded from the nominal split and optionally combined as bad data. |
| `FILE_PATTERN` | `tpc_data_v3_*.npz` | Non-recursive input glob. |
| `TARGET_RUN` | `20142` | Run reported during the first pass; it does not control selection. |

### Flags

| Flag | Default | Effect |
| --- | --- | --- |
| `--with-bad` | off | Also combine the configured bad source files and write `bad_runs.npz`. Without this flag, the script does not touch an existing `bad_runs.npz`. |
| `-h`, `--help` | — | Show the command help. |

### Input requirements

Every source needs a valid one-dimensional `offsets` array beginning at zero
and containing non-decreasing event boundaries. For the nominal split,
`evt_time` and `evt_run` are required. Recognized arrays are grouped by the
script as follows:

- Hit-level arrays: `channels_flat`, `integrals_flat`, `times_flat`,
  `wires_flat`, `planes_flat`, `tpcs_flat`, `widths_flat`, `sumadcs_flat`,
  `mults_flat`, and `hassps_flat`.
- Event-level arrays: `evt_run`, `evt_subrun`, `evt_num`, `evt_time`, and
  `evt_file_idx`.
- Scalar metadata: `n_channels`.

All files combined into the same output must contain the same recognized
hit-level, event-level, and scalar key sets. Scalar `n_channels` values must
agree. The script remaps `evt_file_idx`, rebuilds `offsets`, and records
`filenames`, `combined_file_numbers`, `configured_bad_file_numbers`, and
`num_source_files` in the output.

### Commands

Create only the nominal good/training split:

```bash
python dataset_preparation/train_test_from_npz.py
```

Also create `bad_runs.npz`:

```bash
python dataset_preparation/train_test_from_npz.py --with-bad
```

Existing output files with the same names are overwritten.

## `filter.py`

### Purpose

This script processes exactly three inputs—training, good-test, and bad-test—
using identical channel and empty-event settings. It:

- keeps hits whose global channel satisfies `START <= channel < END`;
- reindexes those channels by subtracting `START`, producing local channel IDs
  in `[0, END - START)`;
- rebuilds `offsets` and keeps all recognized hit/event arrays synchronized;
- removes events with no selected-range hits by default;
- optionally filters each input by its own `evt_run` list;
- optionally retains a subset of zero-selected-hit events to bring the selected
  region's mean per-event active-channel ratio closer to the full-detector
  ratio; and
- stably sorts complete output events by `evt_time`.

All three input paths must be different. If any input fails, the command stops;
already-written earlier outputs are not rolled back.

### Required and preserved arrays

The script directly requires `channels_flat`, `integrals_flat`, `offsets`,
`n_channels`, `evt_run`, `evt_subrun`, `evt_num`, `evt_file_idx`, and
`filenames`. Although `evt_time` is initially treated as optional while loading,
the final mandatory time sort requires it in practice.

When present, it also filters and preserves these synchronized arrays:

- Hit-level: `times_flat`, `wires_flat`, `planes_flat`, `tpcs_flat`,
  `widths_flat`, `sumadcs_flat`, `mults_flat`, and `hassps_flat`.
- Event-level: `evt_time`.

The script adds metadata describing the original channel range, reindexing,
empty-event behavior, run filtering, force-ratio behavior, random seed, and
full-versus-selected channel-occupancy statistics.

### Flags

| Flag | Default | Effect |
| --- | --- | --- |
| `--train PATH`, `--train-input PATH`, `--train_input PATH` | `DEFAULT_TRAIN_NPZ_PATH` | Training NPZ input. |
| `--good PATH`, `--good-input PATH`, `--good_input PATH` | `DEFAULT_GOOD_NPZ_PATH` | Good-test NPZ input. |
| `--bad PATH`, `--bad-input PATH`, `--bad_input PATH` | `DEFAULT_BAD_NPZ_PATH` | Bad-test NPZ input. |
| `--start N` | `8800` | Inclusive global channel start. |
| `--end N` | `9000` | Exclusive global channel end. |
| `--total-n-channels N`, `--total_n_channels N` | `11276` | Full-detector channel count used only in occupancy-ratio calculations. |
| `--keep-empty`, `--keep_empty` | off | Keep every run-eligible event even if it has no hit in the selected channel range. |
| `--force-ratio`, `--force_ratio`, `--force-balance`, `--force_balance`, `--force`, `-f` | off | When empty events would otherwise be removed, try to improve agreement with the full-detector mean active-channel ratio by retaining a selected subset of zero-hit events. |
| `--run-filter`, `--run_filter` | off | Enable per-input filtering using the corresponding run list. Listed runs are kept unless reverse mode is enabled. |
| `--reverse-run-filter`, `--reverse_run_filter` | off | Remove the listed runs instead of keeping them. Requires `--run-filter`. |
| `--train-runs RUN [RUN ...]`, `--train_runs ...` | `DEFAULT_TRAIN_RUNS_TO_KEEP` | Training-file run list used only with `--run-filter`. |
| `--good-runs RUN [RUN ...]`, `--good_runs ...` | `DEFAULT_GOOD_RUNS_TO_KEEP` | Good-test run list used only with `--run-filter`. |
| `--bad-runs RUN [RUN ...]`, `--bad_runs ...` | `DEFAULT_BAD_RUNS_TO_KEEP` | Bad-test run list used only with `--run-filter`. |
| `--random-seed N`, `--random_seed N` | `12345` | Seed used when `--force-ratio` selects zero-hit events. |
| `-h`, `--help` | — | Show the command help. |

The current default run lists are:

- Training: `20184 20186 20188 20190 20199 20201 20202 20205 20207 20208 20209 20211 20218 20219 20221 20223 20224`
- Good test: `20074 20075 20079 20080 20082 20101 20104 20105 20106 20108 20110 20113 20115 20116 20117 20118 20121 20126 20127 20128 20130 20132 20144 20153`
- Bad test: `19627 19946`

When `--run-filter` is enabled, all three run lists must be non-empty. Run
filtering is applied before channel-based event selection.

### Empty-event and force-ratio behavior

- Default: discard events with zero hits in the selected range.
- `--keep-empty`: retain all run-eligible empty events.
- `--force-ratio`: start with every event having selected-range hits, then
  retain only as many available zero-hit events as needed to improve the match
  to the full-detector mean ratio. The result may therefore contain empty
  events even though `--keep-empty` was not supplied.
- `--force-ratio --keep-empty`: force-ratio processing is ignored because all
  empty events are already retained.
- A force-ratio candidate is used only if it improves the absolute difference
  between the selected and full-detector mean ratios. Check the output metadata
  `force_ratio_applied`; the `_force_ratio` filename suffix indicates that the
  option was effective/requested, not necessarily that the candidate was used.

### Output names

Each output is written beside its input. The original stem receives these
suffixes in order:

```text
_ch<START>_<END>_reindexed
[_nonempty]
[_runfilter | _reverse_runfilter]
[_force_ratio]
```

`_nonempty` is added under the default empty-event-removal behavior. For
example:

```text
events_train_ch8800_9000_reindexed_nonempty.npz
```

Existing output files with the same names are overwritten.

### Examples

Filter all three datasets to channels `[8800, 9000)` and remove empty events:

```bash
python dataset_preparation/filter.py \
  --train data/events_train.npz \
  --good data/good_events_test.npz \
  --bad data/bad_events_test.npz \
  --start 8800 \
  --end 9000
```

Keep empty events:

```bash
python dataset_preparation/filter.py \
  --train data/events_train.npz \
  --good data/good_events_test.npz \
  --bad data/bad_events_test.npz \
  --start 8800 \
  --end 9000 \
  --keep-empty
```

Keep selected runs independently in the three datasets:

```bash
python dataset_preparation/filter.py \
  --train data/events_train.npz \
  --good data/good_events_test.npz \
  --bad data/bad_events_test.npz \
  --start 8800 \
  --end 9000 \
  --run-filter \
  --train-runs 20184 20186 20188 \
  --good-runs 20074 20117 20128 \
  --bad-runs 19627 19946
```

Remove listed runs instead:

```bash
python dataset_preparation/filter.py \
  --train data/events_train.npz \
  --good data/good_events_test.npz \
  --bad data/bad_events_test.npz \
  --start 8800 \
  --end 9000 \
  --run-filter \
  --reverse-run-filter \
  --train-runs 20184 \
  --good-runs 20104 \
  --bad-runs 19946
```

## `inspect_run.py`

### Purpose

This is a read-only timestamp and windowing diagnostic for one NPZ file or a
directory of NPZ files. It reports:

- raw timestamp range and consecutive positive, zero, and backward differences;
- run changes and backward jumps that occur without a run change;
- per-run event counts, timestamp segments, durations, and event intervals;
- a continuous diagnostic timeline stitched at run changes and backward
  timestamp resets;
- event-count distributions and padding requirements for sliding timed windows;
  and
- elapsed-time distributions for fixed-event-count windows.

Timed windows are calculated independently within each run and timestamp
segment, so they do not cross run boundaries or backward timestamp resets.

### Arguments and flags

| Argument or flag | Actual default | Effect |
| --- | --- | --- |
| `path` | required | NPZ file or directory. Directory search is non-recursive unless `--recursive` is supplied. |
| `--recursive` | off | Search subdirectories recursively. |
| `--time-key KEY` | `evt_time` | Timestamp-array key. Files missing this key are skipped. |
| `--run-key KEY` | `evt_run` | Run-array key. If absent, the script assigns run `-1` to every event in that file. |
| `--time-scale VALUE` | `1e9` | Raw timestamp units per second; `1e9` converts nanoseconds to seconds. |
| `--window-time-size SECONDS` | `1000` | Duration of each timed window. |
| `--window-time-stride SECONDS` | `15` | Separation between timed-window starts. |
| `--max-events N` | `1000` | Maximum real events selected into the fixed-size model input; excess events are counted as oversubscribed and the remainder determines padding. |
| `--event-window-size N` | `1000` | Number of events in the fixed-event-count comparison window. |
| `--event-stride N` | `10` | Separation, in events, between fixed-event-count window starts. |
| `--output-csv PATH` | none | Write one CSV row per timed window. |
| `-h`, `--help` | — | Show the command help. |

The current `--help` descriptions incorrectly say that some defaults are `600`
or `400`. The actual parser defaults come from the constants listed above:
`1000` seconds, `1000` maximum events, and a `1000`-event comparison window.

### Optional CSV

`--output-csv` writes these columns:

| Column | Meaning |
| --- | --- |
| `window_index` | Sequential output-window index. |
| `run` | Run associated with the window. |
| `time_window_start_sec` | Start time relative to the beginning of the run/timestamp segment. |
| `time_window_end_sec` | Start plus the requested duration. |
| `real_event_count` | Events found in the timed window. |
| `selected_event_count` | `min(real_event_count, max_events)`. |
| `padded_event_count` | `max_events - selected_event_count`. |

Example:

```bash
python dataset_preparation/inspect_run.py \
  data/events_train.npz \
  --window-time-size 1000 \
  --window-time-stride 15 \
  --max-events 1000 \
  --event-window-size 1000 \
  --event-stride 10 \
  --output-csv scratch/timed_window_counts.csv
```

Known implementation detail: the final shorter/longer fixed-event-window report
compares durations with the constant `DEFAULT_WINDOW_TIME_SIZE` (`1000` seconds),
even if `--window-time-size` is overridden.

## `check_time_sequence.py`

### Purpose

This small read-only checker loads one NPZ file, finds every index where
`evt_time[i+1] < evt_time[i]`, prints the total number of decreases, and shows
the first 20 transitions with run, subrun, event number, and timestamp.

It checks the stored event order as one sequence. A timestamp decrease at a run
boundary is therefore included; it is not automatically treated as harmless.

### Settings and command

There is no CLI. Edit the hard-coded `path` near the top of the file. The input
must contain `evt_time`, `evt_run`, `evt_subrun`, and `evt_num`.

```bash
python dataset_preparation/check_time_sequence.py
```

The script executes at import time because it has no `main()` guard.

## Using the scripts together

The scripts are not automatically chained. A typical sequence is:

1. Configure and run `train_test_from_npz.py` to combine source files and create
   the good/training split, optionally including bad data.
2. Pass those outputs explicitly to `filter.py` if a channel subset is needed.
   The default input names in `filter.py` do not match the output names from
   `train_test_from_npz.py`, so explicit paths are required.
3. Run `inspect_run.py` or `check_time_sequence.py` to verify timestamp behavior.

For example:

```bash
python dataset_preparation/train_test_from_npz.py --with-bad

python dataset_preparation/filter.py \
  --train dataset_preparation/windows_train.npz \
  --good dataset_preparation/good_runs.npz \
  --bad dataset_preparation/bad_runs.npz \
  --start 8800 \
  --end 9000

python dataset_preparation/inspect_run.py \
  dataset_preparation/windows_train_ch8800_9000_reindexed_nonempty.npz
```

# Important Findings and Notes about the Dataset

We have three important findings about the dataset we use to train the model.
- First, the dataset MUST be THOROUGH and COMPLETE. Specifically, the dataset, no matter
training or testing, must contain many runs, the more the better. The runs should be different
types of runs (e.g., high detector trigger and low detector trigger). We MUST prevent the dataset
to be dominated by a single run or a few runs. Otherwise, the model will fit to only these runs and 
will not be inclusive enough to cover all types of runs. Currently, the model gives a super high anomaly score
to the good runs when there is a large hit integral value, and we suspect that this is because we do not have
a representative run that has large integral values in the training dataset.
- Secondly, take extra care to the evt_time when concatenating runs to make the dataset. Currently, we make the 
datasets by concatenating different runs, but these different runs all have different run time, and some of them may even be 
a few months apart. As a result, at the junctions of runs, the time interval is going to be problematic. We expect that
using new data will help mitigate this issue (as currently we do not have new data and can only work on past data, which
we cannot control the time interval between).
- Thirdly, pay attention to the detector trigger level in the different runs. As mentioned above, the detector can have high trigger
or low trigger, and different triggers are going to significantly affect the number of events in a run. Our experiments showed that
high trigger runs can only train the model to infer high trigger runs but not low trigger runs (and vice versa). Therefore, the dataset
must be inclusive, or, if the dataset has only high trigger runs or low trigger runs, do not expect the model to perform well on the
other kind of runs.
- Fourth, collection plane data vs non collection plane data. We found that the induction planes usually have more events than the collection planes, and, probably as a result of this, it is more difficult to separate the good and bad runs in the collection planes. 
Maybe our next step should be taking a look to the pulse finding script and try to make our own pulse finding script to better assist
solving this issue.