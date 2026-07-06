#!/usr/bin/env python3
"""Build GraphVAE train/good-test/bad-test sparse event NPZ files.

This is the GraphVAE version of the train/test maker. It keeps the same
whole-run selection and balancing rules as the original script, but writes
separate test files for good and bad runs:

    events_train.npz  - selected good training runs
    good_test.npz     - selected good test runs
    bad_test.npz      - selected bad test runs
"""


from pathlib import Path
from typing import Optional, Dict, List, Set, Tuple

import re
import numpy as np


# ============================================================
# User settings
# ============================================================

# Directory containing the input .npz files
NPZ_DIR = Path("/exp/sbnd/data/users/micarrig/DQM/tpc_data")

# Output files
#
# GraphVAE trains on good-only events and evaluates good/bad test samples
# separately, so we write three files:
#   events_train.npz : selected GOOD training runs
#   good_test.npz    : selected GOOD test runs
#   bad_test.npz     : selected BAD test runs
#
# These names match configs/graph_vae.yaml conventions.
OUTPUT_PARENT_PATH = Path("/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection/data")
TRAIN_OUTPUT_PATH = OUTPUT_PARENT_PATH / "events_train.npz"
GOOD_TEST_OUTPUT_PATH = OUTPUT_PARENT_PATH / "good_test.npz"
BAD_TEST_OUTPUT_PATH = OUTPUT_PARENT_PATH / "bad_test.npz"

# Whether to search subdirectories recursively
RECURSIVE = True

# Random seed for reproducible good-file/run selection
RANDOM_SEED = 12345

# Whether to print detailed file-level information
PRINT_FILES = True

# If True, use np.savez_compressed.
# If False, use np.savez, which is usually much faster but creates larger files.
COMPRESS_OUTPUT = False

# If True:
#   If one input .npz contains multiple evt_run values, split it into virtual
#   per-run inputs during train/test selection and output writing. This is more
#   correct for aggregate files like tpc_data_bad_runs.npz, but slower because
#   writing a selected run slice requires random access into the aggregate file.
#
# If False:
#   Use the older faster behavior: each physical .npz file is treated as one
#   input item using evt_run[0] as its run label. This is faster because full
#   files can be filtered with a vectorized full-file path, but aggregate files
#   will not be split and may be selected as one huge run.
SPLIT_AGGREGATE_FILES_BY_RUN = False


# ============================================================
# Channel selection settings
# ============================================================

# Keep only channels in this half-open interval:
#
#   CHANNEL_START <= channel < CHANNEL_END
#
# Example:
#   CHANNEL_START = 9600
#   CHANNEL_END   = 9650
#
# This keeps exactly 50 channels: 9600, 9601, ..., 9649.
CHANNEL_START = 8800
CHANNEL_END = 9000


# ============================================================
# Balancing settings
# ============================================================
#
# Important:
#   The script still writes all selected windows/events to the output .npz files.
#   However, for balancing good and bad runs, events with no selected hits after
#   channel filtering do NOT count.
#
# In other words:
#   n_windows = total windows in the original sparse file, or in one virtual
#               run slice if an aggregate file is split by evt_run
#   n_balance = number of unique events that have at least one selected hit after
#               applying CHANNEL_START <= channel < CHANNEL_END
#
# Good/bad balancing uses n_balance, not n_windows.
# This prevents empty windows from dominating the class balance.
#
# Event balance is based on unique:
#
#   (evt_run, evt_subrun, evt_num)
#
# So if one event appears in multiple windows, it is counted once.
#
# Aggregate input files:
#   If one .npz contains multiple evt_run values, this script splits it into
#   virtual per-run inputs during selection and output writing. This prevents a
#   combined file such as tpc_data_bad_runs.npz from being treated as one huge
#   unsplittable run.


# If None:
#   use all selected training runs.
#
# If an integer:
#   cap the training set at approximately this many unique non-empty selected
#   events, while keeping whole runs together.
#
# Because runs are never split, the final training balance count may be below
# this cap. If the minimum training selection already exceeds this cap, the
# script raises an error instead of slicing through a run.
MAX_TRAIN_EVENT_N = None
# MAX_TRAIN_EVENT_N = 100000


# If None:
#   use the original split behavior, but balance using unique non-empty selected
#   events:
#     - select all bad runs for test
#     - randomly select enough good runs to approximately match the bad
#       unique non-empty selected event count
#     - put everything else into training
#
# If an integer:
#   first reserve good runs for training until at least this many unique
#   non-empty selected events are available for training.
#
#   Then the test set is built only from the remaining good runs and the bad
#   runs.
#
#   In this mode, if there are not enough remaining good unique non-empty events
#   to match all bad unique non-empty events, or not enough bad unique non-empty
#   events to match all remaining good unique non-empty events, the script uses
#   the lower available balance count as the target for each test class.
#
# Note:
#   Run selection is whole-run based, so final selected counts can be a bit
#   above the target if the last selected run crosses the target. No run is
#   split between train and test.
MIN_TRAIN_EVENT_N = 5000
# MIN_TRAIN_EVENT_N = None


# If True:
#   original channels 9600, 9601, ..., 9649 become 0, 1, ..., 49
#   and n_channels is saved as CHANNEL_END - CHANNEL_START.
#
# If False:
#   original channel numbers are kept unchanged
#   and n_channels is kept from the original files.
#
# For training, True is usually better.
REINDEX_CHANNELS = True


# --------------------------
# Good/Bad Runs List
# --------------------------
GOOD_RUNS = {
    18445,
    19724,
    20141,
    20142,
    20144,
}

BAD_RUNS = {
    19627,
    19946,
    20104,
}


# ============================================================
# NPZ key settings
# ============================================================

# Sparse flat arrays. These are filtered by channels_flat and concatenated.
FLAT_KEYS = [
    "channels_flat",
    "integrals_flat",
    "times_flat",
    "wires_flat",
    "planes_flat",
    "tpcs_flat",
]

# Per-window arrays. These should have length equal to number of windows.
PER_WINDOW_KEYS = [
    "evt_run",
    "evt_subrun",
    "evt_num",
    "evt_file_idx",
]

# Special keys
OFFSETS_KEY = "offsets"
N_CHANNELS_KEY = "n_channels"
FILENAMES_KEY = "filenames"

EventKey = Tuple[int, int, int]


def extract_last_integer(path: Path) -> int:
    """
    Extract the last integer from a filename.

    Example:
      tpc_data_20620.npz -> 20620

    This is used to sort files inside the same run.
    """
    numbers = re.findall(r"\d+", path.stem)

    if not numbers:
        return -1

    return int(numbers[-1])


def get_npz_files() -> List[Path]:
    """Return all .npz files in NPZ_DIR, excluding output files."""
    if not NPZ_DIR.exists():
        raise FileNotFoundError(f"Directory does not exist: {NPZ_DIR}")

    if not NPZ_DIR.is_dir():
        raise NotADirectoryError(f"Not a directory: {NPZ_DIR}")

    if RECURSIVE:
        files = sorted(NPZ_DIR.rglob("*.npz"))
    else:
        files = sorted(NPZ_DIR.glob("*.npz"))

    output_paths = {
        TRAIN_OUTPUT_PATH.resolve(),
        GOOD_TEST_OUTPUT_PATH.resolve(),
        BAD_TEST_OUTPUT_PATH.resolve(),
    }

    files = [
        path
        for path in files
        if path.resolve() not in output_paths
    ]

    return files


def validate_window_arrays(data, npz_path: Path) -> None:
    """Validate per-window arrays and offsets for one input .npz."""
    evt_run = data["evt_run"]
    offsets = data[OFFSETS_KEY]
    n_windows = int(evt_run.shape[0])

    if len(offsets) != n_windows + 1:
        raise ValueError(
            f"Bad offsets length in {npz_path}: "
            f"len(offsets)={len(offsets)}, n_windows={n_windows}. "
            f"Expected len(offsets)=n_windows+1."
        )

    for key_name in ["evt_subrun", "evt_num"]:
        arr = data[key_name]
        if arr.shape[0] != n_windows:
            raise ValueError(
                f"Per-window key {key_name} has wrong length in {npz_path}: "
                f"{arr.shape[0]} vs n_windows={n_windows}"
            )


def get_nonempty_selected_event_keys_for_window_mask(
    data,
    npz_path: Path,
    window_mask: np.ndarray,
) -> Set[EventKey]:
    """
    Return unique event IDs with at least one selected hit, restricted to
    windows where window_mask is True.

    This is used to split aggregate files by run while keeping the balance
    metric as unique non-empty selected events.
    """
    channels = data["channels_flat"]
    offsets = data[OFFSETS_KEY]

    evt_run = data["evt_run"]
    evt_subrun = data["evt_subrun"]
    evt_num = data["evt_num"]

    validate_window_arrays(data, npz_path)
    n_windows = int(evt_run.shape[0])

    if window_mask.shape[0] != n_windows:
        raise ValueError(
            f"window_mask has wrong length in {npz_path}: "
            f"{window_mask.shape[0]} vs n_windows={n_windows}"
        )

    # Hit-level mask after channel selection.
    hit_mask = (channels >= CHANNEL_START) & (channels < CHANNEL_END)

    # Convert hit-level mask into per-window/event non-empty mask.
    selected_cumsum = np.empty(hit_mask.shape[0] + 1, dtype=np.int64)
    selected_cumsum[0] = 0
    selected_cumsum[1:] = np.cumsum(hit_mask, dtype=np.int64)

    filtered_offsets = selected_cumsum[offsets]
    hits_per_window = np.diff(filtered_offsets)

    nonempty_mask = (hits_per_window > 0) & window_mask

    event_keys: Set[EventKey] = set()

    for run, subrun, event in zip(
        evt_run[nonempty_mask],
        evt_subrun[nonempty_mask],
        evt_num[nonempty_mask],
    ):
        event_keys.add((int(run), int(subrun), int(event)))

    return event_keys


def get_nonempty_selected_event_keys(data, npz_path: Path) -> Set[EventKey]:
    """
    Return unique event IDs that have at least one hit in the selected channel range.

    Balance unit:
      unique (evt_run, evt_subrun, evt_num)

    This is different from non-empty window count. If one event appears in
    multiple windows, it is counted once.
    """
    evt_run = data["evt_run"]
    window_mask = np.ones(evt_run.shape[0], dtype=bool)

    return get_nonempty_selected_event_keys_for_window_mask(
        data=data,
        npz_path=npz_path,
        window_mask=window_mask,
    )


def count_nonempty_selected_events(data, npz_path: Path) -> int:
    """
    Count unique events that have at least one selected hit after channel filtering.

    This is the count used for good/bad balancing.
    """
    return len(get_nonempty_selected_event_keys(data, npz_path))


def read_file_infos(npz_path: Path) -> List[Dict]:
    """
    Read basic information from one .npz file.

    If the file contains multiple evt_run values, return one virtual info
    dictionary per run. This lets aggregate files such as tpc_data_bad_runs.npz
    be split by run during train/test selection and output writing.

    Each returned info has:
      {
        "path": Path,
        "run": int,
        "n_windows": int,
        "n_balance": int,
        "event_keys_balance": set[(evt_run, evt_subrun, evt_num)],
        "file_number": int,
        "run_filter": int,
      }

    Returns an empty list if the file cannot be used.
    """
    infos: List[Dict] = []

    try:
        with np.load(npz_path, allow_pickle=True) as data:
            required_for_balance = {
                "evt_run",
                "evt_subrun",
                "evt_num",
                OFFSETS_KEY,
                "channels_flat",
            }
            missing_for_balance = required_for_balance - set(data.files)

            if missing_for_balance:
                print(
                    f"[WARNING] Missing keys needed for balance counting in "
                    f"{npz_path}: {sorted(missing_for_balance)}",
                    flush=True,
                )
                return []

            evt_run = data["evt_run"]

            if evt_run.size == 0:
                print(f"[WARNING] Empty evt_run: {npz_path}", flush=True)
                return []

            validate_window_arrays(data, npz_path)

            unique_runs = np.unique(evt_run)
            file_number = extract_last_integer(npz_path)

            if (not SPLIT_AGGREGATE_FILES_BY_RUN) or unique_runs.size == 1:
                # Older faster behavior: one physical file is one input item.
                # If this is an aggregate file and splitting is disabled, the
                # whole file is labeled using evt_run[0].
                if unique_runs.size > 1:
                    print(
                        f"[WARNING] File contains multiple evt_run values, but "
                        f"SPLIT_AGGREGATE_FILES_BY_RUN=False so it will be "
                        f"treated as one physical input: {npz_path}\n"
                        f"          unique evt_run values: {unique_runs.tolist()}\n"
                        f"          run label used: {int(evt_run.flat[0])}",
                        flush=True,
                    )

                event_keys_balance = get_nonempty_selected_event_keys(
                    data=data,
                    npz_path=npz_path,
                )

                infos.append(
                    {
                        "path": npz_path,
                        "run": int(evt_run.flat[0]),
                        "n_windows": int(evt_run.shape[0]),
                        "n_balance": len(event_keys_balance),
                        "event_keys_balance": event_keys_balance,
                        "file_number": file_number,
                        "run_filter": None,
                    }
                )

                return infos

            # Slower but more correct behavior for aggregate files: split by evt_run.
            print(
                f"[INFO] Splitting aggregate file by evt_run: {npz_path}\n"
                f"       unique evt_run values: {unique_runs.tolist()}",
                flush=True,
            )

            for run_value in unique_runs:
                run = int(run_value)
                run_mask = evt_run == run
                n_windows = int(np.count_nonzero(run_mask))

                event_keys_balance = get_nonempty_selected_event_keys_for_window_mask(
                    data=data,
                    npz_path=npz_path,
                    window_mask=run_mask,
                )
                n_balance = len(event_keys_balance)

                infos.append(
                    {
                        "path": npz_path,
                        "run": run,
                        "n_windows": n_windows,
                        "n_balance": n_balance,
                        "event_keys_balance": event_keys_balance,
                        "file_number": file_number,
                        "run_filter": run,
                    }
                )

            return infos

    except Exception as e:
        print(f"[ERROR] Could not read {npz_path}: {e}", flush=True)
        return []


def sort_infos_by_run_then_file(infos: List[Dict]) -> List[Dict]:
    """
    Sort file/run-slice infos so that:

      1. Files from the same run stay together.
      2. Within one run, files are sorted by filename number.
      3. Aggregate-file virtual slices remain deterministic.

    This prevents interleaving two different runs.

    Important:
      run_filter can be None when SPLIT_AGGREGATE_FILES_BY_RUN=False.
      In that case, use x["run"] as the deterministic fallback instead of
      trying int(None).
    """
    def sort_key(x: Dict):
        run_filter = x.get("run_filter")

        if run_filter is None:
            run_filter_for_sort = x["run"]
        else:
            run_filter_for_sort = run_filter

        return (
            int(x["run"]),
            int(x["file_number"]),
            str(x["path"]),
            int(run_filter_for_sort),
        )

    return sorted(infos, key=sort_key)


def sort_run_groups(run_groups: List[Dict]) -> List[Dict]:
    """Sort run groups by run number."""
    return sorted(run_groups, key=lambda group: group["run"])


def group_infos_by_run(infos: List[Dict]) -> List[Dict]:
    """
    Group file-level or virtual file-slice info dictionaries by run.

    Returns a list of run groups:
      {
        "run": int,
        "infos": List[Dict],
        "n_windows": int,
        "n_balance": int,
        "event_keys_balance": set[(evt_run, evt_subrun, evt_num)],
      }

    Each group's infos are sorted by run/file order. Keeping these groups
    together prevents a run from being split between train and test.

    n_windows:
      Total windows in this run.

    n_balance:
      Unique non-empty selected events in this run.
      This is the value used for good/bad balancing.
    """
    groups_by_run = {}

    for info in infos:
        groups_by_run.setdefault(info["run"], []).append(info)

    run_groups = []

    for run, run_infos in groups_by_run.items():
        sorted_run_infos = sort_infos_by_run_then_file(run_infos)

        run_event_keys_balance: Set[EventKey] = set()
        for info in sorted_run_infos:
            run_event_keys_balance.update(info.get("event_keys_balance", set()))

        run_groups.append(
            {
                "run": run,
                "infos": sorted_run_infos,
                "n_windows": sum(info["n_windows"] for info in sorted_run_infos),
                "n_balance": len(run_event_keys_balance),
                "event_keys_balance": run_event_keys_balance,
            }
        )

    return sort_run_groups(run_groups)


def flatten_run_groups(run_groups: List[Dict]) -> List[Dict]:
    """Flatten run groups back to file-level info dictionaries."""
    flattened = []

    for group in sort_run_groups(run_groups):
        flattened.extend(sort_infos_by_run_then_file(group["infos"]))

    return flattened


def shuffle_run_groups(run_groups: List[Dict], random_seed: int) -> List[Dict]:
    """Return a reproducibly shuffled copy of run groups."""
    rng = np.random.default_rng(random_seed)
    shuffled = list(run_groups)
    rng.shuffle(shuffled)
    return shuffled


def balance_sum(run_groups: List[Dict]) -> int:
    """Return total unique non-empty selected event count for a list of run groups."""
    return sum(int(group["n_balance"]) for group in run_groups)


def window_sum(run_groups: List[Dict]) -> int:
    """Return total window/event rows for a list of run groups."""
    return sum(int(group["n_windows"]) for group in run_groups)


def info_identity(info: Dict) -> Tuple[str, int]:
    """
    Return a unique identity for one selected info.

    A physical aggregate file can appear in both train and test as long as the
    run_filter values are different. Therefore, path alone is not a valid
    overlap key.

    If run_filter is None, fall back to the run label.
    """
    run_filter = info.get("run_filter")

    if run_filter is None:
        run_filter_for_identity = info["run"]
    else:
        run_filter_for_identity = run_filter

    return (str(info["path"].resolve()), int(run_filter_for_identity))


def select_run_groups_until_min_balance(
    run_groups: List[Dict],
    min_balance: int,
    random_seed: int,
) -> List[Dict]:
    """
    Select whole runs until at least min_balance unique non-empty selected events
    are reached.

    The random choice happens at the run level, not at the file level.
    Therefore, all files from a selected run stay together.

    Runs with n_balance <= 0 are skipped because they do not contribute any
    unique non-empty selected events to the balancing target.
    """
    min_balance = int(min_balance)

    if min_balance <= 0:
        raise ValueError(f"min_balance must be positive or None, got {min_balance}")

    selected = []
    selected_balance = 0

    for group in shuffle_run_groups(run_groups, random_seed):
        group_balance = int(group["n_balance"])

        if group_balance <= 0:
            continue

        selected.append(group)
        selected_balance += group_balance

        if selected_balance >= min_balance:
            break

    if selected_balance < min_balance:
        raise ValueError(
            f"Could not reserve the requested minimum training balance count. "
            f"Requested MIN_TRAIN_EVENT_N={min_balance}, "
            f"but only {selected_balance} unique non-empty selected good-run events "
            f"are available."
        )

    return selected


def select_run_groups_for_balance_target(
    run_groups: List[Dict],
    target_balance: int,
    random_seed: int,
) -> List[Dict]:
    """
    Select whole runs until reaching target_balance unique non-empty selected events.

    For test balancing, using a random order can accidentally select one huge run
    when a smaller run would be much closer to a tiny target. Therefore this
    function chooses smaller available runs first. This still keeps whole runs
    together; it does not slice a run to hit the target exactly.

    random_seed is kept in the signature for compatibility with older calls.
    """
    del random_seed

    target_balance = int(target_balance)

    if target_balance <= 0:
        return []

    usable_groups = [
        group
        for group in run_groups
        if int(group["n_balance"]) > 0
    ]

    usable_groups = sorted(
        usable_groups,
        key=lambda group: (
            int(group["n_balance"]),
            int(group["run"]),
        ),
    )

    selected = []
    selected_balance = 0

    for group in usable_groups:
        selected.append(group)
        selected_balance += int(group["n_balance"])

        if selected_balance >= target_balance:
            break

    return selected


def select_run_groups_up_to_max_balance(
    run_groups: List[Dict],
    max_balance: Optional[int],
) -> List[Dict]:
    """
    Select whole runs up to an optional maximum unique non-empty selected event cap.

    If max_balance is None, all run groups are returned.

    If max_balance is set, runs are added in sorted order only when adding the
    full run would not exceed the cap. This avoids slicing through a run.

    The cap is based on n_balance, not n_windows.
    """
    if max_balance is None:
        return sort_run_groups(run_groups)

    max_balance = int(max_balance)

    if max_balance <= 0:
        raise ValueError(
            f"MAX_TRAIN_EVENT_N must be positive or None, got {max_balance}"
        )

    selected = []
    selected_balance = 0

    for group in sort_run_groups(run_groups):
        group_balance = int(group["n_balance"])

        if group_balance <= 0:
            continue

        if selected_balance + group_balance > max_balance:
            continue

        selected.append(group)
        selected_balance += group_balance

    return selected


def add_extra_run_groups_up_to_max_balance(
    base_groups: List[Dict],
    extra_groups: List[Dict],
    max_balance: Optional[int],
) -> List[Dict]:
    """
    Start with required whole-run groups, then optionally add more whole runs.

    This is used when MIN_TRAIN_EVENT_N is enabled. The base groups are the
    required training runs. Extra good runs are added only if they do not make
    the training set exceed MAX_TRAIN_EVENT_N.

    The cap is based on n_balance, not n_windows.
    """
    selected = sort_run_groups(base_groups)

    if max_balance is None:
        return sort_run_groups(selected + extra_groups)

    max_balance = int(max_balance)

    if max_balance <= 0:
        raise ValueError(
            f"MAX_TRAIN_EVENT_N must be positive or None, got {max_balance}"
        )

    selected_balance = balance_sum(selected)

    if selected_balance > max_balance:
        raise ValueError(
            f"The required whole-run training selection has balance count "
            f"{selected_balance}, which is larger than "
            f"MAX_TRAIN_EVENT_N={max_balance}. "
            f"The script will not slice through a run. Increase "
            f"MAX_TRAIN_EVENT_N, lower MIN_TRAIN_EVENT_N, or set "
            f"MAX_TRAIN_EVENT_N=None."
        )

    selected_runs = {group["run"] for group in selected}

    for group in sort_run_groups(extra_groups):
        if group["run"] in selected_runs:
            continue

        group_balance = int(group["n_balance"])

        if group_balance <= 0:
            continue

        if selected_balance + group_balance > max_balance:
            continue

        selected.append(group)
        selected_runs.add(group["run"])
        selected_balance += group_balance

    return sort_run_groups(selected)


def validate_channel_settings() -> None:
    """Check that the channel range is valid."""
    if CHANNEL_START < 0:
        raise ValueError(f"CHANNEL_START must be >= 0, got {CHANNEL_START}")

    if CHANNEL_END <= CHANNEL_START:
        raise ValueError(
            f"CHANNEL_END must be larger than CHANNEL_START, "
            f"got CHANNEL_START={CHANNEL_START}, CHANNEL_END={CHANNEL_END}"
        )


def filter_one_file_by_channel_range(data, npz_path: Path):
    """
    Fast full-file channel filter.

    Keeps all windows/events, even if they become empty after channel filtering.
    Filters flat arrays by CHANNEL_START <= channels_flat < CHANNEL_END.
    Rebuilds offsets after hit filtering.

    Returns:
      filtered_flat_parts:
        dict of filtered flat arrays for all windows in this file

      new_offsets:
        offsets after filtering, with length n_windows + 1

      n_kept_hits:
        total number of selected flat entries in this file
    """
    validate_window_arrays(data, npz_path)

    channels = data["channels_flat"]
    offsets = data[OFFSETS_KEY]

    hit_mask = (channels >= CHANNEL_START) & (channels < CHANNEL_END)

    selected_cumsum = np.empty(hit_mask.shape[0] + 1, dtype=np.int64)
    selected_cumsum[0] = 0
    selected_cumsum[1:] = np.cumsum(hit_mask, dtype=np.int64)

    new_offsets = selected_cumsum[offsets].astype(np.int64, copy=False)
    n_kept_hits = int(new_offsets[-1])

    filtered_flat_parts = {}

    for key in FLAT_KEYS:
        selected = data[key][hit_mask]

        if key == "channels_flat" and REINDEX_CHANNELS:
            selected = selected - CHANNEL_START

        filtered_flat_parts[key] = selected

    return filtered_flat_parts, new_offsets, n_kept_hits


def filter_one_file_by_channel_range_and_window_indices(
    data,
    npz_path: Path,
    window_indices: np.ndarray,
):
    """
    Filter one file by both:
      1. selected window indices
      2. selected channel range

    Keeps the selected windows even if they become empty after channel filtering.

    Important optimization:
      If an aggregate .npz is being split by run, selected windows may be only a
      tiny slice of a huge file. In that case, do NOT build a full-length flat
      mask or a full-length cumsum over the whole aggregate file. Instead, only
      inspect the selected windows' flat ranges.

    Returns:
      filtered_flat_parts:
        dict of filtered flat arrays for selected windows

      new_offsets:
        offsets after filtering, with length len(window_indices) + 1

      n_kept_hits:
        total number of selected flat entries in this file/run slice
    """
    validate_window_arrays(data, npz_path)

    evt_run = data["evt_run"]
    offsets = data[OFFSETS_KEY]
    n_windows = int(evt_run.shape[0])

    window_indices = np.asarray(window_indices, dtype=np.int64)

    if window_indices.ndim != 1:
        raise ValueError(f"window_indices must be 1-D for {npz_path}")

    if window_indices.size == 0:
        return {key: np.asarray([]) for key in FLAT_KEYS}, np.asarray([0], dtype=np.int64), 0

    if int(window_indices.min()) < 0 or int(window_indices.max()) >= n_windows:
        raise ValueError(
            f"window_indices out of range for {npz_path}: "
            f"min={int(window_indices.min())}, max={int(window_indices.max())}, "
            f"n_windows={n_windows}"
        )

    # Fast path: if this item is the complete file in original order, use the
    # simple full-file vectorized filter. This keeps normal single-run files fast.
    if (
        window_indices.size == n_windows
        and int(window_indices[0]) == 0
        and int(window_indices[-1]) == n_windows - 1
        and np.all(window_indices == np.arange(n_windows, dtype=np.int64))
    ):
        return filter_one_file_by_channel_range(data, npz_path)

    # Split-run path: only process the selected windows. This is much faster for
    # aggregate files like tpc_data_bad_runs.npz when we only need run 19627.
    channels = data["channels_flat"]
    flat_arrays = {key: data[key] for key in FLAT_KEYS}

    filtered_chunks = {key: [] for key in FLAT_KEYS}
    kept_counts = np.empty(window_indices.size, dtype=np.int64)

    for out_i, win_idx_raw in enumerate(window_indices):
        win_idx = int(win_idx_raw)
        start = int(offsets[win_idx])
        end = int(offsets[win_idx + 1])

        if end <= start:
            kept_counts[out_i] = 0
            continue

        local_channels = channels[start:end]
        local_hit_mask = (local_channels >= CHANNEL_START) & (local_channels < CHANNEL_END)
        n_kept = int(np.count_nonzero(local_hit_mask))
        kept_counts[out_i] = n_kept

        if n_kept == 0:
            continue

        for key in FLAT_KEYS:
            selected = flat_arrays[key][start:end][local_hit_mask]

            if key == "channels_flat" and REINDEX_CHANNELS:
                selected = selected - CHANNEL_START

            filtered_chunks[key].append(selected)

    filtered_flat_parts = {}
    for key in FLAT_KEYS:
        if filtered_chunks[key]:
            filtered_flat_parts[key] = np.concatenate(filtered_chunks[key], axis=0)
        else:
            # Preserve dtype where possible.
            filtered_flat_parts[key] = flat_arrays[key][:0]

    new_offsets = np.empty(window_indices.size + 1, dtype=np.int64)
    new_offsets[0] = 0
    new_offsets[1:] = np.cumsum(kept_counts, dtype=np.int64)

    n_kept_hits = int(new_offsets[-1])

    return filtered_flat_parts, new_offsets, n_kept_hits

def concatenate_npz_items(
    input_items: List[Dict],
    output_path: Path,
    max_windows: Optional[int] = None,
) -> None:
    """
    Concatenate sparse window/event .npz files or virtual run-slices into one
    .npz file.

    Important handling:
      - if an input item has run_filter, only windows with evt_run == run_filter
        are written from that physical .npz file
      - flat arrays are filtered by channel range before concatenation
      - windows/events with zero selected hits are still kept
      - per-window arrays are filtered by run_filter but NOT by channel selection
      - offsets are rebuilt after hit filtering
      - evt_file_idx is shifted according to concatenated filenames
      - n_channels is set to CHANNEL_END - CHANNEL_START if REINDEX_CHANNELS=True

    max_windows:
      If None, write all selected windows/events from input_items.
      If an integer, write exactly up to max_windows windows/events.
      If the limit falls inside one item, only the first needed selected windows
      from that item are written.

    Note:
      In normal train/test construction, this script does NOT pass
      MAX_TRAIN_EVENT_N here. The train/test selections above already enforce
      the training cap using whole runs, so this function should usually write
      the complete selected items.
    """
    if not input_items:
        raise ValueError(f"No input items provided for {output_path}")

    if max_windows is not None:
        max_windows = int(max_windows)
        if max_windows <= 0:
            raise ValueError(f"max_windows must be positive or None, got {max_windows}")

    flat_parts = {key: [] for key in FLAT_KEYS}
    per_window_parts = {key: [] for key in PER_WINDOW_KEYS}

    combined_offsets = [0]
    combined_filenames = []

    total_flat_length = 0
    filename_offset = 0
    original_n_channels_value = None

    total_windows = 0
    total_kept_hits = 0
    total_nonempty_event_keys_after_filtering: Set[EventKey] = set()

    for i_item, item in enumerate(input_items, start=1):
        if max_windows is not None and total_windows >= max_windows:
            break

        npz_path = item["path"]
        run_filter = item.get("run_filter", None)

        print(
            f"[{output_path.name}] reading/filtering "
            f"{i_item}/{len(input_items)}: {npz_path}"
            f" run_filter={run_filter}",
            flush=True,
        )

        with np.load(npz_path, allow_pickle=True) as data:
            keys = set(data.files)

            required_keys = set(
                FLAT_KEYS
                + PER_WINDOW_KEYS
                + [OFFSETS_KEY, N_CHANNELS_KEY]
            )
            missing_keys = required_keys - keys

            if missing_keys:
                raise KeyError(
                    f"{npz_path} is missing required keys: {sorted(missing_keys)}"
                )

            evt_run = data["evt_run"]
            evt_subrun = data["evt_subrun"]
            evt_num = data["evt_num"]

            if evt_run.size == 0:
                raise ValueError(f"Empty evt_run in {npz_path}")

            validate_window_arrays(data, npz_path)
            n_windows_in_file = int(evt_run.shape[0])

            if run_filter is None:
                window_indices = np.arange(n_windows_in_file, dtype=np.int64)
            else:
                window_indices = np.flatnonzero(evt_run == int(run_filter))

            if window_indices.size == 0:
                print(
                    f"[WARNING] No windows found for run_filter={run_filter} "
                    f"in {npz_path}; skipping this item.",
                    flush=True,
                )
                continue

            # Decide how many selected windows to take from this item.
            # If max_windows is None, take the full item. If max_windows is set,
            # take only what is needed to reach the limit.
            if max_windows is None:
                selected_window_indices = window_indices
            else:
                remaining_windows = max_windows - total_windows
                selected_window_indices = window_indices[:remaining_windows]

            n_windows_to_take = int(selected_window_indices.shape[0])

            if n_windows_to_take <= 0:
                break

            # Original files may have slightly different n_channels if some
            # channels were not recorded. Do not require consistency.
            n_channels_current = int(np.asarray(data[N_CHANNELS_KEY]).flat[0])

            if original_n_channels_value is None:
                original_n_channels_value = n_channels_current

            if CHANNEL_START >= n_channels_current:
                print(
                    f"[WARNING] CHANNEL_START={CHANNEL_START} is outside "
                    f"n_channels={n_channels_current} for {npz_path}. "
                    f"This item may contribute no hits in the selected range.",
                    flush=True,
                )

            print(
                f"[{output_path.name}] selected windows from this item: "
                f"{n_windows_to_take}/{n_windows_in_file}",
                flush=True,
            )

            (
                filtered_flat_parts,
                filtered_offsets,
                n_kept_hits,
            ) = filter_one_file_by_channel_range_and_window_indices(
                data=data,
                npz_path=npz_path,
                window_indices=selected_window_indices,
            )

            for key in FLAT_KEYS:
                flat_parts[key].append(filtered_flat_parts[key])

            hits_per_taken_window = np.diff(filtered_offsets)
            nonempty_mask = hits_per_taken_window > 0

            evt_run_taken = evt_run[selected_window_indices]
            evt_subrun_taken = evt_subrun[selected_window_indices]
            evt_num_taken = evt_num[selected_window_indices]

            for run, subrun, event in zip(
                evt_run_taken[nonempty_mask],
                evt_subrun_taken[nonempty_mask],
                evt_num_taken[nonempty_mask],
            ):
                total_nonempty_event_keys_after_filtering.add(
                    (int(run), int(subrun), int(event))
                )

            # Shift and append offsets.
            shifted_offsets = filtered_offsets[1:] + total_flat_length
            combined_offsets.extend(shifted_offsets.tolist())

            total_flat_length += n_kept_hits
            total_kept_hits += n_kept_hits

            # Concatenate per-window arrays.
            # These are filtered by selected_window_indices, but not by channel.
            for key in PER_WINDOW_KEYS:
                arr = data[key]

                if arr.shape[0] != n_windows_in_file:
                    raise ValueError(
                        f"Per-window key {key} has wrong length in {npz_path}: "
                        f"{arr.shape[0]} vs n_windows={n_windows_in_file}"
                    )

                arr = arr[selected_window_indices]

                if key == "evt_file_idx":
                    # evt_file_idx points into the filenames array.
                    # Since we concatenate filenames, shift indices.
                    arr = arr + filename_offset

                per_window_parts[key].append(arr)

            # Concatenate filenames.
            # We keep the full filenames array from any item that contributes at
            # least one selected window. If the same physical aggregate file is
            # used for multiple virtual runs, its filename list can appear more
            # than once; evt_file_idx is shifted so references remain correct.
            if FILENAMES_KEY in data.files:
                filenames = data[FILENAMES_KEY]
                combined_filenames.extend(filenames.tolist())
                filename_offset += len(filenames)
            else:
                combined_filenames.append(str(npz_path))
                filename_offset += 1

            total_windows += n_windows_to_take

    if total_windows == 0:
        raise ValueError(
            f"No windows/events were written for {output_path}. "
            f"Check input_items and evt_run arrays."
        )

    if max_windows is not None and total_windows != max_windows:
        raise ValueError(
            f"Could not write requested max_windows={max_windows} for {output_path}. "
            f"Only {total_windows} windows/events were available."
        )

    print(f"[{output_path.name}] concatenating arrays...", flush=True)

    output = {}

    for key in FLAT_KEYS:
        if flat_parts[key]:
            output[key] = np.concatenate(flat_parts[key], axis=0)
        else:
            output[key] = np.asarray([])

    for key in PER_WINDOW_KEYS:
        if per_window_parts[key]:
            output[key] = np.concatenate(per_window_parts[key], axis=0)
        else:
            output[key] = np.asarray([])

    output[OFFSETS_KEY] = np.asarray(combined_offsets, dtype=np.int64)

    if REINDEX_CHANNELS:
        output[N_CHANNELS_KEY] = np.asarray(CHANNEL_END - CHANNEL_START)
    else:
        output[N_CHANNELS_KEY] = np.asarray(original_n_channels_value)

    output[FILENAMES_KEY] = np.asarray([str(f) for f in combined_filenames], dtype=str)

    # Final consistency checks
    if output["evt_run"].shape[0] != total_windows:
        raise RuntimeError(
            f"Internal error for {output_path}: "
            f"evt_run length {output['evt_run'].shape[0]} != total_windows "
            f"{total_windows}"
        )

    if output[OFFSETS_KEY].shape[0] != total_windows + 1:
        raise RuntimeError(
            f"Internal error for {output_path}: "
            f"offsets length {output[OFFSETS_KEY].shape[0]} != total_windows + 1 "
            f"{total_windows + 1}"
        )

    if output[OFFSETS_KEY][-1] != output["channels_flat"].shape[0]:
        raise RuntimeError(
            f"Internal error for {output_path}: "
            f"last offset {output[OFFSETS_KEY][-1]} != "
            f"channels_flat length {output['channels_flat'].shape[0]}"
        )

    # Zero-hit windows/events are allowed.
    # Therefore, do NOT require np.diff(offsets) > 0.

    if output["channels_flat"].size > 0:
        min_channel = int(output["channels_flat"].min())
        max_channel = int(output["channels_flat"].max())

        if REINDEX_CHANNELS:
            if min_channel < 0 or max_channel >= CHANNEL_END - CHANNEL_START:
                raise RuntimeError(
                    f"Reindexed channel range error in {output_path}: "
                    f"min={min_channel}, max={max_channel}, "
                    f"expected [0, {CHANNEL_END - CHANNEL_START})"
                )
        else:
            if min_channel < CHANNEL_START or max_channel >= CHANNEL_END:
                raise RuntimeError(
                    f"Channel range error in {output_path}: "
                    f"min={min_channel}, max={max_channel}, "
                    f"expected [{CHANNEL_START}, {CHANNEL_END})"
                )

    print(f"[{output_path.name}] saving to disk...", flush=True)

    if COMPRESS_OUTPUT:
        np.savez_compressed(output_path, **output)
    else:
        np.savez(output_path, **output)

    print(f"[SAVED] {output_path}", flush=True)
    print(f"        windows written: {total_windows}", flush=True)
    print(
        "        unique non-empty selected events: "
        f"{len(total_nonempty_event_keys_after_filtering)}",
        flush=True,
    )
    print(f"        selected flat entries: {total_kept_hits}", flush=True)
    print(f"        n_channels: {int(output[N_CHANNELS_KEY])}", flush=True)
    print(f"        channel range: [{CHANNEL_START}, {CHANNEL_END})", flush=True)
    print(f"        reindex channels: {REINDEX_CHANNELS}", flush=True)
    print(f"        compressed output: {COMPRESS_OUTPUT}", flush=True)
    print(f"        max_windows limit: {max_windows}", flush=True)


def main() -> None:
    validate_channel_settings()

    OUTPUT_PARENT_PATH.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Runtime settings")
    print("=" * 80)
    print(f"MIN_TRAIN_EVENT_N = {MIN_TRAIN_EVENT_N}")
    print(f"MAX_TRAIN_EVENT_N = {MAX_TRAIN_EVENT_N}")
    print(f"RANDOM_SEED = {RANDOM_SEED}")
    print(f"SPLIT_AGGREGATE_FILES_BY_RUN = {SPLIT_AGGREGATE_FILES_BY_RUN}")
    print("=" * 80)

    npz_files = get_npz_files()

    all_infos = []
    skipped_files = []

    for npz_path in npz_files:
        infos = read_file_infos(npz_path)

        if not infos:
            skipped_files.append(npz_path)
        else:
            all_infos.extend(infos)

    good_infos = []
    bad_infos = []

    for info in all_infos:
        run = info["run"]

        if run in BAD_RUNS:
            bad_infos.append(info)
        else:
            # Default behavior:
            # any run not explicitly listed as bad is treated as good.
            good_infos.append(info)

    explicit_good_infos = [
        info
        for info in good_infos
        if info["run"] in GOOD_RUNS
    ]

    default_good_infos = [
        info
        for info in good_infos
        if info["run"] not in GOOD_RUNS
    ]

    good_run_groups = group_infos_by_run(good_infos)
    bad_run_groups = group_infos_by_run(bad_infos)
    explicit_good_run_groups = group_infos_by_run(explicit_good_infos)
    default_good_run_groups = group_infos_by_run(default_good_infos)

    total_good_windows = window_sum(good_run_groups)
    total_bad_windows = window_sum(bad_run_groups)

    total_good_balance = balance_sum(good_run_groups)
    total_bad_balance = balance_sum(bad_run_groups)

    total_default_good_windows = window_sum(default_good_run_groups)
    total_default_good_balance = balance_sum(default_good_run_groups)

    if MIN_TRAIN_EVENT_N is None:
        print("[DEBUG] Running MIN_TRAIN_EVENT_N is None branch")

        # Original behavior, but now whole-run based and balanced by n_balance:
        #   - all bad runs go to test
        #   - select enough whole good runs to approximately match the total bad
        #     unique non-empty selected event count
        #   - remaining good runs go to train

        available_good_balance_total = balance_sum(good_run_groups)
        available_bad_balance_total = balance_sum(bad_run_groups)

        print("-" * 80)
        print("Pre-selection unique non-empty event balance counts")
        print("-" * 80)
        print(
            "Available good unique non-empty selected events: "
            f"{available_good_balance_total}"
        )
        print(
            "Available bad unique non-empty selected events: "
            f"{available_bad_balance_total}"
        )
        print(
            "Minimum training balance setting is None, so no minimum good "
            "training reserve is applied."
        )
        print("-" * 80)

        selected_bad_run_groups = [
            group
            for group in bad_run_groups
            if int(group["n_balance"]) > 0
        ]

        selected_good_run_groups = select_run_groups_for_balance_target(
            run_groups=good_run_groups,
            target_balance=total_bad_balance,
            random_seed=RANDOM_SEED,
        )

        selected_good_runs = {group["run"] for group in selected_good_run_groups}

        train_candidate_run_groups = [
            group
            for group in good_run_groups
            if group["run"] not in selected_good_runs
        ]

        train_run_groups = select_run_groups_up_to_max_balance(
            run_groups=train_candidate_run_groups,
            max_balance=MAX_TRAIN_EVENT_N,
        )

        remaining_good_balance_after_train_reserve = None
        test_target_balance_per_class = None

    else:
        print("[DEBUG] Running MIN_TRAIN_EVENT_N enabled branch")

        # New behavior, whole-run based and balanced by n_balance:
        #   1. Reserve whole good runs for training until MIN_TRAIN_EVENT_N
        #      unique non-empty selected events are reached.
        #   2. Build the test set only from the remaining whole good runs and
        #      whole bad runs.
        #   3. Use the lower of remaining-good balance count and available-bad
        #      balance count as the per-class test target.
        #
        # No run is split between train and test. If MAX_TRAIN_EVENT_N is
        # set, it is enforced by selecting whole runs only, not by slicing.

        available_good_balance_for_training = balance_sum(good_run_groups)
        available_bad_balance_total = balance_sum(bad_run_groups)

        print("-" * 80)
        print("Pre-selection unique non-empty event balance counts")
        print("-" * 80)
        print(
            "Available good unique non-empty selected events: "
            f"{available_good_balance_for_training}"
        )
        print(
            "Available bad unique non-empty selected events: "
            f"{available_bad_balance_total}"
        )
        print(
            "Requested minimum training good unique non-empty selected events: "
            f"{MIN_TRAIN_EVENT_N}"
        )
        print("-" * 80)

        if MAX_TRAIN_EVENT_N is not None and MAX_TRAIN_EVENT_N < MIN_TRAIN_EVENT_N:
            raise ValueError(
                f"MAX_TRAIN_EVENT_N={MAX_TRAIN_EVENT_N} is smaller than "
                f"MIN_TRAIN_EVENT_N={MIN_TRAIN_EVENT_N}. "
                f"This would cap the written training set below the requested minimum."
            )

        reserved_train_good_run_groups = select_run_groups_until_min_balance(
            run_groups=good_run_groups,
            min_balance=MIN_TRAIN_EVENT_N,
            random_seed=RANDOM_SEED,
        )

        reserved_train_good_runs = {
            group["run"]
            for group in reserved_train_good_run_groups
        }

        remaining_good_run_groups = [
            group
            for group in good_run_groups
            if group["run"] not in reserved_train_good_runs
        ]

        remaining_good_balance_after_train_reserve = balance_sum(
            remaining_good_run_groups
        )

        available_bad_balance = balance_sum(bad_run_groups)

        test_target_balance_per_class = min(
            remaining_good_balance_after_train_reserve,
            available_bad_balance,
        )

        if test_target_balance_per_class <= 0:
            raise ValueError(
                "Could not build a balanced test set after reserving the minimum "
                "training balance count. Either no unique non-empty selected good "
                "events remain for test, or no unique non-empty selected bad "
                "events are available."
            )

        selected_good_run_groups = select_run_groups_for_balance_target(
            run_groups=remaining_good_run_groups,
            target_balance=test_target_balance_per_class,
            random_seed=RANDOM_SEED + 1,
        )

        selected_bad_run_groups = select_run_groups_for_balance_target(
            run_groups=bad_run_groups,
            target_balance=test_target_balance_per_class,
            random_seed=RANDOM_SEED + 2,
        )

        selected_good_runs = {
            group["run"]
            for group in selected_good_run_groups
        }

        extra_train_good_run_groups = [
            group
            for group in remaining_good_run_groups
            if group["run"] not in selected_good_runs
        ]

        train_run_groups = add_extra_run_groups_up_to_max_balance(
            base_groups=reserved_train_good_run_groups,
            extra_groups=extra_train_good_run_groups,
            max_balance=MAX_TRAIN_EVENT_N,
        )

    # Sort within each class, but do NOT globally sort the test set.
    #
    # This keeps the test output grouped as:
    #   1. all selected good runs
    #   2. all selected bad runs
    #
    # Within each class, runs are sorted and files inside each run are ordered.
    selected_good_run_groups = sort_run_groups(selected_good_run_groups)
    selected_bad_run_groups = sort_run_groups(selected_bad_run_groups)
    train_run_groups = sort_run_groups(train_run_groups)

    selected_good_infos = sort_infos_by_run_then_file(
        flatten_run_groups(selected_good_run_groups)
    )
    selected_bad_infos = sort_infos_by_run_then_file(
        flatten_run_groups(selected_bad_run_groups)
    )

    # GraphVAE evaluation keeps good and bad test samples in separate files.
    # Selection rules are unchanged; only the output layout changes.
    good_test_infos = selected_good_infos
    bad_test_infos = selected_bad_infos
    all_test_infos = good_test_infos + bad_test_infos

    # Train contains only good runs, so normal sorting is fine.
    train_infos = sort_infos_by_run_then_file(flatten_run_groups(train_run_groups))

    selected_good_windows = window_sum(selected_good_run_groups)
    selected_good_balance = balance_sum(selected_good_run_groups)

    test_bad_windows = window_sum(selected_bad_run_groups)
    test_bad_balance = balance_sum(selected_bad_run_groups)

    test_total_windows = selected_good_windows + test_bad_windows
    test_total_balance = selected_good_balance + test_bad_balance

    train_total_windows = window_sum(train_run_groups)
    train_total_balance = balance_sum(train_run_groups)

    # Safety check: no overlap between train and test.
    # Use (path, run_filter), not path alone, because one aggregate physical file
    # can validly contribute different runs to different outputs.
    test_ids = {info_identity(info) for info in all_test_infos}
    train_ids = {info_identity(info) for info in train_infos}
    overlap = test_ids & train_ids
    if overlap:
        raise RuntimeError(
            "Some file/run slices are in both train and test. This should never happen:\n"
            + "\n".join(str(item) for item in sorted(overlap))
        )

    # Print summary before writing
    print("=" * 80)
    print("Input summary")
    print("=" * 80)
    print(f"Directory: {NPZ_DIR}")
    print(f"Recursive search: {RECURSIVE}")
    print(f"Total usable file/run entries: {len(all_infos)}")
    print(f"Skipped/error files: {len(skipped_files)}")
    print("-" * 80)
    print(f"Good-run file/run entries: {len(good_infos)}")
    print(f"Good-run total windows before channel filtering: {total_good_windows}")
    print(f"Good-run unique non-empty selected events: {total_good_balance}")
    print(f"Bad-run file/run entries: {len(bad_infos)}")
    print(f"Bad-run total windows before channel filtering: {total_bad_windows}")
    print(f"Bad-run unique non-empty selected events: {total_bad_balance}")
    print(f"Explicit good-run file/run entries: {len(explicit_good_infos)}")
    print(f"Default good-run file/run entries: {len(default_good_infos)}")
    print(
        "Default good-run total windows before channel filtering: "
        f"{total_default_good_windows}"
    )
    print(
        "Default good-run unique non-empty selected events: "
        f"{total_default_good_balance}"
    )
    print("-" * 80)
    print("Channel selection")
    print("-" * 80)
    print(f"Channel start: {CHANNEL_START}")
    print(f"Channel end: {CHANNEL_END}")
    print(f"Number of selected channels: {CHANNEL_END - CHANNEL_START}")
    print(f"Reindex channels: {REINDEX_CHANNELS}")
    print("-" * 80)
    print("Train/test sizing settings")
    print("-" * 80)
    print(f"Minimum training balance setting: {MIN_TRAIN_EVENT_N}")
    print(f"Max training balance output cap: {MAX_TRAIN_EVENT_N}")
    print("Balance metric: unique non-empty selected events after channel filtering")
    if SPLIT_AGGREGATE_FILES_BY_RUN:
        print("Aggregate files: split by evt_run into virtual per-run inputs")
    else:
        print("Aggregate files: NOT split; older faster physical-file mode")

    if test_target_balance_per_class is not None:
        print(
            "Remaining good unique non-empty selected events after training reserve: "
            f"{remaining_good_balance_after_train_reserve}"
        )
        print(f"Test target balance per class: {test_target_balance_per_class}")

    print("-" * 80)
    print("Test selection before channel filtering")
    print("-" * 80)
    print("Test outputs are split by class: good_test.npz and bad_test.npz")
    print(f"Bad file/run entries selected for test: {len(selected_bad_infos)}")
    print(f"Bad total windows selected for test before filtering: {test_bad_windows}")
    print(f"Bad unique non-empty selected events for test: {test_bad_balance}")
    print(f"Good file/run entries selected for test: {len(selected_good_infos)}")
    print(f"Good total windows selected for test before filtering: {selected_good_windows}")
    print(f"Good unique non-empty selected events for test: {selected_good_balance}")
    print(f"Total test file/run entries: {len(all_test_infos)}")
    print(f"Total test windows before filtering: {test_total_windows}")
    print(f"Total test unique non-empty selected events: {test_total_balance}")
    print("-" * 80)
    print("Train selection before channel filtering")
    print("-" * 80)
    print(f"Total train file/run entries: {len(train_infos)}")
    print(f"Total train windows before filtering: {train_total_windows}")
    print(f"Total train unique non-empty selected events: {train_total_balance}")
    print("=" * 80)

    if PRINT_FILES:
        print("\nGood-test files/run slices:")
        for info in good_test_infos:
            label = "GOOD" if info["run"] in GOOD_RUNS else "GOOD_DEFAULT"

            print(
                f"  [{label}] run={info['run']} "
                f"run_filter={info.get('run_filter')} "
                f"windows_before_filtering={info['n_windows']} "
                f"unique_nonempty_selected_events={info['n_balance']} "
                f"file_number={info['file_number']} "
                f"path={info['path']}"
            )

        print("\nBad-test files/run slices:")
        for info in bad_test_infos:
            label = "BAD"

            print(
                f"  [{label}] run={info['run']} "
                f"run_filter={info.get('run_filter')} "
                f"windows_before_filtering={info['n_windows']} "
                f"unique_nonempty_selected_events={info['n_balance']} "
                f"file_number={info['file_number']} "
                f"path={info['path']}"
            )

        print("\nTrain files/run slices:")
        for info in train_infos:
            if info["run"] in BAD_RUNS:
                label = "BAD"
            elif info["run"] in GOOD_RUNS:
                label = "GOOD"
            else:
                label = "GOOD_DEFAULT"

            print(
                f"  [{label}] run={info['run']} "
                f"run_filter={info.get('run_filter')} "
                f"windows_before_filtering={info['n_windows']} "
                f"unique_nonempty_selected_events={info['n_balance']} "
                f"file_number={info['file_number']} "
                f"path={info['path']}"
            )

    # Write output files
    print("\n" + "=" * 80)
    print("Writing output files")
    print("=" * 80)

    # Do not pass MAX_TRAIN_EVENT_N into concatenate_npz_items here.
    # The train/test selections above already enforce the training cap using
    # whole runs. Passing max_windows here could slice through a run/file.
    concatenate_npz_items(train_infos, TRAIN_OUTPUT_PATH)
    concatenate_npz_items(good_test_infos, GOOD_TEST_OUTPUT_PATH)
    concatenate_npz_items(bad_test_infos, BAD_TEST_OUTPUT_PATH)

    print("=" * 80)
    print("Done")
    print("=" * 80)
    print(f"Train output:     {TRAIN_OUTPUT_PATH}")
    print(f"Good-test output: {GOOD_TEST_OUTPUT_PATH}")
    print(f"Bad-test output:  {BAD_TEST_OUTPUT_PATH}")


if __name__ == "__main__":
    main()