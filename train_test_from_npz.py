#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Optional
import re

import numpy as np


# ============================================================
# User settings
# ============================================================

INPUT_DIR = Path(
    "/exp/sbnd/data/users/micarrig/DQM/tpc_data_v3"
)

OUTPUT_DIR = Path(
    "/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection/scratch"
)

WINDOWS_TRAIN_PATH = OUTPUT_DIR / "windows_train.npz"
GOOD_RUNS_PATH = OUTPUT_DIR / "good_runs.npz"
BAD_RUNS_PATH = OUTPUT_DIR / "bad_runs.npz"

# Fraction of all eligible events assigned to good_runs.npz.
# The remaining events are assigned to windows_train.npz.
GOOD_RUNS_FRACTION = 0.75

# File numbers to exclude completely before splitting.
BAD_FILE_NUMBERS = {
    830,
    20173,
    20614,
    20615,
    20620,
    20621,
    # 20769,
}

FILE_PATTERN = "tpc_data_v3_*.npz"

# Run number whose source files should be reported.
TARGET_RUN = 20142


# ============================================================
# NPZ key categories
# ============================================================

# One value per hit/channel entry. Their first dimension must equal
# offsets[-1].
FLAT_KEYS = {
    "channels_flat",
    "integrals_flat",
    "times_flat",
    "wires_flat",
    "planes_flat",
    "tpcs_flat",
    "widths_flat",
    "sumadcs_flat",
    "mults_flat",
    "hassps_flat",
}

# One value per event. Their first dimension must equal len(offsets) - 1.
EVENT_KEYS = {
    "evt_run",
    "evt_subrun",
    "evt_num",
    "evt_time",
    "evt_file_idx",
}

# Scalar metadata copied into both outputs after consistency checks.
SCALAR_KEYS = {
    "n_channels",
}


# ============================================================
# Helper functions
# ============================================================

def extract_file_number(path: Path) -> int:
    """Extract the numeric suffix from ``tpc_data_v3_20516.npz``."""
    match = re.fullmatch(r"tpc_data_v3_(\d+)\.npz", path.name)

    if match is None:
        raise ValueError(
            f"Filename does not match the expected format: {path.name}"
        )

    return int(match.group(1))


def find_input_files() -> tuple[list[Path], list[Path]]:
    """
    Find source NPZ files and divide them into good-candidate and bad files.

    Files whose numeric suffix is listed in BAD_FILE_NUMBERS are excluded from
    the good/train split and concatenated into bad_runs.npz.
    """
    good_candidates: list[tuple[int, Path]] = []
    bad_candidates: list[tuple[int, Path]] = []

    for path in INPUT_DIR.glob(FILE_PATTERN):
        try:
            file_number = extract_file_number(path)
        except ValueError:
            print(f"Skipping unexpected filename: {path.name}")
            continue

        if file_number in BAD_FILE_NUMBERS:
            bad_candidates.append((file_number, path))
        else:
            good_candidates.append((file_number, path))

    good_candidates.sort(key=lambda item: item[0])
    bad_candidates.sort(key=lambda item: item[0])

    good_files = [path for _, path in good_candidates]
    bad_files = [path for _, path in bad_candidates]
    return good_files, bad_files


def validate_offsets(
    path: Path,
    offsets: np.ndarray,
) -> tuple[int, int]:
    """Validate offsets and return ``(number of events, flat entries)``."""
    if offsets.ndim != 1:
        raise ValueError(
            f"{path.name}: 'offsets' must be one-dimensional, "
            f"but has shape {offsets.shape}."
        )

    if offsets.size == 0:
        raise ValueError(
            f"{path.name}: 'offsets' is empty; expected at least [0]."
        )

    if offsets[0] != 0:
        raise ValueError(
            f"{path.name}: offsets must begin at zero, "
            f"but begin at {offsets[0]}."
        )

    if np.any(np.diff(offsets) < 0):
        raise ValueError(
            f"{path.name}: offsets are not monotonically increasing."
        )

    return offsets.size - 1, int(offsets[-1])


def concatenate_arrays(
    arrays: list[np.ndarray],
    key: str,
) -> np.ndarray:
    """Concatenate arrays with a useful error if shapes are inconsistent."""
    if not arrays:
        raise ValueError(f"No arrays were collected for key '{key}'.")

    try:
        return np.concatenate(arrays, axis=0)
    except ValueError as exc:
        shapes = [array.shape for array in arrays]
        raise ValueError(
            f"Could not concatenate key '{key}'. Source shapes: {shapes}"
        ) from exc


def build_flat_indices(
    offsets: np.ndarray,
    event_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return flat-entry indices and rebuilt offsets for selected events.

    Event order is preserved. Empty events remain represented by repeated
    offsets.
    """
    event_indices = np.asarray(event_indices, dtype=np.int64)

    if event_indices.size == 0:
        return np.empty(0, dtype=np.int64), np.array([0], dtype=np.int64)

    counts = offsets[event_indices + 1] - offsets[event_indices]
    new_offsets = np.empty(event_indices.size + 1, dtype=np.int64)
    new_offsets[0] = 0
    np.cumsum(counts, out=new_offsets[1:])

    total_selected = int(new_offsets[-1])
    flat_indices = np.empty(total_selected, dtype=np.int64)

    destination = 0
    for event_index in event_indices:
        start = int(offsets[event_index])
        stop = int(offsets[event_index + 1])
        length = stop - start

        if length:
            flat_indices[destination:destination + length] = np.arange(
                start,
                stop,
                dtype=np.int64,
            )
            destination += length

    return flat_indices, new_offsets


class OutputAccumulator:
    """Collect selected events and write one internally consistent NPZ."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.collected: dict[str, list[np.ndarray]] = {}
        self.scalar_metadata: dict[str, np.ndarray] = {}
        self.combined_offsets: list[int] = [0]
        self.loaded_filenames: list[str] = []
        self.loaded_file_numbers: list[int] = []
        self.total_events = 0
        self.total_flat_entries = 0
        self.reference_flat_keys: Optional[set[str]] = None
        self.reference_event_keys: Optional[set[str]] = None
        self.reference_scalar_keys: Optional[set[str]] = None

    def establish_or_validate_keys(
        self,
        path: Path,
        flat_keys: set[str],
        event_keys: set[str],
        scalar_keys: set[str],
    ) -> None:
        if self.reference_flat_keys is None:
            self.reference_flat_keys = set(flat_keys)
            self.reference_event_keys = set(event_keys)
            self.reference_scalar_keys = set(scalar_keys)

            if not self.reference_flat_keys:
                raise KeyError(
                    f"{path.name}: none of the expected flat arrays were found."
                )

            if "evt_run" not in self.reference_event_keys:
                raise KeyError(
                    f"{path.name}: required event array 'evt_run' is missing."
                )
            return

        checks = (
            ("flat", self.reference_flat_keys, flat_keys),
            ("event", self.reference_event_keys or set(), event_keys),
            ("scalar", self.reference_scalar_keys or set(), scalar_keys),
        )

        for category, expected, current in checks:
            missing = expected - current
            extra = current - expected

            if missing:
                raise KeyError(
                    f"{path.name}: missing {category} keys for {self.name}: "
                    f"{sorted(missing)}"
                )

            if extra:
                raise KeyError(
                    f"{path.name}: unexpected {category} keys for {self.name}: "
                    f"{sorted(extra)}"
                )

    def add_selected_events(
        self,
        path: Path,
        file_number: int,
        data: np.lib.npyio.NpzFile,
        offsets: np.ndarray,
        event_indices: np.ndarray,
        flat_keys: set[str],
        event_keys: set[str],
        scalar_keys: set[str],
    ) -> None:
        event_indices = np.asarray(event_indices, dtype=np.int64)
        if event_indices.size == 0:
            return

        self.establish_or_validate_keys(
            path,
            flat_keys,
            event_keys,
            scalar_keys,
        )

        flat_indices, local_offsets = build_flat_indices(offsets, event_indices)
        selected_event_count = int(event_indices.size)
        selected_flat_count = int(flat_indices.size)

        output_file_index = len(self.loaded_filenames)

        for key in sorted(flat_keys):
            array = np.asarray(data[key])
            self.collected.setdefault(key, []).append(array[flat_indices])

        for key in sorted(event_keys):
            array = np.asarray(data[key])

            if key == "evt_file_idx":
                selected = np.full(
                    selected_event_count,
                    output_file_index,
                    dtype=array.dtype,
                )
            else:
                selected = array[event_indices]

            self.collected.setdefault(key, []).append(selected)

        for key in sorted(scalar_keys):
            value = np.asarray(data[key])

            if value.ndim != 0:
                raise ValueError(
                    f"{path.name}: metadata '{key}' should be scalar, "
                    f"but has shape {value.shape}."
                )

            if key not in self.scalar_metadata:
                self.scalar_metadata[key] = value.copy()
            elif not np.array_equal(self.scalar_metadata[key], value):
                raise ValueError(
                    f"{path.name}: scalar metadata '{key}' differs between "
                    f"files for {self.name}. First value: "
                    f"{self.scalar_metadata[key].item()}; current value: "
                    f"{value.item()}."
                )

        shifted_offsets = local_offsets[1:] + self.total_flat_entries
        self.combined_offsets.extend(shifted_offsets.tolist())

        self.total_events += selected_event_count
        self.total_flat_entries += selected_flat_count
        self.loaded_filenames.append(path.name)
        self.loaded_file_numbers.append(file_number)

    def make_output(
        self,
        split_fraction: Optional[float],
    ) -> dict[str, np.ndarray]:
        if self.total_events == 0:
            raise RuntimeError(
                f"No events were assigned to output '{self.name}'."
            )

        output: dict[str, np.ndarray] = {}

        for key, arrays in self.collected.items():
            output[key] = concatenate_arrays(arrays, key)

        output["offsets"] = np.asarray(
            self.combined_offsets,
            dtype=np.int64,
        )
        output.update(self.scalar_metadata)

        output["filenames"] = np.asarray(self.loaded_filenames, dtype=str)
        output["combined_file_numbers"] = np.asarray(
            self.loaded_file_numbers,
            dtype=np.int64,
        )
        output["configured_bad_file_numbers"] = np.asarray(
            sorted(BAD_FILE_NUMBERS),
            dtype=np.int64,
        )

        if split_fraction is not None:
            output["good_runs_fraction"] = np.asarray(
                split_fraction,
                dtype=np.float64,
            )
        output["num_source_files"] = np.asarray(
            len(self.loaded_filenames),
            dtype=np.int64,
        )

        expected_offsets = self.total_events + 1
        if output["offsets"].size != expected_offsets:
            raise RuntimeError(
                f"{self.name}: expected {expected_offsets:,} offsets, got "
                f"{output['offsets'].size:,}."
            )

        if int(output["offsets"][-1]) != self.total_flat_entries:
            raise RuntimeError(
                f"{self.name}: final offset should be "
                f"{self.total_flat_entries:,}, got "
                f"{int(output['offsets'][-1]):,}."
            )

        for key in self.reference_flat_keys or set():
            if output[key].shape[0] != self.total_flat_entries:
                raise RuntimeError(
                    f"{self.name}: flat array '{key}' has "
                    f"{output[key].shape[0]:,} entries; expected "
                    f"{self.total_flat_entries:,}."
                )

        for key in self.reference_event_keys or set():
            if output[key].shape[0] != self.total_events:
                raise RuntimeError(
                    f"{self.name}: event array '{key}' has "
                    f"{output[key].shape[0]:,} entries; expected "
                    f"{self.total_events:,}."
                )

        return output


# ============================================================
# Main
# ============================================================

def main() -> None:
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist: {INPUT_DIR}"
        )

    if not 0.0 < GOOD_RUNS_FRACTION < 1.0:
        raise ValueError(
            "GOOD_RUNS_FRACTION must be strictly between 0 and 1."
        )

    input_files, bad_input_files = find_input_files()
    if not input_files:
        raise RuntimeError(
            f"No non-bad input files were found in {INPUT_DIR}"
        )
    if not bad_input_files:
        raise RuntimeError(
            "None of the configured BAD_FILE_NUMBERS were found in "
            f"{INPUT_DIR}"
        )

    missing_bad_file_numbers = sorted(
        BAD_FILE_NUMBERS
        - {extract_file_number(path) for path in bad_input_files}
    )

    print()
    print(f"Found {len(input_files)} non-bad input files.")
    print(f"First non-bad file: {input_files[0].name}")
    print(f"Last non-bad file:  {input_files[-1].name}")
    print(f"Found {len(bad_input_files)} bad input files.")
    print("Bad files:")
    for path in bad_input_files:
        print(f"  {path.name}")

    if missing_bad_file_numbers:
        print(
            "Warning: configured bad file numbers not found: "
            + ", ".join(str(number) for number in missing_bad_file_numbers)
        )
    print(f"good_runs fraction:     {GOOD_RUNS_FRACTION:.2%}")
    print(f"windows_train fraction: {1.0 - GOOD_RUNS_FRACTION:.2%}")
    print(f"Good-runs output: {GOOD_RUNS_PATH}")
    print(f"Bad-runs output:  {BAD_RUNS_PATH}")
    print(f"Training output:  {WINDOWS_TRAIN_PATH}")
    print()

    # First pass: count all eligible events so the 75/25 split is global,
    # rather than independently splitting every source file.
    total_eligible_events = 0
    target_run_files: list[tuple[Path, int, Optional[int], Optional[int]]] = []

    print("First pass: counting eligible events...")

    for display_index, path in enumerate(input_files, start=1):
        with np.load(path, allow_pickle=True) as data:
            if "offsets" not in data.files:
                raise KeyError(
                    f"{path.name} does not contain required array 'offsets'."
                )

            offsets = np.asarray(data["offsets"], dtype=np.int64)
            n_events, _ = validate_offsets(path, offsets)
            total_eligible_events += n_events

            if "evt_run" not in data.files:
                print(
                    f"Warning: {path.name} has no 'evt_run' array; "
                    f"cannot check for run {TARGET_RUN}."
                )
            else:
                evt_run = np.asarray(data["evt_run"])

                if evt_run.ndim != 1 or evt_run.shape[0] != n_events:
                    raise ValueError(
                        f"{path.name}: 'evt_run' has shape {evt_run.shape}; "
                        f"expected ({n_events},)."
                    )

                matching_indices = np.flatnonzero(evt_run == TARGET_RUN)

                if matching_indices.size:
                    first_event_number: Optional[int] = None
                    last_event_number: Optional[int] = None

                    if "evt_num" in data.files:
                        evt_num = np.asarray(data["evt_num"])

                        if evt_num.ndim != 1 or evt_num.shape[0] != n_events:
                            raise ValueError(
                                f"{path.name}: 'evt_num' has shape "
                                f"{evt_num.shape}; expected ({n_events},)."
                            )

                        matching_event_numbers = evt_num[matching_indices]
                        first_event_number = int(matching_event_numbers.min())
                        last_event_number = int(matching_event_numbers.max())

                    target_run_files.append(
                        (
                            path,
                            int(matching_indices.size),
                            first_event_number,
                            last_event_number,
                        )
                    )

        print(
            f"[{display_index}/{len(input_files)}] "
            f"{path.name}: {n_events:,} events"
        )

    print()
    print(f"Files containing run {TARGET_RUN}:")

    if not target_run_files:
        print(
            f"  No eligible files containing run {TARGET_RUN} were found."
        )
    else:
        total_target_events = 0

        for path, event_count, first_event, last_event in target_run_files:
            total_target_events += event_count

            if first_event is not None and last_event is not None:
                event_range_text = (
                    f", event numbers {first_event} through {last_event}"
                )
            else:
                event_range_text = ""

            print(
                f"  {path.name}: {event_count:,} events"
                f"{event_range_text}"
            )

        print(
            f"  Total: {total_target_events:,} events from "
            f"{len(target_run_files):,} file(s)"
        )

    print()

    if total_eligible_events < 2:
        raise RuntimeError(
            "At least two eligible events are required for a 75/25 split."
        )

    # Use floor so good_runs receives at most the requested fraction.
    good_event_target = int(
        np.floor(total_eligible_events * GOOD_RUNS_FRACTION)
    )

    # Keep both outputs non-empty.
    good_event_target = max(
        1,
        min(good_event_target, total_eligible_events - 1),
    )
    train_event_target = total_eligible_events - good_event_target

    print()
    print(f"Total eligible events: {total_eligible_events:,}")
    print(f"good_runs target:      {good_event_target:,}")
    print(f"windows_train target:  {train_event_target:,}")
    print()
    print("Second pass: splitting and collecting events...")

    good_output = OutputAccumulator("good_runs")
    train_output = OutputAccumulator("windows_train")

    events_seen = 0

    for display_index, path in enumerate(input_files, start=1):
        file_number = extract_file_number(path)
        print(f"[{display_index}/{len(input_files)}] Loading {path.name}")

        with np.load(path, allow_pickle=True) as data:
            keys = set(data.files)

            if "offsets" not in keys:
                raise KeyError(
                    f"{path.name} does not contain required array 'offsets'."
                )

            offsets = np.asarray(data["offsets"], dtype=np.int64)
            n_events, n_flat_entries = validate_offsets(path, offsets)

            flat_keys = keys.intersection(FLAT_KEYS)
            event_keys = keys.intersection(EVENT_KEYS)
            scalar_keys = keys.intersection(SCALAR_KEYS)

            for key in sorted(flat_keys):
                array = np.asarray(data[key])
                if array.ndim == 0 or array.shape[0] != n_flat_entries:
                    raise ValueError(
                        f"{path.name}: flat array '{key}' has shape "
                        f"{array.shape}; expected first dimension "
                        f"{n_flat_entries:,}."
                    )

            for key in sorted(event_keys):
                array = np.asarray(data[key])
                if array.ndim == 0 or array.shape[0] != n_events:
                    raise ValueError(
                        f"{path.name}: event array '{key}' has shape "
                        f"{array.shape}; expected first dimension "
                        f"{n_events:,}."
                    )

            global_start = events_seen
            global_stop = events_seen + n_events

            # Events before good_event_target go to good_runs.
            local_good_stop = max(
                0,
                min(n_events, good_event_target - global_start),
            )

            good_indices = np.arange(
                0,
                local_good_stop,
                dtype=np.int64,
            )
            train_indices = np.arange(
                local_good_stop,
                n_events,
                dtype=np.int64,
            )

            good_output.add_selected_events(
                path=path,
                file_number=file_number,
                data=data,
                offsets=offsets,
                event_indices=good_indices,
                flat_keys=flat_keys,
                event_keys=event_keys,
                scalar_keys=scalar_keys,
            )

            train_output.add_selected_events(
                path=path,
                file_number=file_number,
                data=data,
                offsets=offsets,
                event_indices=train_indices,
                flat_keys=flat_keys,
                event_keys=event_keys,
                scalar_keys=scalar_keys,
            )

            events_seen = global_stop

            print(
                f"    total events={n_events:,}, "
                f"good_runs={good_indices.size:,}, "
                f"windows_train={train_indices.size:,}"
            )

    if events_seen != total_eligible_events:
        raise RuntimeError(
            f"Second-pass event count {events_seen:,} does not match "
            f"first-pass count {total_eligible_events:,}."
        )

    print()
    print("Collecting bad files...")

    bad_output = OutputAccumulator("bad_runs")

    for display_index, path in enumerate(bad_input_files, start=1):
        file_number = extract_file_number(path)
        print(
            f"[{display_index}/{len(bad_input_files)}] "
            f"Loading bad file {path.name}"
        )

        with np.load(path, allow_pickle=True) as data:
            keys = set(data.files)

            if "offsets" not in keys:
                raise KeyError(
                    f"{path.name} does not contain required array 'offsets'."
                )

            offsets = np.asarray(data["offsets"], dtype=np.int64)
            n_events, n_flat_entries = validate_offsets(path, offsets)

            flat_keys = keys.intersection(FLAT_KEYS)
            event_keys = keys.intersection(EVENT_KEYS)
            scalar_keys = keys.intersection(SCALAR_KEYS)

            for key in sorted(flat_keys):
                array = np.asarray(data[key])
                if array.ndim == 0 or array.shape[0] != n_flat_entries:
                    raise ValueError(
                        f"{path.name}: flat array '{key}' has shape "
                        f"{array.shape}; expected first dimension "
                        f"{n_flat_entries:,}."
                    )

            for key in sorted(event_keys):
                array = np.asarray(data[key])
                if array.ndim == 0 or array.shape[0] != n_events:
                    raise ValueError(
                        f"{path.name}: event array '{key}' has shape "
                        f"{array.shape}; expected first dimension "
                        f"{n_events:,}."
                    )

            all_event_indices = np.arange(n_events, dtype=np.int64)

            bad_output.add_selected_events(
                path=path,
                file_number=file_number,
                data=data,
                offsets=offsets,
                event_indices=all_event_indices,
                flat_keys=flat_keys,
                event_keys=event_keys,
                scalar_keys=scalar_keys,
            )

            print(
                f"    bad_runs={n_events:,} events, "
                f"{n_flat_entries:,} flat entries"
            )

    print()
    print("Combining arrays...")

    good_npz = good_output.make_output(GOOD_RUNS_FRACTION)
    train_npz = train_output.make_output(GOOD_RUNS_FRACTION)
    bad_npz = bad_output.make_output(None)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Writing {GOOD_RUNS_PATH}...")
    np.savez_compressed(GOOD_RUNS_PATH, **good_npz)

    print(f"Writing {WINDOWS_TRAIN_PATH}...")
    np.savez_compressed(WINDOWS_TRAIN_PATH, **train_npz)

    print(f"Writing {BAD_RUNS_PATH}...")
    np.savez_compressed(BAD_RUNS_PATH, **bad_npz)

    print()
    print("Finished.")
    print(
        f"good_runs.npz:     {good_output.total_events:,} events, "
        f"{good_output.total_flat_entries:,} flat entries, "
        f"{len(good_output.loaded_filenames):,} contributing files"
    )
    print(
        f"windows_train.npz: {train_output.total_events:,} events, "
        f"{train_output.total_flat_entries:,} flat entries, "
        f"{len(train_output.loaded_filenames):,} contributing files"
    )
    print(
        f"bad_runs.npz:      {bad_output.total_events:,} events, "
        f"{bad_output.total_flat_entries:,} flat entries, "
        f"{len(bad_output.loaded_filenames):,} contributing files"
    )
    print(f"Good-runs output: {GOOD_RUNS_PATH}")
    print(f"Bad-runs output:  {BAD_RUNS_PATH}")
    print(f"Training output:  {WINDOWS_TRAIN_PATH}")


if __name__ == "__main__":
    main()
