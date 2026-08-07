#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# User settings
# ============================================================

GOOD_NPZ_PATH = Path(
    "/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection/"
    "scratch/8800_to_9000_200_channels/good_events_test.npz"
)

BAD_NPZ_PATH = Path(
    "/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection/"
    "scratch/8800_to_9000_200_channels/bad_events_test.npz"
)

TARGET_RUN = 20142

# Number of common histogram bins.
N_BINS = 100

# Optional x-axis upper limit:
#   None  -> show the complete range, including the extreme tail
#   99.9  -> show values up to the combined 99.9th percentile
X_MAX_PERCENTILE = 99.9

# Use logarithmic scaling on the y-axis.
LOG_Y = False

# Save beside the good-runs NPZ file.
OUTPUT_PATH = GOOD_NPZ_PATH.with_name(
    f"{GOOD_NPZ_PATH.stem}_run_{TARGET_RUN}_good_bad_integral_histogram.png"
)


# ============================================================
# Helper functions
# ============================================================

def load_npz_arrays(
    npz_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and validate offsets, evt_run, and integrals_flat from an NPZ file.

    Returns:
        offsets
        evt_run
        integrals_flat
    """
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ file does not exist: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        required_keys = {
            "offsets",
            "evt_run",
            "integrals_flat",
        }

        missing_keys = required_keys - set(data.files)

        if missing_keys:
            raise KeyError(
                f"{npz_path.name} is missing required keys: "
                f"{sorted(missing_keys)}"
            )

        offsets = np.asarray(
            data["offsets"],
            dtype=np.int64,
        )

        evt_run = np.asarray(
            data["evt_run"],
            dtype=np.int64,
        )

        integrals_flat = np.asarray(
            data["integrals_flat"],
            dtype=np.float64,
        )

    validate_npz_arrays(
        npz_path=npz_path,
        offsets=offsets,
        evt_run=evt_run,
        integrals_flat=integrals_flat,
    )

    return offsets, evt_run, integrals_flat


def validate_npz_arrays(
    npz_path: Path,
    offsets: np.ndarray,
    evt_run: np.ndarray,
    integrals_flat: np.ndarray,
) -> None:
    """Validate the relationships between NPZ event and flat arrays."""
    if offsets.ndim != 1:
        raise ValueError(
            f"{npz_path.name}: 'offsets' must be one-dimensional, "
            f"but has shape {offsets.shape}."
        )

    if evt_run.ndim != 1:
        raise ValueError(
            f"{npz_path.name}: 'evt_run' must be one-dimensional, "
            f"but has shape {evt_run.shape}."
        )

    if integrals_flat.ndim != 1:
        raise ValueError(
            f"{npz_path.name}: 'integrals_flat' must be one-dimensional, "
            f"but has shape {integrals_flat.shape}."
        )

    if offsets.size != evt_run.size + 1:
        raise ValueError(
            f"{npz_path.name}: expected len(offsets) = len(evt_run) + 1, "
            f"but got {offsets.size:,} offsets and "
            f"{evt_run.size:,} events."
        )

    if offsets.size == 0:
        raise ValueError(
            f"{npz_path.name}: 'offsets' is empty."
        )

    if offsets[0] != 0:
        raise ValueError(
            f"{npz_path.name}: 'offsets' must start at zero, "
            f"but starts at {offsets[0]}."
        )

    if np.any(np.diff(offsets) < 0):
        raise ValueError(
            f"{npz_path.name}: 'offsets' is not monotonically increasing."
        )

    if int(offsets[-1]) != integrals_flat.size:
        raise ValueError(
            f"{npz_path.name}: final offset is "
            f"{int(offsets[-1]):,}, but integrals_flat contains "
            f"{integrals_flat.size:,} entries."
        )


def expand_runs_to_integrals(
    offsets: np.ndarray,
    evt_run: np.ndarray,
    integrals_flat: np.ndarray,
    npz_path: Path,
) -> np.ndarray:
    """
    Expand event-level run numbers so each flat integral entry has a run label.
    """
    entries_per_event = np.diff(offsets)

    run_per_integral = np.repeat(
        evt_run,
        entries_per_event,
    )

    if run_per_integral.size != integrals_flat.size:
        raise RuntimeError(
            f"{npz_path.name}: expanded run array has "
            f"{run_per_integral.size:,} entries, but integrals_flat has "
            f"{integrals_flat.size:,} entries."
        )

    return run_per_integral


def finite_values(values: np.ndarray) -> np.ndarray:
    """Return only finite values."""
    return values[np.isfinite(values)]


# ============================================================
# Load good-run NPZ
# ============================================================

good_offsets, good_evt_run, good_integrals_flat = load_npz_arrays(
    GOOD_NPZ_PATH
)

good_run_per_integral = expand_runs_to_integrals(
    offsets=good_offsets,
    evt_run=good_evt_run,
    integrals_flat=good_integrals_flat,
    npz_path=GOOD_NPZ_PATH,
)

target_mask = good_run_per_integral == TARGET_RUN

target_integrals = finite_values(
    good_integrals_flat[target_mask]
)

other_good_integrals = finite_values(
    good_integrals_flat[~target_mask]
)

if target_integrals.size == 0:
    available_runs = np.unique(good_evt_run)

    raise RuntimeError(
        f"No integral entries were found for run {TARGET_RUN} "
        f"in {GOOD_NPZ_PATH.name}.\n"
        f"Available good runs: {available_runs.tolist()}"
    )

if other_good_integrals.size == 0:
    raise RuntimeError(
        f"No integral entries outside run {TARGET_RUN} were found "
        f"in {GOOD_NPZ_PATH.name}."
    )


# ============================================================
# Load bad-run NPZ
# ============================================================

bad_offsets, bad_evt_run, bad_integrals_flat = load_npz_arrays(
    BAD_NPZ_PATH
)

# Expand the run labels as an additional consistency check.
bad_run_per_integral = expand_runs_to_integrals(
    offsets=bad_offsets,
    evt_run=bad_evt_run,
    integrals_flat=bad_integrals_flat,
    npz_path=BAD_NPZ_PATH,
)

bad_integrals = finite_values(
    bad_integrals_flat
)

if bad_integrals.size == 0:
    raise RuntimeError(
        f"No finite integral entries were found in {BAD_NPZ_PATH.name}."
    )


# ============================================================
# Construct common histogram bins
# ============================================================

all_integrals = np.concatenate(
    [
        target_integrals,
        other_good_integrals,
        bad_integrals,
    ]
)

x_min = float(np.min(all_integrals))

if X_MAX_PERCENTILE is None:
    x_max = float(np.max(all_integrals))
else:
    if not 0.0 < X_MAX_PERCENTILE <= 100.0:
        raise ValueError(
            "X_MAX_PERCENTILE must be greater than 0 and at most 100."
        )

    x_max = float(
        np.percentile(
            all_integrals,
            X_MAX_PERCENTILE,
        )
    )

if x_max <= x_min:
    x_max = x_min + 1.0

bins = np.linspace(
    x_min,
    x_max,
    N_BINS + 1,
)


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(10, 6))

# Other good runs: blue
plt.hist(
    other_good_integrals,
    bins=bins,
    density=True,
    histtype="step",
    linewidth=1.8,
    color="blue",
    label=f"Good runs except {TARGET_RUN}",
)

# Target good run: red
plt.hist(
    target_integrals,
    bins=bins,
    density=True,
    histtype="step",
    linewidth=1.8,
    color="red",
    label=f"Good run {TARGET_RUN}",
)

# Bad runs: orange
plt.hist(
    bad_integrals,
    bins=bins,
    density=True,
    histtype="step",
    linewidth=1.8,
    color="orange",
    label="Bad runs",
)

plt.xlabel("Integral value")
plt.ylabel("Normalized density")
plt.title(
    f"Integral-value distributions: run {TARGET_RUN}, "
    "other good runs, and bad runs"
)

plt.xlim(x_min, x_max)

if LOG_Y:
    plt.yscale("log")

plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight",
)

print()
print(f"Good NPZ: {GOOD_NPZ_PATH}")
print(f"Bad NPZ:  {BAD_NPZ_PATH}")
print()
print(f"Target run: {TARGET_RUN}")
print(
    f"Run {TARGET_RUN} integral entries: "
    f"{target_integrals.size:,}"
)
print(
    f"Other good-run integral entries: "
    f"{other_good_integrals.size:,}"
)
print(
    f"Bad-run integral entries: "
    f"{bad_integrals.size:,}"
)
print()
print(
    f"Run {TARGET_RUN} maximum integral: "
    f"{np.max(target_integrals):,.3f}"
)
print(
    f"Other good-run maximum integral: "
    f"{np.max(other_good_integrals):,.3f}"
)
print(
    f"Bad-run maximum integral: "
    f"{np.max(bad_integrals):,.3f}"
)
print()
print(
    f"Good runs found: "
    f"{np.unique(good_evt_run).tolist()}"
)
print(
    f"Bad runs found: "
    f"{np.unique(bad_evt_run).tolist()}"
)
print()
print(f"Plot saved to: {OUTPUT_PATH}")
print()

plt.show()