#!/usr/bin/env python3

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for remote servers

import matplotlib.pyplot as plt
import numpy as np
import ROOT as r


ROOT_FILES = [
    "/exp/icarus/app/users/micarrig/ml/sbnAnomalyDetection/scripts/out_evt103_ch4878.root",
    "/exp/icarus/app/users/micarrig/ml/sbnAnomalyDetection/scripts/out_evt103_ch4888.root",
    "/exp/icarus/app/users/micarrig/ml/sbnAnomalyDetection/scripts/out_evt42237_ch4878.root",
    "/exp/icarus/app/users/micarrig/ml/sbnAnomalyDetection/scripts/out_evt42237_ch4888.root",
]

CANVAS_PATHS = [
    "Run20104/Event103/Channel4878",
    "Run20104/Event103/Channel4888",
    "Run20153/Event42237/Channel4878",
    "Run20153/Event42237/Channel4888",
]

LABELS = [
    "Ch 4878 (Bad Run)",
    "Ch 4888 (Bad Run)",
    "Ch 4878 (Good Run)",
    "Ch 4888 (Good Run)",
]

COLORS = ["red", "orange", "blue", "green"]
LINESTYLES = ["-", "-", "-", "-"]

OUTPUT_PATH = "pulse_comparison.png"


def hist_to_arrays(canvas, hist_name="hRaw"):
    if not canvas:
        raise RuntimeError("Canvas was not found in the ROOT file.")

    primitives = canvas.GetListOfPrimitives()
    hist = primitives.FindObject(hist_name)

    if not hist:
        available = [
            primitive.GetName()
            for primitive in primitives
        ]
        raise RuntimeError(
            f"Histogram '{hist_name}' was not found. "
            f"Available primitives: {available}"
        )

    n_bins = hist.GetNbinsX()

    x = np.array(
        [hist.GetBinCenter(i) for i in range(1, n_bins + 1)],
        dtype=float,
    )
    y = np.array(
        [hist.GetBinContent(i) for i in range(1, n_bins + 1)],
        dtype=float,
    )

    return x, y


root_files = []
canvases = []

try:
    for file_path, canvas_path in zip(ROOT_FILES, CANVAS_PATHS):
        root_file = r.TFile.Open(file_path)

        if not root_file or root_file.IsZombie():
            raise RuntimeError(f"Could not open ROOT file: {file_path}")

        canvas = root_file.Get(canvas_path)

        if not canvas:
            raise RuntimeError(
                f"Could not find '{canvas_path}' in '{file_path}'"
            )

        # Keep ROOT files open while their objects are being used.
        root_files.append(root_file)
        canvases.append(canvas)

    fig, ax = plt.subplots(figsize=(12, 5))

    for index, (canvas, label) in enumerate(zip(canvases, LABELS)):
        x, y = hist_to_arrays(canvas)

        ax.plot(
            x,
            y,
            linewidth=1,
            label=label,
            color=COLORS[index],
            linestyle=LINESTYLES[index],
        )

    ax.set_xlabel("Time Ticks")
    ax.set_ylabel("Baseline-Subtracted Signal Amplitude (ADC Counts)")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to: {OUTPUT_PATH}")

finally:
    for root_file in root_files:
        root_file.Close()