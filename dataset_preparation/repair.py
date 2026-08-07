#!/usr/bin/env python3

from pathlib import Path
import shutil


# Directory containing the model subdirectories
MODELS_ROOT = Path(
    "/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection/checkpoints/graph_vae"
)

# Directory where renamed YAML files will be saved
OUTPUT_DIR = Path(
    "/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection/tuning_configs/graph_vae_sweep"
)


def copy_model_configs() -> None:
    if not MODELS_ROOT.is_dir():
        raise NotADirectoryError(f"Models directory does not exist: {MODELS_ROOT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0
    failed = 0

    for model_dir in sorted(MODELS_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue

        source_config = model_dir / "config_run.yaml"
        destination_config = OUTPUT_DIR / f"{model_dir.name}.yaml"

        if not source_config.is_file():
            print(f"[MISSING] {source_config}")
            missing += 1
            continue

        try:
            # copy2 preserves file metadata when possible.
            # Existing destination files with the same name are overwritten.
            shutil.copy2(source_config, destination_config)
            print(f"[COPIED] {source_config} -> {destination_config}")
            copied += 1

        except OSError as exc:
            print(f"[FAILED] {source_config}: {exc}")
            failed += 1

    print("\nSummary")
    print(f"  Copied: {copied}")
    print(f"  Missing config_run.yaml: {missing}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    copy_model_configs()