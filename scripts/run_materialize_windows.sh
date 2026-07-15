#!/usr/bin/env bash
set -euo pipefail

################################################################################
# run_materialize_windows.sh
#
# For each CI_build_lar_ci_<N> directory beneath a base path, finds the
# DQM*.root files in its reco/ subdirectory and runs
# sbn_anomaly.data.materialize_windows separately for that directory alone,
# writing a tagged output file per directory (tag = the number after
# CI_build_lar_ci_). Processing one directory at a time avoids loading every
# file across every build directory into memory at once.
#
# Pass -l/--file-list to skip directory discovery entirely and instead
# materialize an explicit list of ROOT files (one per line) into a single,
# untagged output file.
#
# Default base path:
#   /pnfs/sbnd/scratch/ci_validation/dqm/v09_93_01_02/CI_build_lar_ci_<N>/reco/DQM*.root
#
# Usage:
#   ./run_materialize_windows.sh [-f] [-t] [-d DATA_DIR] [-c CONFIG] [-l FILE_LIST] [output_base]
#
# Flags:
#   -f, --force           Force reprocessing: overwrite output files that
#                         already exist. Without this flag, directories
#                         whose output file is already present are skipped.
#   -t, --test            Test mode: process only the first file found in
#                         the first directory (or the first file in
#                         --file-list), then stop. Useful for a quick smoke
#                         test of the pipeline before a full run.
#   -d, --data-dir DIR    Base directory to search for CI_build_lar_ci_*
#                         subdirectories. Overrides the hardcoded default.
#   -c, --config PATH     Network/data config YAML passed to
#                         materialize_windows (default: configs/graph_vae.yaml).
#   -l, --file-list PATH  A text file listing ROOT file paths (one per line,
#                         '#' comments and blank lines ignored). When given,
#                         directory discovery is skipped and every file in
#                         the list is materialized together into a single
#                         output. Combine with -t to only process the
#                         list's first file.
#
# Arguments (all optional):
#   output_base   (default: data/events_cache.npz)
#                 In directory mode, each directory's output is written as
#                 <output_base_without_ext>_<N><ext>, e.g. for the default,
#                 build dir CI_build_lar_ci_42 writes data/events_cache_42.npz.
#                 In --file-list mode, output is written to output_base as-is.
#
################################################################################

force=0
test_mode=0
data_dir=""
config_path="configs/graph_vae.yaml"
file_list=""
args=()

while (( "$#" )); do
    case "$1" in
        -f|--force)
            force=1
            shift
            ;;
        -t|--test)
            test_mode=1
            shift
            ;;
        -d|--data-dir)
            [ $# -ge 2 ] || { echo "[ERROR] $1 requires an argument." >&2; exit 1; }
            data_dir="$2"
            shift 2
            ;;
        -c|--config)
            [ $# -ge 2 ] || { echo "[ERROR] $1 requires an argument." >&2; exit 1; }
            config_path="$2"
            shift 2
            ;;
        -l|--file-list)
            [ $# -ge 2 ] || { echo "[ERROR] $1 requires an argument." >&2; exit 1; }
            file_list="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "[ERROR] Unknown option: $1" >&2
            exit 1
            ;;
        *)
            args+=("$1")
            shift
            ;;
    esac
done
set -- "${args[@]+"${args[@]}"}"

output_base="${1:-data/events_cache.npz}"

base_dir="${data_dir:-/pnfs/sbnd/scratch/ci_validation/dqm/v09_93_01_02}"

# Split output_base into directory, stem, and extension so we can insert the
# per-build tag before the extension (e.g. events_cache.npz -> events_cache_42.npz).
output_dir="$(dirname "$output_base")"
output_name="$(basename "$output_base")"
case "$output_name" in
    *.*)
        output_stem="${output_name%.*}"
        output_ext=".${output_name##*.}"
        ;;
    *)
        output_stem="$output_name"
        output_ext=""
        ;;
esac

mkdir -p "$output_dir"

if (( force )); then
    echo "[INFO] Force mode enabled: existing output files will be overwritten."
fi
if (( test_mode )); then
    echo "[INFO] Test mode enabled: only the first file (from the first directory, or from --file-list) will be processed."
fi
echo "[INFO] Using config: $config_path"

processed=0
skipped=0

################################################################################
# --file-list mode: bypass CI_build_lar_ci_* directory discovery entirely and
# materialize an explicit list of ROOT files into one output.
################################################################################
if [[ -n "$file_list" ]]; then
    if [[ ! -f "$file_list" ]]; then
        echo "[ERROR] File list not found: $file_list" >&2
        exit 1
    fi

    mapfile -t list_files < <(grep -vE '^\s*(#|$)' "$file_list")

    if (( ${#list_files[@]} == 0 )); then
        echo "[ERROR] No files found in $file_list." >&2
        exit 1
    fi

    if (( test_mode )); then
        list_files=("${list_files[0]}")
    fi

    if [ -f "$output_base" ] && (( ! force )); then
        echo "[INFO] Skipping: $output_base already exists. Use -f to overwrite."
        skipped=$((skipped + 1))
    else
        echo "[INFO] --file-list mode: ${#list_files[@]} file(s) -> $output_base"
        printf '    %s\n' "${list_files[@]}"

        python -m sbn_anomaly.data.materialize_windows \
            --config "$config_path" \
            --root-files "${list_files[@]}" \
            --output "$output_base"

        processed=$((processed + 1))
    fi

    echo "[SUCCESS] Processed $processed run(s), skipped $skipped."
    exit 0
fi

################################################################################
# Default mode: discover CI_build_lar_ci_* directories under base_dir.
################################################################################
echo "[INFO] Searching for CI_build_lar_ci_* directories under: $base_dir"

# Find each build directory directly (not the files within it yet). Using
# find rather than shell glob expansion since PNFS/dCache namespaces have
# proven unreliable with bash wildcard matching even when `ls` works fine.
build_dirs=()
while IFS= read -r -d '' d; do
    build_dirs+=("$d")
done < <(find "$base_dir" -mindepth 1 -maxdepth 1 -type d \
    -name "CI_build_lar_ci_*" -print0 2>/dev/null)

numDirs=${#build_dirs[@]}
echo "[INFO] Found $numDirs CI_build_lar_ci_* director(ies)."

if (( numDirs == 0 )); then
    echo "[ERROR] No CI_build_lar_ci_* directories found under $base_dir." >&2
    exit 1
fi

for build_dir in "${build_dirs[@]}"; do
    dir_name="$(basename "$build_dir")"

    # Extract the numeric tag that follows CI_build_lar_ci_
    if [[ "$dir_name" =~ CI_build_lar_ci_([0-9]+) ]]; then
        tag="${BASH_REMATCH[1]}"
    else
        echo "[WARN] Skipping $dir_name: couldn't extract a numeric tag." >&2
        skipped=$((skipped + 1))
        continue
    fi

    dir_output="${output_dir}/${output_stem}_${tag}${output_ext}"

    # Skip if output already exists and -f was not passed.
    if [ -f "$dir_output" ] && (( ! force )); then
        echo "[INFO] Skipping $dir_name (tag $tag): $dir_output already exists. Use -f to overwrite."
        skipped=$((skipped + 1))
        continue
    fi

    # Find this directory's DQM*.root files under reco/
    dir_files=()
    while IFS= read -r -d '' f; do
        dir_files+=("$f")
    done < <(find "$build_dir/reco" -maxdepth 1 -type f -name "DQM*.root" -print0 2>/dev/null)

    if (( ${#dir_files[@]} == 0 )); then
        echo "[WARN] Skipping $dir_name (tag $tag): no DQM*.root files found in reco/."
        skipped=$((skipped + 1))
        continue
    fi

    if (( test_mode )); then
        dir_files=("${dir_files[0]}")
    fi

    echo "[INFO] $dir_name (tag $tag): ${#dir_files[@]} file(s) -> $dir_output"
    printf '    %s\n' "${dir_files[@]}"

    python -m sbn_anomaly.data.materialize_windows \
        --config "$config_path" \
        --root-files "${dir_files[@]}" \
        --output "$dir_output"

    processed=$((processed + 1))

    if (( test_mode )); then
        echo "[INFO] Test mode: stopping after the first directory."
        break
    fi
done

echo "[SUCCESS] Processed $processed director(ies), skipped $skipped."
