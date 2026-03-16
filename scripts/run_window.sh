#!/usr/bin/env bash
# Train the Window autoencoder.
# Requires TPC, PMT, and Fusion checkpoints in checkpoint_dir.

set -euo pipefail

CONFIG="${CONFIG:-configs/window_train.yaml}"
FUSION_CKPT="${FUSION_CKPT:-checkpoints/fusion_best.pt}"

echo "=== Window Autoencoder Training ==="
echo "Config      : $CONFIG"
echo "Fusion ckpt : $FUSION_CKPT"
echo "Started     : $(date)"

python -m sbn_anomaly_detection.train.train_window \
    --config "$CONFIG" \
    --fusion-checkpoint "$FUSION_CKPT"

echo "Finished: $(date)"
