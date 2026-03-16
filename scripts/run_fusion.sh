#!/usr/bin/env bash
# Train the Fusion autoencoder.
# Requires TPC and PMT checkpoints to already exist in checkpoint_dir.

set -euo pipefail

CONFIG="${CONFIG:-configs/fusion_train.yaml}"
TPC_CKPT="${TPC_CKPT:-checkpoints/tpc_best.pt}"
PMT_CKPT="${PMT_CKPT:-checkpoints/pmt_best.pt}"

echo "=== Fusion Autoencoder Training ==="
echo "Config     : $CONFIG"
echo "TPC ckpt   : $TPC_CKPT"
echo "PMT ckpt   : $PMT_CKPT"
echo "Started    : $(date)"

python -m sbn_anomaly_detection.train.train_fusion \
    --config "$CONFIG" \
    --tpc-checkpoint "$TPC_CKPT" \
    --pmt-checkpoint "$PMT_CKPT"

echo "Finished: $(date)"
