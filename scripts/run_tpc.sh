#!/usr/bin/env bash
# Train the TPC autoencoder.
# Adjust CONFIG and any environment variables to suit your cluster.

set -euo pipefail

CONFIG="${CONFIG:-configs/tpc_train.yaml}"

echo "=== TPC Autoencoder Training ==="
echo "Config : $CONFIG"
echo "Started: $(date)"

python -m sbn_anomaly_detection.train.train_tpc \
    --config "$CONFIG"

echo "Finished: $(date)"
