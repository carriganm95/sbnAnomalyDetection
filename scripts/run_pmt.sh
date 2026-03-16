#!/usr/bin/env bash
# Train the PMT autoencoder.

set -euo pipefail

CONFIG="${CONFIG:-configs/pmt_train.yaml}"

echo "=== PMT Autoencoder Training ==="
echo "Config : $CONFIG"
echo "Started: $(date)"

python -m sbn_anomaly_detection.train.train_pmt \
    --config "$CONFIG"

echo "Finished: $(date)"
