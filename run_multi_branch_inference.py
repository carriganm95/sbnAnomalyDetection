#!/usr/bin/env python3
"""Multi-branch inference: score test set with TPC, PMT, and Window models."""

import numpy as np
import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_multi_branch_inference():
    """Score test set using all three models."""
    from sbn_anomaly.infer.multi_branch import MultiBranchScorer
    
    # Load configs
    tpc_config = yaml.safe_load(open("/nashome/m/micarrig/icarus/ml/sbnAnomalyDetection/configs/tpc.yaml"))
    pmt_config = yaml.safe_load(open("/nashome/m/micarrig/icarus/ml/sbnAnomalyDetection/configs/pmt.yaml"))
    window_config = yaml.safe_load(open("/nashome/m/micarrig/icarus/ml/sbnAnomalyDetection/configs/window.yaml"))
    
    # Load test features
    test_archive = np.load("/exp/icarus/data/users/micarrig/tpc_test_features.npz", allow_pickle=True)
    tpc_features = test_archive['features']
    
    logger.info(f"TPC test features shape: {tpc_features.shape}")
    
    # Create PMT features (adapter: use first 128 dims of TPC)
    pmt_features = tpc_features[:, :128]
    pmt_features = (pmt_features - pmt_features.mean(axis=0)) / (pmt_features.std(axis=0) + 1e-6)
    
    # Create Window features (adapter: reshape TPC to (n, 1, 256))
    window_features = np.zeros((tpc_features.shape[0], 1, 256), dtype=np.float32)
    for i in range(len(tpc_features)):
        feat = tpc_features[i]
        if len(feat) >= 256:
            window_features[i, 0, :] = feat[:256]
        else:
            window_features[i, 0, :len(feat)] = feat
    window_features = (window_features - window_features.mean()) / (window_features.std() + 1e-6)
    
    # Load/truncate features to model input dimensions
    tpc_input_dim = tpc_config['model'].get('input_dim', 256)
    tpc_features = tpc_features[:, :tpc_input_dim]
    if tpc_features.shape[1] < tpc_input_dim:
        pad_width = ((0, 0), (0, tpc_input_dim - tpc_features.shape[1]))
        tpc_features = np.pad(tpc_features, pad_width)
    
    logger.info(f"Prepared features:")
    logger.info(f"  TPC: {tpc_features.shape}")
    logger.info(f"  PMT: {pmt_features.shape}")
    logger.info(f"  Window: {window_features.shape}")
    
    # Create multi-branch scorer
    scorer = MultiBranchScorer.from_checkpoints(
        tpc_checkpoint="/nashome/m/micarrig/icarus/ml/sbnAnomalyDetection/checkpoints/tpc/tpc_final.pt",
        pmt_checkpoint="/nashome/m/micarrig/icarus/ml/sbnAnomalyDetection/checkpoints/pmt/pmt_final.pt",
        window_checkpoint="/nashome/m/micarrig/icarus/ml/sbnAnomalyDetection/checkpoints/window/window_final.pt",
        tpc_config=tpc_config['model'],
        pmt_config=pmt_config['model'],
        window_config=window_config['model'],
        tpc_data_config=tpc_config['data'],
        pmt_data_config=pmt_config['data'],
        window_data_config=window_config['data'],
        tpc_threshold=0.25,
        pmt_threshold=0.25,
        window_threshold=0.25,
    )
    
    # Score
    logger.info("Scoring with multi-branch detector...")
    results = scorer.score(
        tpc_features=tpc_features,
        pmt_features=pmt_features,
        window_features=window_features,
    )
    
    # Save results
    output_path = Path("/nashome/m/micarrig/icarus/ml/sbnAnomalyDetection/multi_branch_scores.npz")
    
    # Prepare output
    output_dict = {
        'tpc_scores': results['tpc_scores'],
        'pmt_scores': results['pmt_scores'],
        'window_scores': results['window_scores'],
        'max_scores': results['max_scores'],
        'alert_flags': results['alert_flags'],
    }
    
    # Add metadata from test archive
    for key in ['feature_branch_names', 'tpc_branch_values', 'tpc_branch_names', 'input_filenames']:
        if key in test_archive:
            output_dict[key] = test_archive[key]
    
    np.savez_compressed(output_path, **output_dict)
    logger.info(f"Saved multi-branch results to {output_path}")
    
    # Print statistics
    logger.info(f"TPC scores: min={results['tpc_scores'].min():.4f}, max={results['tpc_scores'].max():.4f}, mean={results['tpc_scores'].mean():.4f}")
    logger.info(f"PMT scores: min={results['pmt_scores'].min():.4f}, max={results['pmt_scores'].max():.4f}, mean={results['pmt_scores'].mean():.4f}")
    logger.info(f"Window scores: min={results['window_scores'].min():.4f}, max={results['window_scores'].max():.4f}, mean={results['window_scores'].mean():.4f}")
    logger.info(f"Max scores: min={results['max_scores'].min():.4f}, max={results['max_scores'].max():.4f}, mean={results['max_scores'].mean():.4f}")
    logger.info(f"Alerts (>= 0.25 threshold): {results['alert_flags'].sum()} / {len(results['alert_flags'])}")


if __name__ == "__main__":
    run_multi_branch_inference()
