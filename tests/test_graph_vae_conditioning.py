"""Tests for graph-level conditioning (data.graph_features -> GraphVAE(graph_dim=...)).

Covers:
  - SparseWindowDatasetPyG packs graph_features into a standardized
    Data.graph_attr tensor of shape (1, graph_dim).
  - graph_attr survives PyG batching as (num_graphs, graph_dim), broadcastable
    to nodes via data.batch.
  - GraphVAE with graph_dim=0 (default) is unaffected -- same shapes/behavior
    as before this feature existed.
  - GraphVAE with graph_dim>0 conditions both encode() and decode(), and
    raises clear errors when graph_attr is missing or the wrong width.
"""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import torch
from torch_geometric.data import Batch

from sbn_anomaly.data.sparse_window_dataset import SparseWindowDatasetPyG
from sbn_anomaly.models.graph_vae import GraphVAE


def _graph_cond_dataset(graph_features, **overrides):
    # 2 channels, 6 events, one hit per event alternating channel 0/1.
    channels = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    integrals = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    offsets = np.arange(7, dtype=np.int64)
    kwargs = dict(
        channels_flat=channels, integrals_flat=integrals, offsets=offsets,
        n_channels=2, window_size=3, n_bins=1, stride=3,
        node_features=["sum"], prune_inactive=False,
        reconstruction=True, standardize=True,
        graph_features=graph_features,
    )
    kwargs.update(overrides)
    return SparseWindowDatasetPyG(**kwargs)


def test_graph_attr_shape_and_dim():
    ds = _graph_cond_dataset(["event_count", "log1p_event_count"])
    assert ds.graph_feat_dim == 2
    data = ds[0]
    assert data.graph_attr.shape == (1, 2)


def test_graph_attr_standardization_roundtrip():
    ds = _graph_cond_dataset(["event_count"])
    assert ds.graph_feature_mean is not None
    assert ds.graph_feature_mean.shape == (1,)
    std_dict = ds.standardization()
    assert "graph_feature_mean" in std_dict and "graph_feature_std" in std_dict

    # A second dataset built with those same fitted stats should reproduce
    # identical standardized graph_attr values (inference-time reuse).
    ds2 = _graph_cond_dataset(
        ["event_count"],
        graph_feature_mean=std_dict["graph_feature_mean"],
        graph_feature_std=std_dict["graph_feature_std"],
    )
    assert torch.allclose(ds[0].graph_attr, ds2[0].graph_attr)


def test_no_graph_features_means_no_graph_attr():
    ds = _graph_cond_dataset(None)
    assert ds.graph_feat_dim == 0
    data = ds[0]
    assert getattr(data, "graph_attr", None) is None


def test_graph_attr_batches_to_num_graphs_by_dim():
    ds = _graph_cond_dataset(["event_count"])
    batch = Batch.from_data_list([ds[0], ds[1]])
    assert batch.graph_attr.shape == (2, 1)
    # broadcast via batch.batch should give one row per node
    broadcast = batch.graph_attr[batch.batch]
    assert broadcast.shape[0] == batch.x.shape[0]


def _tiny_model(graph_dim=0, in_dim=1):
    return GraphVAE(
        in_dim=in_dim, latent_dim=4,
        encoder_hidden_dims=[8], decoder_hidden_dims=[],
        mask_ratio=0.0, use_channel_idx=True, graph_dim=graph_dim,
    ).eval()


def test_forward_unaffected_when_graph_dim_zero():
    ds = _graph_cond_dataset(None, node_features=["sum"])
    batch = Batch.from_data_list([ds[0], ds[1]])
    model = _tiny_model(graph_dim=0, in_dim=ds.node_feat_dim)
    x_hat, mu, logvar, z = model(batch)
    assert x_hat.shape == batch.y.shape


def test_forward_with_graph_conditioning():
    ds = _graph_cond_dataset(["event_count"], node_features=["sum"])
    batch = Batch.from_data_list([ds[0], ds[1]])
    model = _tiny_model(graph_dim=1, in_dim=ds.node_feat_dim)
    x_hat, mu, logvar, z = model(batch)
    assert x_hat.shape == batch.y.shape


def test_forward_raises_when_graph_attr_missing():
    ds = _graph_cond_dataset(None, node_features=["sum"])  # no graph_attr attached
    batch = Batch.from_data_list([ds[0], ds[1]])
    model = _tiny_model(graph_dim=1, in_dim=ds.node_feat_dim)
    with pytest.raises(ValueError):
        model(batch)


def test_forward_raises_on_graph_attr_width_mismatch():
    ds = _graph_cond_dataset(["event_count", "log1p_event_count"], node_features=["sum"])
    batch = Batch.from_data_list([ds[0], ds[1]])  # graph_attr width 2
    model = _tiny_model(graph_dim=1, in_dim=ds.node_feat_dim)  # expects width 1
    with pytest.raises(ValueError):
        model(batch)
