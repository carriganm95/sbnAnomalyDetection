"""Tests for the TPC hit-branch ablation helpers."""

from __future__ import annotations

from sbn_anomaly.train.ablation import (
    BranchVariant,
    build_leave_one_out_variants,
    build_variant_config,
    split_metadata_and_hit_branches,
)


def test_split_metadata_and_hit_branches():
    metadata, hits = split_metadata_and_hit_branches(
        ["meta.run", "meta.subrun", "hits0.h.integral", "hits0.h.sumadc"],
        ["hits0.h.integral", "hits0.h.sumadc"],
    )

    assert metadata == ["meta.run", "meta.subrun"]
    assert hits == ["hits0.h.integral", "hits0.h.sumadc"]


def test_build_leave_one_out_variants_includes_baseline():
    variants = build_leave_one_out_variants(["a", "b", "c"], include_baseline=True)

    assert [variant.name for variant in variants] == ["baseline_all", "minus_a", "minus_b", "minus_c"]
    assert variants[1].hit_branches == ["b", "c"]
    assert variants[2].hit_branches == ["a", "c"]
    assert variants[3].hit_branches == ["a", "b"]


def test_build_variant_config_keeps_metadata_separate(tmp_path):
    base_cfg = {
        "data": {
            "hit_branches": ["hits0.h.integral", "hits0.h.sumadc"],
            "tpc_branches": ["meta.run", "meta.subrun", "hits0.h.integral", "hits0.h.sumadc"],
        },
        "training": {
            "checkpoint_dir": "checkpoints/tpc",
            "output_path": "checkpoints/tpc/tpc_final.pt",
        },
    }
    variant = BranchVariant(name="minus_hits0_h_integral", hit_branches=["hits0.h.sumadc"], removed_branch="hits0.h.integral")

    cfg = build_variant_config(base_cfg, variant, ["meta.run", "meta.subrun"], tmp_path)

    assert cfg["data"]["hit_branches"] == ["hits0.h.sumadc"]
    assert cfg["data"]["tpc_branches"] == ["meta.run", "meta.subrun"]
    assert cfg["training"]["checkpoint_dir"].endswith("minus_hits0_h_integral/checkpoints")
    assert cfg["training"]["output_path"].endswith("minus_hits0_h_integral/model.pt")