"""Branch-ablation runner for TPC hit features.

This module trains a TPC autoencoder repeatedly while removing one candidate
``hit_branches`` entry at a time. It is intended for comparing candidate input
variables with the same model, optimizer, and data split.
"""

from __future__ import annotations

import argparse
import copy
import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import yaml

from sbn_anomaly.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BranchVariant:
    """A single ablation variant."""

    name: str
    hit_branches: list[str]
    removed_branch: str | None = None


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    """Return unique items while preserving the original order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        item_str = str(item)
        if item_str in seen:
            continue
        seen.add(item_str)
        result.append(item_str)
    return result


def split_metadata_and_hit_branches(
    tpc_branches: Sequence[str] | None,
    hit_branches: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Split loaded ROOT branches into metadata and hit-branch lists.

    Any branch that appears in ``hit_branches`` is treated as a feature branch;
    everything else in ``tpc_branches`` is preserved as metadata or auxiliary
    loading.
    """
    hit_set = {str(branch) for branch in hit_branches}
    metadata = [str(branch) for branch in (tpc_branches or []) if str(branch) not in hit_set]
    return dedupe_preserve_order(metadata), dedupe_preserve_order(hit_branches)


def build_leave_one_out_variants(
    hit_branches: Sequence[str],
    include_baseline: bool = True,
) -> list[BranchVariant]:
    """Build a baseline variant plus one leave-one-out variant per branch."""
    branches = dedupe_preserve_order(hit_branches)
    if not branches:
        raise ValueError("hit_branches must contain at least one branch")

    variants: list[BranchVariant] = []
    if include_baseline:
        variants.append(BranchVariant(name="baseline_all", hit_branches=list(branches)))

    for branch in branches:
        remaining = [candidate for candidate in branches if candidate != branch]
        variants.append(
            BranchVariant(
                name=f"minus_{_slugify_branch_name(branch)}",
                hit_branches=remaining,
                removed_branch=branch,
            )
        )
    return variants


def build_variant_config(
    base_cfg: dict,
    variant: BranchVariant,
    metadata_branches: Sequence[str],
    output_dir: str | Path,
) -> dict:
    """Return a deep-copied config for one ablation run."""
    cfg = copy.deepcopy(base_cfg)
    data_cfg = cfg.setdefault("data", {})
    training_cfg = cfg.setdefault("training", {})

    run_dir = Path(output_dir) / variant.name
    data_cfg["hit_branches"] = list(variant.hit_branches)
    data_cfg["tpc_branches"] = list(metadata_branches)
    training_cfg["checkpoint_dir"] = str(run_dir / "checkpoints")
    training_cfg["output_path"] = str(run_dir / "model.pt")
    training_cfg.setdefault("save_best_only", True)
    return cfg


def run_ablation(
    cfg_path: str | Path,
    root_files: Sequence[str],
    output_dir: str | Path,
    eval_root_files: Sequence[str] | None = None,
    candidate_hit_branches: Sequence[str] | None = None,
    include_baseline: bool = True,
    log_level: str = "INFO",
) -> Path:
    """Train and evaluate one model per leave-one-out branch variant.

    The script writes a CSV summary to ``output_dir / 'ablation_summary.csv'``.
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from sbn_anomaly.data.root_files import resolve_root_files
    from sbn_anomaly.data.stream_dataset import TPCStreamDataset
    from sbn_anomaly.models.tpc_model import TPCAutoencoder
    from sbn_anomaly.train.tpc_trainer import TPCTrainer
    from sbn_anomaly.utils.metrics import anomaly_score_stats

    setup_logging(log_level)
    logger.info("Loading ablation config from %s", cfg_path)

    cfg_path = Path(cfg_path)
    with cfg_path.open() as fh:
        base_cfg = yaml.safe_load(fh)

    if base_cfg.get("model_type", "").lower() != "tpc":
        raise ValueError("Branch ablation is only supported for model_type='tpc'.")

    data_cfg = base_cfg.get("data", {})
    model_cfg = base_cfg.get("model", {})
    train_cfg = base_cfg.get("training", {})

    train_files = resolve_root_files(root_files)
    eval_files = resolve_root_files(eval_root_files) if eval_root_files else []

    base_hit_branches = list(candidate_hit_branches or data_cfg.get("hit_branches") or [])
    if not base_hit_branches:
        raise ValueError("No hit_branches available to ablate.")

    metadata_branches, base_hit_branches = split_metadata_and_hit_branches(
        data_cfg.get("tpc_branches"),
        base_hit_branches,
    )
    variants = build_leave_one_out_variants(base_hit_branches, include_baseline=include_baseline)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "ablation_summary.csv"

    rows: list[dict[str, object]] = []
    for idx, variant in enumerate(variants, start=1):
        logger.info("[%d/%d] Running %s", idx, len(variants), variant.name)
        variant_cfg = build_variant_config(base_cfg, variant, metadata_branches, output_dir)

        input_dim = int(model_cfg.get("input_dim", 256))
        dataset = TPCStreamDataset(
            file_paths=train_files,
            tree_name=data_cfg.get("tree_name", "sbn_tree"),
            waveform_branch=data_cfg.get("waveform_branch", "tpc_waveform"),
            hit_branches=variant.hit_branches,
            branches=variant_cfg["data"].get("tpc_branches"),
            input_dim=input_dim,
            batch_size=int(data_cfg.get("batch_size_stream", 512)),
            normalize=bool(data_cfg.get("normalize", False)),
            max_events=train_cfg.get("max_events"),
        )
        loader = DataLoader(
            dataset,
            batch_size=int(train_cfg.get("batch_size", 256)),
            num_workers=int(train_cfg.get("num_workers", 0)),
        )
        validation_loader = None
        if eval_files:
            validation_dataset = TPCStreamDataset(
                file_paths=eval_files,
                tree_name=data_cfg.get("tree_name", "sbn_tree"),
                waveform_branch=data_cfg.get("waveform_branch", "tpc_waveform"),
                hit_branches=variant.hit_branches,
                branches=variant_cfg["data"].get("tpc_branches"),
                input_dim=input_dim,
                batch_size=int(data_cfg.get("batch_size_stream", 512)),
                normalize=bool(data_cfg.get("normalize", False)),
                max_events=train_cfg.get("validation_max_events", train_cfg.get("max_events")),
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=int(train_cfg.get("batch_size", 256)),
                num_workers=int(train_cfg.get("num_workers", 0)),
            )
        steps_per_epoch = int(train_cfg.get("steps_per_epoch", 100))

        model = TPCAutoencoder(
            input_dim=input_dim,
            latent_dim=int(model_cfg.get("latent_dim", 32)),
            hidden_dims=tuple(model_cfg.get("hidden_dims", (128, 64))),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
        trainer = TPCTrainer(
            model=model,
            lr=float(train_cfg.get("lr", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
            max_epochs=int(train_cfg.get("max_epochs", 50)),
            checkpoint_dir=variant_cfg["training"]["checkpoint_dir"],
            log_interval=int(train_cfg.get("log_interval", 50)),
            steps_per_epoch=steps_per_epoch,
            anomaly_threshold=train_cfg.get("anomaly_threshold"),
            reconstruction_plot_max_values=int(train_cfg.get("reconstruction_plot_max_values", 50000)),
            save_best_only=bool(train_cfg.get("save_best_only", True)),
        )

        trainer.train(
            loader,
            validation_loader=validation_loader,
            metrics_max_samples=int(train_cfg.get("metrics_max_samples", 20000)),
        )
        output_path = variant_cfg["training"].get("output_path")
        if output_path:
            trainer.save(output_path)

        run_dir = Path(variant_cfg["training"]["checkpoint_dir"]).parent
        trainer.save_training_history(run_dir)
        trainer.save_training_plots(
            run_dir, bins=variant_cfg["training"].get("reconstruction_hist2d_bins")
        )

        eval_metrics = _evaluate_on_files(
            trainer,
            eval_files or train_files,
            data_cfg,
            train_cfg,
            input_dim=input_dim,
            hit_branches=variant.hit_branches,
            branches=variant_cfg["data"].get("tpc_branches"),
        )

        train_loss = float(trainer.history["loss"][-1]) if trainer.history.get("loss") else float("nan")
        train_score_p95 = float(trainer.history["score_p95"][-1]) if trainer.history.get("score_p95") else float("nan")
        train_score_p99 = float(trainer.history["score_p99"][-1]) if trainer.history.get("score_p99") else float("nan")
        train_anomaly_fraction = (
            float(trainer.history["anomaly_fraction_above_threshold"][-1])
            if trainer.history.get("anomaly_fraction_above_threshold")
            else float("nan")
        )

        rows.append(
            {
                "variant": variant.name,
                "removed_branch": variant.removed_branch or "",
                "hit_branches": ";".join(variant.hit_branches),
                "metadata_branches": ";".join(metadata_branches),
                "n_hit_branches": len(variant.hit_branches),
                "train_loss": train_loss,
                "train_score_p95": train_score_p95,
                "train_score_p99": train_score_p99,
                "train_anomaly_fraction_above_threshold": train_anomaly_fraction,
                **eval_metrics,
            }
        )

    fieldnames = list(rows[0].keys()) if rows else []
    with summary_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    logger.info("Saved ablation summary to %s", summary_path)
    return summary_path


def _evaluate_on_files(
    trainer,
    root_files: Sequence[str],
    data_cfg: dict,
    train_cfg: dict,
    *,
    input_dim: int,
    hit_branches: Sequence[str],
    branches: Sequence[str] | None,
) -> dict[str, float]:
    """Evaluate a trained TPC model on a stream of ROOT files."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from sbn_anomaly.data.stream_dataset import TPCStreamDataset
    from sbn_anomaly.utils.metrics import anomaly_score_stats

    dataset = TPCStreamDataset(
        file_paths=root_files,
        tree_name=data_cfg.get("tree_name", "sbn_tree"),
        waveform_branch=data_cfg.get("waveform_branch", "tpc_waveform"),
        hit_branches=list(hit_branches),
        branches=list(branches) if branches is not None else None,
        input_dim=input_dim,
        batch_size=int(data_cfg.get("batch_size_stream", 512)),
        normalize=bool(data_cfg.get("normalize", False)),
        max_events=train_cfg.get("max_events"),
    )
    loader = DataLoader(
        dataset, batch_size=int(train_cfg.get("batch_size", 256)),
        num_workers=int(train_cfg.get("num_workers", 0)),
    )

    trainer.model.eval()
    total_loss = 0.0
    total_events = 0
    scores: list[float] = []

    with torch.no_grad():
        for batch in loader:
            batch_size = trainer._infer_batch_size(batch)
            loss = trainer.compute_loss(batch)
            total_loss += float(loss.item()) * max(batch_size, 1)
            total_events += max(batch_size, 1)

            score_t = trainer.compute_scores(batch)
            if score_t is not None:
                score_arr = score_t.detach().cpu().float().view(-1).numpy()
                scores.extend(score_arr.tolist())

    eval_loss = total_loss / total_events if total_events else float("nan")
    if scores:
        score_stats = anomaly_score_stats(np.asarray(scores, dtype=np.float64))
    else:
        score_stats = {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
        }

    threshold = train_cfg.get("anomaly_threshold")
    if threshold is not None and scores:
        anomaly_fraction = float(np.mean(np.asarray(scores, dtype=np.float64) >= float(threshold)))
    else:
        anomaly_fraction = float("nan")

    return {
        "eval_loss": float(eval_loss),
        "eval_score_mean": float(score_stats["mean"]),
        "eval_score_std": float(score_stats["std"]),
        "eval_score_p50": float(score_stats["p50"]),
        "eval_score_p95": float(score_stats["p95"]),
        "eval_score_p99": float(score_stats["p99"]),
        "eval_anomaly_fraction_above_threshold": float(anomaly_fraction),
    }


def _slugify_branch_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") or "branch"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for branch ablation."""
    parser = argparse.ArgumentParser(description="Run TPC hit-branch ablation.")
    parser.add_argument("--config", required=True, help="Path to the TPC YAML config.")
    parser.add_argument(
        "--root-files",
        nargs="+",
        default=None,
        metavar="PATH",
        help="Training ROOT files or glob patterns.",
    )
    parser.add_argument(
        "--root-file-list",
        nargs="+",
        default=None,
        metavar="FILE",
        help="Text file(s) listing training ROOT files, one per line.",
    )
    parser.add_argument(
        "--eval-root-files",
        nargs="+",
        default=None,
        metavar="PATH",
        help="Optional validation ROOT files or glob patterns.",
    )
    parser.add_argument(
        "--eval-root-file-list",
        nargs="+",
        default=None,
        metavar="FILE",
        help="Optional validation ROOT file list(s), one file path per line.",
    )
    parser.add_argument(
        "--output-dir",
        default="ablation_results/tpc_hit_branches",
        help="Directory for per-variant artifacts and the summary CSV.",
    )
    parser.add_argument(
        "--branches",
        nargs="+",
        default=None,
        metavar="BRANCH",
        help="Optional subset of hit_branches to ablate.",
    )
    parser.add_argument("--no-baseline", action="store_true", help="Skip the full-branch baseline.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    root_inputs: list[str] = []
    if args.root_files:
        root_inputs.extend(args.root_files)
    if args.root_file_list:
        root_inputs.extend(args.root_file_list)

    eval_inputs: list[str] = []
    if args.eval_root_files:
        eval_inputs.extend(args.eval_root_files)
    if args.eval_root_file_list:
        eval_inputs.extend(args.eval_root_file_list)

    if not root_inputs:
        raise SystemExit("Provide --root-files or --root-file-list.")

    if not eval_inputs:
        logger.warning("No eval files provided; using the training files for comparison.")

    summary_path = run_ablation(
        cfg_path=args.config,
        root_files=root_inputs,
        eval_root_files=eval_inputs or None,
        output_dir=args.output_dir,
        candidate_hit_branches=args.branches,
        include_baseline=not args.no_baseline,
        log_level=args.log_level,
    )
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())