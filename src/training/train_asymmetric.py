"""Asymmetric pipeline training: hitfinder-guided crop labeling with ResNet18.

Train a fresh ResNet18 for each LODO fold using AsymmetricCXIDataset:
  - Training: hitfinder assigns per-crop labels based on peak centroid content
  - Validation: blind 224×224 grid with vote-count aggregation
  - Test: cross-detector frames with same aggregation

Usage:
    python -m src.training.train_asymmetric --config configs/supervised/resnet18_asymmetric.yaml
    python -m src.training.train_asymmetric --config ... --folds 1   # single fold smoke test
    python -m src.training.train_asymmetric --config ... --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.evaluation.benchmark import (
    build_lodo_folds,
    build_session_stratified_split,
    format_results_table,
    save_split_artifact,
)
from src.hitfinders import get_hitfinder
from src.training.lodo import _build_intra_split, _train_fold, build_sessions
from src.utils.config import load_config


def main(
    config_path: str | Path,
    folds: list[int] | None = None,
    device: str | None = None,
    intra: bool = False,
    tags: list[str] | None = None,
    resume_training: bool = False,
) -> None:
    cfg = load_config(config_path)
    if tags is not None:
        cfg.setdefault("wandb", {})["tags"] = tags

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Config: {config_path}")

    # GPU hitfinder + multiprocessing DataLoader workers do not mix:
    # the hitfinder holds CUDA tensors that cannot be forked into worker processes.
    num_workers = cfg["training"]["num_workers"]
    if cfg["hitfinder"]["backend"] == "gpu" and num_workers > 0:
        print(
            "WARNING: hitfinder.backend='gpu' is incompatible with num_workers > 0. "
            "Overriding to num_workers=0 for this run to avoid CUDA fork issues."
        )
        num_workers = 0

    hitfinder = get_hitfinder(cfg)

    sessions, session_map = build_sessions(cfg["lodo"])
    total_frames = sum(s["frame_count"] for s in sessions)
    print(f"Sessions: {len(sessions)}  total frames: {total_frames}")
    for det in cfg["lodo"]["detector_dirs"]:
        det_sessions = [s for s in sessions if s["detector"] == det]
        print(f"  {det}: {len(det_sessions)} sessions")

    fold_results: dict[str, dict] = {}

    if intra:
        if len({s["detector"] for s in sessions}) > 1:
            raise ValueError(
                "--intra requires exactly one detector in lodo.detector_dirs. "
                f"Found: {sorted({s['detector'] for s in sessions})}"
            )
        split_artifact = _build_intra_split(sessions)
        fold = {"fold_id": 0, "test_detector": split_artifact["test_detector"]}
        artifacts_dir = Path("checkpoints") / "intra_splits"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        save_split_artifact(split_artifact, artifacts_dir / "fold_0.json")
        result = _train_fold(
            fold,
            split_artifact,
            session_map,
            cfg,
            hitfinder,
            device,
            num_workers_override=num_workers,
            resume_training=resume_training,
        )
        fold_results["fold_0"] = result
    else:
        all_folds = build_lodo_folds()
        if folds is not None:
            all_folds = [f for f in all_folds if f["fold_id"] in folds]

        # Guard: detector names in build_lodo_folds() must match the keys in
        # lodo.detector_dirs, otherwise build_session_stratified_split silently
        # produces an empty cross-detector split and metrics are meaningless.
        known_detectors = {s["detector"] for s in sessions}
        for fold in all_folds:
            if fold["test_detector"] not in known_detectors:
                raise ValueError(
                    f"Fold {fold['fold_id']} test_detector={fold['test_detector']!r} "
                    f"not found in sessions (have: {sorted(known_detectors)}). "
                    "Ensure lodo.detector_dirs keys in the YAML match DETECTORS in benchmark.py."
                )

        artifacts_dir = Path("checkpoints") / "asymmetric_splits"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        for fold in all_folds:
            split_artifact = build_session_stratified_split(
                sessions,
                test_detector=fold["test_detector"],
                fold=fold["fold_id"],
                seed=cfg["seed"],
            )
            save_split_artifact(
                split_artifact,
                artifacts_dir / f"fold_{fold['fold_id']}.json",
            )
            result = _train_fold(
                fold,
                split_artifact,
                session_map,
                cfg,
                hitfinder,
                device,
                num_workers_override=num_workers,
                resume_training=resume_training,
            )
            fold_results[f"fold_{fold['fold_id']}"] = result

    # Summary table over completed folds
    results_for_table: dict = {}
    ap_values = []
    for key, r in fold_results.items():
        results_for_table[key] = {"ap": r["ap"], "test_detector": r["test_detector"]}
        ap_values.append(r["ap"])

    if len(ap_values) > 1:
        results_for_table["mean_ap"] = float(np.mean(ap_values))
        results_for_table["std_ap"] = float(np.std(ap_values, ddof=1))
    elif ap_values:
        results_for_table["mean_ap"] = ap_values[0]
        results_for_table["std_ap"] = float("nan")

    print("\n" + format_results_table(results_for_table))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Asymmetric pipeline training for SFX hitfinder"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=None,
        help="Fold IDs to run (1-4). Omit to run all four.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use: 'cpu' or 'cuda'. Default: auto-detect.",
    )
    parser.add_argument(
        "--intra",
        action="store_true",
        help=(
            "Single-detector mode: 80/10/10 intra-split instead of LODO. "
            "Requires exactly one detector in lodo.detector_dirs. "
            "Use for fast smoke tests — no cross-detector generalization is measured."
        ),
    )
    parser.add_argument(
        "--tags",
        default=None,
        help="Comma-separated wandb tags (overrides wandb.tags in the config YAML).",
    )
    parser.add_argument(
        "--resume-training",
        action="store_true",
        default=False,
        help=(
            "When a checkpoint exists, resume training from it instead of skipping to "
            "evaluation. Restores model weights, optimizer state, and best val F1. "
            "Has no effect when no checkpoint is present."
        ),
    )
    args = parser.parse_args()
    tags = [t.strip() for t in args.tags.split(",")] if args.tags else None
    main(
        args.config,
        folds=args.folds,
        device=args.device,
        intra=args.intra,
        tags=tags,
        resume_training=args.resume_training,
    )
