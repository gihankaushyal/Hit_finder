"""Fine-tune an MAE-pretrained ViT-S through the unchanged asymmetric pipeline.

Usage:
    python -m src.training.train_ssl_finetune --config configs/ssl/mae_finetune.yaml \
        --fold 1 --pretrain-checkpoint checkpoints/mae-vits16-fold1-seed42/last.pt
    # add --linear-probe to freeze the encoder
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn

from src.evaluation.benchmark import (
    build_lodo_folds,
    build_session_stratified_split,
    save_split_artifact,
)
from src.hitfinders import get_hitfinder
from src.models.ssl import build_ssl_classifier
from src.training.lodo import _train_fold, build_sessions
from src.utils.config import load_config

SPLIT_DIR = Path("checkpoints") / "asymmetric_splits"


def build_finetune_model_builder(
    cfg: dict,
    pretrain_checkpoint: str | Path,
    linear_probe: bool = False,
) -> Callable[[], nn.Module]:
    """Return a zero-argument callable that builds a ViTClassifier from the MAE checkpoint."""

    def _builder() -> nn.Module:
        return build_ssl_classifier(
            cfg, mae_checkpoint=pretrain_checkpoint, freeze_encoder=linear_probe
        )

    return _builder


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--pretrain-checkpoint", required=True)
    p.add_argument("--linear-probe", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--resume-training", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    hitfinder = get_hitfinder(cfg)
    sessions, session_map = build_sessions(cfg["lodo"])

    fold = next(f for f in build_lodo_folds() if f["fold_id"] == args.fold)
    split_artifact = build_session_stratified_split(
        sessions,
        test_detector=fold["test_detector"],
        fold=fold["fold_id"],
        seed=cfg["seed"],
    )
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    save_split_artifact(split_artifact, SPLIT_DIR / f"fold_{args.fold}.json")

    probe = args.linear_probe
    prefix = "vits16-mae-probe" if probe else "vits16-mae-finetune"
    result = _train_fold(
        fold,
        split_artifact,
        session_map,
        cfg,
        hitfinder,
        device,
        resume_training=args.resume_training,
        model_builder=build_finetune_model_builder(
            cfg, args.pretrain_checkpoint, linear_probe=probe
        ),
        run_name_prefix=prefix,
        extra_results={
            "track": "ssl",
            "probe": probe,
            "pretrain_checkpoint": str(args.pretrain_checkpoint),
        },
    )
    print(result)


if __name__ == "__main__":
    main()
