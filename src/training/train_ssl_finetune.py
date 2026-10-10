"""Fine-tune an MAE-pretrained ViT-S through the unchanged asymmetric pipeline.

Run naming convention (--run-name-prefix, REQUIRED, no default):

    <backbone>-mae-v<N>          e.g. vits16-mae-v2

    <N> is the pipeline generation, as defined in src/training/run_naming.py.
    The mode is inserted before the version, so the run names are
    vits16-mae-finetune-v2-fold{N}-seed{S} and, with --linear-probe,
    vits16-mae-probe-v2-fold{N}-seed{S}. If best.pt already exists under the
    resolved name, the script exits and asks for --resume-training (continue),
    --override-training (discard best.pt and results.json, then restart) or
    --inference-only (evaluate best.pt without training; no pretrain checkpoint needed).

Usage:
    python -m src.training.train_ssl_finetune --config configs/ssl/mae_finetune.yaml \
        --fold 1 --run-name-prefix vits16-mae-v2 \
        --pretrain-checkpoint checkpoints/mae-vits16-v2-fold1-seed42/last.pt
    # add --linear-probe to freeze the encoder
    # re-evaluate a finished run (no --pretrain-checkpoint needed):
    python -m src.training.train_ssl_finetune --config configs/ssl/mae_finetune.yaml \
        --fold 1 --run-name-prefix vits16-mae-v2 --inference-only
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn

from src.data.frame_cache import frame_cache_from_cfg
from src.evaluation.benchmark import (
    build_lodo_folds,
    build_session_stratified_split,
    save_split_artifact,
)
from src.hitfinders import get_hitfinder
from src.models.ssl import build_ssl_classifier
from src.training.lodo import _train_fold, build_sessions
from src.training.run_naming import (
    SSL_FINETUNE_CONVENTION,
    SSL_FINETUNE_EXAMPLE,
    SSL_FINETUNE_PREFIX_RE,
    check_checkpoint_collisions,
    expand_finetune_prefix,
    fold_run_name,
    validate_run_name_prefix,
)
from src.utils.config import load_config

SPLIT_DIR = Path("checkpoints") / "asymmetric_splits"


def build_finetune_model_builder(
    cfg: dict,
    pretrain_checkpoint: str | Path | None,
    linear_probe: bool = False,
) -> Callable[[], nn.Module]:
    """Return a zero-argument callable that builds a ViTClassifier from the MAE checkpoint.

    `pretrain_checkpoint=None` builds the same architecture without loading MAE weights
    (inference: best.pt supplies every weight).
    """

    def _builder() -> nn.Module:
        return build_ssl_classifier(
            cfg, mae_checkpoint=pretrain_checkpoint, freeze_encoder=linear_probe
        )

    return _builder


def prepare_finetune_run(
    run_name_prefix: str,
    fold_id: int,
    cfg: dict,
    linear_probe: bool = False,
    resume_training: bool = False,
    override_training: bool = False,
    inference_only: bool = False,
    dry_run: bool = False,
) -> str:
    """Validate the prefix, apply the checkpoint gate and return the expanded prefix.

    With `dry_run` the gate only checks (exits on an unresolved collision) and
    deletes nothing; call again without it right before training.

    The return value (e.g. ``vits16-mae-finetune-v2``) is what `_train_fold`
    takes as `run_name_prefix`.
    """
    validate_run_name_prefix(
        run_name_prefix,
        SSL_FINETUNE_PREFIX_RE,
        SSL_FINETUNE_CONVENTION,
        SSL_FINETUNE_EXAMPLE,
    )
    prefix = expand_finetune_prefix(run_name_prefix, linear_probe)
    run_suffix = cfg.get("wandb", {}).get("run_suffix", "")
    check_checkpoint_collisions(
        {fold_id: fold_run_name(prefix, fold_id, cfg["seed"], run_suffix)},
        "best.pt",
        resume_training,
        override_training,
        extra_delete=("results.json",),
        dry_run=dry_run,
        inference_only=inference_only,
    )
    return prefix


def read_pretrain_epoch(pretrain_checkpoint: str | Path) -> int:
    """Epoch stored in an MAE pretrain checkpoint (exposes a partial pretrain).

    mmap=True maps the file instead of reading it, so the model and optimizer tensors
    are never pulled into memory just to read one integer.
    """
    state = torch.load(
        pretrain_checkpoint, map_location="cpu", weights_only=True, mmap=True
    )
    if "epoch" not in state:
        raise SystemExit(
            f"{pretrain_checkpoint} has no 'epoch' key — it is not a train_ssl_pretrain "
            "last.pt checkpoint."
        )
    return int(state["epoch"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument(
        "--pretrain-checkpoint",
        default=None,
        help=(
            "MAE pretrain last.pt to initialise from. Required unless "
            "--inference-only (best.pt already holds every weight)."
        ),
    )
    p.add_argument("--linear-probe", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument(
        "--run-name-prefix",
        required=True,
        help=(
            "Required. Naming convention: <backbone>-mae-v<N> (e.g. vits16-mae-v2). "
            "Expanded to <backbone>-mae-finetune-v<N> or, with --linear-probe, "
            "<backbone>-mae-probe-v<N>. See module docstring for details."
        ),
    )
    resume_group = p.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume-training",
        action="store_true",
        help=(
            "When best.pt exists for the resolved run name, resume training from "
            "it instead of exiting. Has no effect when no checkpoint is present."
        ),
    )
    resume_group.add_argument(
        "--override-training",
        action="store_true",
        help=(
            "When best.pt exists for the resolved run name, discard it and "
            "results.json and start that fold from scratch under the same run name."
        ),
    )
    resume_group.add_argument(
        "--inference-only",
        action="store_true",
        help=(
            "Evaluate the existing best.pt on the in-domain and cross-detector sets "
            "without training. Writes results.json if the run has none, else "
            "results.inference.json. Needs no --pretrain-checkpoint."
        ),
    )
    p.add_argument(
        "--cache-root",
        default=None,
        help="Override cache.root — the permanent NFS frame cache directory.",
    )
    p.add_argument(
        "--cache-nvme",
        default=None,
        help="Override cache.nvme_root — the local NVMe frame cache tier, "
        "checked before cache.root.",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the frame cache and recompute assembly/hitfinder/GCN live.",
    )
    args = p.parse_args()
    if not args.inference_only and args.pretrain_checkpoint is None:
        p.error("--pretrain-checkpoint is required unless --inference-only is given")

    cfg = load_config(args.config)

    validate_run_name_prefix(
        args.run_name_prefix,
        SSL_FINETUNE_PREFIX_RE,
        SSL_FINETUNE_CONVENTION,
        SSL_FINETUNE_EXAMPLE,
    )
    probe = args.linear_probe
    pretrain_epoch = None
    if not args.inference_only:
        # Read the pretrain checkpoint early so a bad path fails before any setup.
        pretrain_epoch = read_pretrain_epoch(args.pretrain_checkpoint)
        print(f"[pretrain] {args.pretrain_checkpoint} — stored epoch {pretrain_epoch}")
    prefix = prepare_finetune_run(
        args.run_name_prefix,
        args.fold,
        cfg,
        linear_probe=probe,
        resume_training=args.resume_training,
        override_training=args.override_training,
        inference_only=args.inference_only,
        dry_run=True,  # fail fast; the real override happens just before training
    )
    if not args.inference_only:
        # _train_fold passes cfg to wandb.init(config=...), so these land in the run config.
        cfg["pretrain_checkpoint"] = str(args.pretrain_checkpoint)
        cfg["pretrain_epoch"] = pretrain_epoch

    cache_cfg = cfg.setdefault("cache", {})
    if args.cache_root is not None:
        cache_cfg["root"] = args.cache_root
    if args.cache_nvme is not None:
        cache_cfg["nvme_root"] = args.cache_nvme
    if args.no_cache:
        cache_cfg["enabled"] = False
    frame_cache = frame_cache_from_cfg(cfg)
    print(
        f"[cache] {'roots: ' + ', '.join(str(r) for r in frame_cache.roots) if frame_cache else 'disabled — computing live'}"
    )
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

    # Everything that can fail before training has succeeded: only now discard
    # the previous attempt's checkpoint.
    prepare_finetune_run(
        args.run_name_prefix,
        args.fold,
        cfg,
        linear_probe=probe,
        resume_training=args.resume_training,
        override_training=args.override_training,
        inference_only=args.inference_only,
    )
    result = _train_fold(
        fold,
        split_artifact,
        session_map,
        cfg,
        hitfinder,
        device,
        resume_training=args.resume_training,
        model_builder=build_finetune_model_builder(
            cfg,
            None if args.inference_only else args.pretrain_checkpoint,
            linear_probe=probe,
        ),
        run_name_prefix=prefix,
        extra_results={
            "track": "ssl",
            "probe": probe,
            **(
                {}
                if args.inference_only
                else {
                    "pretrain_checkpoint": str(args.pretrain_checkpoint),
                    "pretrain_epoch": pretrain_epoch,
                }
            ),
        },
        frame_cache=frame_cache,
        inference_only=args.inference_only,
    )
    print(result)


if __name__ == "__main__":
    main()
