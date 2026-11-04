"""Fine-tune an MAE-pretrained ViT-S through the unchanged asymmetric pipeline.

Run naming convention (--run-name-prefix, REQUIRED, no default):

    <backbone>-mae-v<N>          e.g. vits16-mae-v2

    <N> is the pipeline generation, as defined in src/training/run_naming.py.
    The mode is inserted before the version, so the run names are
    vits16-mae-finetune-v2-fold{N}-seed{S} and, with --linear-probe,
    vits16-mae-probe-v2-fold{N}-seed{S}. If best.pt already exists under the
    resolved name, the script exits and asks for --resume-training (continue),
    --override-training (discard best.pt and results.json, then restart; the old
    W&B run is kept and tagged 'overridden', the restart logs to <run_name>-o<N>) or
    --inference-only (evaluate best.pt without training; no pretrain checkpoint needed).

Usage:
    python -m src.training.train_ssl_finetune --config configs/ssl/mae_finetune.yaml \
        --fold 1 --run-name-prefix vits16-mae-v2 \
        --pretrain-checkpoint checkpoints/mae-vits16-v2-fold1-seed42/last.pt
    # add --linear-probe to freeze the encoder
    # recipe experiments (tag the run with --run-suffix <name>; recipe knobs live in the config):
    #   --config configs/ssl/mae_finetune_careful.yaml --run-suffix careful
    #   --head-checkpoint checkpoints/vits16-mae-probe-v2-fold4-seed42/best.pt --run-suffix lpft
    #   --linear-probe --random-init --run-suffix randinit
    # re-evaluate a finished run (no --pretrain-checkpoint needed):
    python -m src.training.train_ssl_finetune --config configs/ssl/mae_finetune.yaml \
        --fold 1 --run-name-prefix vits16-mae-v2 --inference-only
"""

from __future__ import annotations

import argparse
import re
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
    effective_resume,
    expand_finetune_prefix,
    fold_run_name,
    validate_run_name_prefix,
)
from src.training.inference_results import INFERENCE_RESULTS_NAME, RESULTS_NAME
from src.training.wandb_identity import override_hook_from_cfg
from src.utils.config import load_config

SPLIT_DIR = Path("checkpoints") / "asymmetric_splits"
RUN_SUFFIX_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def non_default_recipe_keys(cfg: dict) -> list[str]:
    """Names of recipe/diagnostic settings that differ from the v2 fine-tune defaults.

    A run that sets any of these must carry a --run-suffix, otherwise it would write into
    (or, with --override-training, delete) the directory of the genuine v2 run.
    """
    training, ssl = cfg.get("training", {}), cfg.get("ssl", {})
    checks = {
        "warmup_epochs": training.get("warmup_epochs", 0) != 0,
        "lr_schedule": training.get("lr_schedule", "constant") != "constant",
        "layer_decay": training.get("layer_decay", 1.0) != 1.0,
        "wd_skip_norm_bias": bool(training.get("wd_skip_norm_bias", False)),
        "log_val_breakdown": bool(training.get("log_val_breakdown", False)),
        "save_epochs": bool(training.get("save_epochs")),
        "drop_path_rate": ssl.get("drop_path_rate", 0.0) != 0.0,
        "head_init": ssl.get("head_init", "default") != "default",
    }
    return [k for k, differs in checks.items() if differs]


def check_head_checkpoint(path: str | Path) -> None:
    """Fail before any setup (and before --override-training deletes a run) on a bad head file."""
    from src.models.ssl import HEAD_KEYS

    if not Path(path).exists():
        raise SystemExit(f"--head-checkpoint not found: {path}")
    sd = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    keys = {k for k in sd.get("model_state_dict", {}) if k.startswith("head.")}
    if keys != HEAD_KEYS:
        raise SystemExit(f"{path}: head keys {sorted(keys)} != {sorted(HEAD_KEYS)}")


def build_finetune_model_builder(
    cfg: dict,
    pretrain_checkpoint: str | Path | None,
    linear_probe: bool = False,
    head_checkpoint: str | Path | None = None,
) -> Callable[[], nn.Module]:
    """Return a zero-argument callable that builds a ViTClassifier from the MAE checkpoint.

    `pretrain_checkpoint=None` builds the same architecture without loading MAE weights
    (inference: best.pt supplies every weight).
    """

    def _builder() -> nn.Module:
        return build_ssl_classifier(
            cfg,
            mae_checkpoint=pretrain_checkpoint,
            freeze_encoder=linear_probe,
            head_checkpoint=head_checkpoint,
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
    continued: set[str] | None = None,
) -> str:
    """Validate the prefix, apply the checkpoint gate and return the expanded prefix.

    `continued`, when given, is updated in place with the run names whose override
    this launch already applied (resume them via `effective_resume`).

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
    result = check_checkpoint_collisions(
        {fold_id: fold_run_name(prefix, fold_id, cfg["seed"], run_suffix)},
        "best.pt",
        resume_training,
        override_training,
        extra_delete=(RESULTS_NAME, INFERENCE_RESULTS_NAME),
        dry_run=dry_run,
        inference_only=inference_only,
        on_override=override_hook_from_cfg(cfg),
    )
    if continued is not None:
        continued |= result
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
    p.add_argument(
        "--head-checkpoint",
        default=None,
        help="Copy head.* weights from this classifier checkpoint (e.g. a probe best.pt) "
        "before training: LP-FT. Name the run with --run-suffix.",
    )
    p.add_argument(
        "--random-init",
        action="store_true",
        help="Skip loading MAE weights: the encoder keeps its random init (control for "
        "'does pretraining help?'; use with --linear-probe and --run-suffix). The init is "
        "the library default of ViTClassifier, not the MAE pretraining (xavier) init.",
    )
    p.add_argument(
        "--run-suffix",
        default=None,
        help="Appended to the run name as '-<suffix>' (e.g. lpft -> ...-seed42-lpft), "
        "overriding wandb.run_suffix. "
        "Keeps the vN prefix convention while tagging the experimental factor.",
    )
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
            "results.json and results.inference.json and start that fold from scratch under the same run name. The old W&B run is kept "
            "and tagged 'overridden'; the restart logs to '<run_name>-o<N>'."
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
    if args.random_init:
        if args.pretrain_checkpoint is not None:
            print(
                "[random-init] ignoring --pretrain-checkpoint: the encoder keeps its random init"
            )
        if not args.linear_probe:
            print(
                "[random-init] WARNING: without --linear-probe this is a full fine-tune from scratch"
            )
        args.pretrain_checkpoint = None
    elif not args.inference_only and args.pretrain_checkpoint is None:
        p.error(
            "--pretrain-checkpoint is required unless --inference-only or --random-init"
        )
    if args.run_suffix is not None and not RUN_SUFFIX_RE.fullmatch(
        args.run_suffix.lstrip("-")
    ):
        p.error(f"--run-suffix {args.run_suffix!r} must match {RUN_SUFFIX_RE.pattern}")
    if (args.random_init or args.head_checkpoint) and not args.run_suffix:
        p.error(
            "--random-init / --head-checkpoint need --run-suffix so the run cannot "
            "collide with (or override) a real v2 run"
        )
    if args.head_checkpoint and not args.inference_only:
        check_head_checkpoint(args.head_checkpoint)

    cfg = load_config(args.config)
    recipe_keys = non_default_recipe_keys(cfg)
    if recipe_keys and not args.run_suffix:
        p.error(
            f"{args.config} sets non-default recipe keys ({', '.join(recipe_keys)}); pass "
            "--run-suffix (e.g. careful) so the run cannot collide with a real v2 run"
        )
    if args.run_suffix is not None:
        # argparse reads a value starting with "-" as a flag, so accept "lpft" and add the dash.
        suffix = args.run_suffix.lstrip("-")
        cfg.setdefault("wandb", {})["run_suffix"] = f"-{suffix}"

    validate_run_name_prefix(
        args.run_name_prefix,
        SSL_FINETUNE_PREFIX_RE,
        SSL_FINETUNE_CONVENTION,
        SSL_FINETUNE_EXAMPLE,
    )
    probe = args.linear_probe
    pretrain_epoch = None
    if not args.inference_only and not args.random_init:
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
        cfg["pretrain_checkpoint"] = (
            None if args.pretrain_checkpoint is None else str(args.pretrain_checkpoint)
        )
        cfg["pretrain_epoch"] = pretrain_epoch
        cfg["head_checkpoint"] = args.head_checkpoint
        cfg["random_init"] = args.random_init

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
    continued: set[str] = set()
    prepare_finetune_run(
        args.run_name_prefix,
        args.fold,
        cfg,
        linear_probe=probe,
        resume_training=args.resume_training,
        override_training=args.override_training,
        inference_only=args.inference_only,
        continued=continued,
    )
    run_name = fold_run_name(
        prefix, args.fold, cfg["seed"], cfg.get("wandb", {}).get("run_suffix", "")
    )
    result = _train_fold(
        fold,
        split_artifact,
        session_map,
        cfg,
        hitfinder,
        device,
        resume_training=effective_resume(args.resume_training, run_name, continued),
        model_builder=build_finetune_model_builder(
            cfg,
            None if args.inference_only else args.pretrain_checkpoint,
            linear_probe=probe,
            head_checkpoint=None if args.inference_only else args.head_checkpoint,
        ),
        run_name_prefix=prefix,
        extra_results={
            "track": "ssl",
            "probe": probe,
            **(
                {}
                if args.inference_only
                else {
                    "pretrain_checkpoint": (
                        None
                        if args.pretrain_checkpoint is None
                        else str(args.pretrain_checkpoint)
                    ),
                    "pretrain_epoch": pretrain_epoch,
                    "head_checkpoint": args.head_checkpoint,
                    "random_init": args.random_init,
                }
            ),
        },
        frame_cache=frame_cache,
        inference_only=args.inference_only,
    )
    print(result)


if __name__ == "__main__":
    main()
