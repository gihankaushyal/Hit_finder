"""Asymmetric pipeline training: hitfinder-guided crop labeling with ResNet18.

Train a fresh ResNet18 for each LODO fold using AsymmetricCXIDataset:
  - Training: hitfinder assigns per-crop labels based on peak centroid content
  - Validation: blind 224×224 grid with vote-count aggregation
  - Test: cross-detector frames with same aggregation

Run naming convention (--run-name-prefix, REQUIRED, no default):

    <backbone>-asymmetric-v<N>

    <N> identifies the pipeline generation that produced this run, incremented
    whenever the preprocessing/pipeline changes in a way that invalidates a
    direct numeric comparison with the previous generation (e.g. v1 =
    pre-frame-cache, v2 = frame-cache-backed). It is NOT a free-text label —
    scripts and aggregate_lodo_results.py parse the trailing "-v<N>".

    Example: --run-name-prefix resnet18-asymmetric-v2

    This prefix becomes both the wandb run id/name
    (resnet18-asymmetric-v2-fold{N}-seed{S}) and the checkpoint directory
    (checkpoints/resnet18-asymmetric-v2-fold{N}-seed{S}/) — so two different
    pipeline generations can never collide under the same name. If a
    checkpoint already exists under the resolved name, the script exits and
    asks you to pass --resume-training (continue) or --override-training
    (discard and restart).

Usage:
    python -m src.training.train_asymmetric --config configs/supervised/resnet18_asymmetric.yaml --run-name-prefix resnet18-asymmetric-v2
    python -m src.training.train_asymmetric --config ... --run-name-prefix resnet18-asymmetric-v2 --folds 1   # single fold smoke test
    python -m src.training.train_asymmetric --config ... --run-name-prefix resnet18-asymmetric-v2 --device cpu
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch

from src.data.frame_cache import frame_cache_from_cfg
from src.evaluation.benchmark import (
    build_lodo_folds,
    build_session_stratified_split,
    format_results_table,
    save_split_artifact,
)
from src.hitfinders import get_hitfinder
from src.training.lodo import _build_intra_split, _train_fold, build_sessions
from src.utils.config import load_config

_RUN_NAME_PREFIX_RE = re.compile(r"^[a-z0-9]+-asymmetric-v\d+$")


def _validate_run_name_prefix(prefix: str) -> None:
    if not _RUN_NAME_PREFIX_RE.match(prefix):
        raise SystemExit(
            f"Invalid --run-name-prefix: {prefix!r}\n\n"
            "Run names must follow the convention:\n\n"
            "    <backbone>-asymmetric-v<N>\n\n"
            "<N> identifies the pipeline generation that produced this run, incremented\n"
            "whenever the preprocessing/pipeline changes in a way that invalidates a direct\n"
            "numeric comparison with the previous generation (e.g. v1 = pre-frame-cache,\n"
            "v2 = frame-cache-backed).\n\n"
            "Example: --run-name-prefix resnet18-asymmetric-v2"
        )


def _check_checkpoint_collisions(
    fold_ids: list[int],
    cfg: dict,
    run_name_prefix: str,
    resume_training: bool,
    override_training: bool,
) -> None:
    seed = cfg["seed"]
    run_suffix = cfg.get("wandb", {}).get("run_suffix", "")
    for fold_id in fold_ids:
        run_name = f"{run_name_prefix}-fold{fold_id}-seed{seed}{run_suffix}"
        ckpt_path = Path("checkpoints") / run_name / "best.pt"
        if not ckpt_path.exists():
            continue
        if override_training:
            ckpt_path.unlink()
            results_path = ckpt_path.parent / "results.json"
            if results_path.exists():
                results_path.unlink()
        elif not resume_training:
            raise SystemExit(
                f"Checkpoint already exists for fold {fold_id}: {ckpt_path}\n\n"
                "Re-run with --resume-training to continue training from it, or "
                f"--override-training to discard it and start fold {fold_id} from scratch."
            )


def main(
    config_path: str | Path,
    run_name_prefix: str,
    folds: list[int] | None = None,
    device: str | None = None,
    intra: bool = False,
    tags: list[str] | None = None,
    resume_training: bool = False,
    override_training: bool = False,
    cache_root: str | None = None,
    cache_nvme: str | None = None,
    no_cache: bool = False,
) -> None:
    _validate_run_name_prefix(run_name_prefix)
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

    cache_cfg = cfg.setdefault("cache", {})
    if cache_root is not None:
        cache_cfg["root"] = cache_root
    if cache_nvme is not None:
        cache_cfg["nvme_root"] = cache_nvme
    if no_cache:
        cache_cfg["enabled"] = False
    frame_cache = frame_cache_from_cfg(cfg)
    print(
        f"[cache] {'roots: ' + ', '.join(str(r) for r in frame_cache.roots) if frame_cache else 'disabled — computing live'}"
    )

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
        _check_checkpoint_collisions(
            [0], cfg, run_name_prefix, resume_training, override_training
        )
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
            run_name_prefix=run_name_prefix,
            frame_cache=frame_cache,
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

        _check_checkpoint_collisions(
            [f["fold_id"] for f in all_folds],
            cfg,
            run_name_prefix,
            resume_training,
            override_training,
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
                run_name_prefix=run_name_prefix,
                frame_cache=frame_cache,
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
        "--run-name-prefix",
        required=True,
        help=(
            "Required. Naming convention: <backbone>-asymmetric-v<N> "
            "(e.g. resnet18-asymmetric-v2). Becomes the wandb run id/name and "
            "checkpoint directory for every fold. See module docstring for details."
        ),
    )
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
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume-training",
        action="store_true",
        default=False,
        help=(
            "When a checkpoint exists for the resolved run name, resume training from "
            "it instead of exiting. Restores model weights, optimizer state, and best "
            "val F1. Has no effect when no checkpoint is present."
        ),
    )
    resume_group.add_argument(
        "--override-training",
        action="store_true",
        default=False,
        help=(
            "When a checkpoint exists for the resolved run name, discard it "
            "(best.pt and results.json) and start that fold from scratch under the "
            "same run name. Mutually exclusive with --resume-training."
        ),
    )
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--cache-nvme", default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    tags = [t.strip() for t in args.tags.split(",")] if args.tags else None
    main(
        args.config,
        args.run_name_prefix,
        folds=args.folds,
        device=args.device,
        intra=args.intra,
        tags=tags,
        resume_training=args.resume_training,
        override_training=args.override_training,
        cache_root=args.cache_root,
        cache_nvme=args.cache_nvme,
        no_cache=args.no_cache,
    )
