"""Leave-One-Detector-Out (LODO) training and evaluation for SFX hitfinding.

Trains a fresh ResNet18 for each of the 4 LODO folds (one detector held out
per fold). Uses the full benchmark.py session-split infrastructure.

Session granularity: one CXI file = one session.
  AGIPD:       5 files × 4000 frames = 20 000 frames, 5 sessions
  JUNGFRAU_4M: 10 files × 2000 frames = 20 000 frames, 10 sessions
  ePix10k:     5 files × 4000 frames = 20 000 frames, 5 sessions
  Eiger4M:     5 files × 4000 frames = 20 000 frames, 5 sessions
  Total: 25 sessions, 80 000 frames

Usage:
    /home/gketawal/.conda/envs/sfx-hitfinder/bin/python \\
        scripts/train_lodo.py \\
        --config configs/supervised/resnet18_lodo.yaml

    # single fold for smoke-testing:
    scripts/train_lodo.py --config ... --folds 1

SLURM:
    sbatch scripts/submit_lodo.sh
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataloader import cxi_session_loader
from src.evaluation.benchmark import (
    SPLIT_CROSS_DETECTOR,
    SPLIT_IN_DOMAIN_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    build_lodo_folds,
    build_session_stratified_split,
    format_results_table,
    run_on_loader,
    save_split_artifact,
)
from src.models.supervised import build_supervised_model
from src.training.train_supervised import _set_seeds, evaluate, train_one_epoch
from src.utils.config import load_config


def build_sessions(
    lodo_cfg: dict,
) -> tuple[list[dict], dict[str, Path]]:
    """Discover CXI files under each detector dir and build session records.

    Returns:
        sessions:    List of dicts with keys session_id, detector, frame_count.
        session_map: Mapping from session_id to absolute CXI Path.
    """
    sessions: list[dict] = []
    session_map: dict[str, Path] = {}
    pattern = lodo_cfg.get("cxi_pattern", "compressed*.cxi")
    label_key = lodo_cfg.get("label_key", "entry_1/labels/hit")

    for detector, dir_str in lodo_cfg["detector_dirs"].items():
        det_dir = Path(dir_str)
        for cxi in sorted(det_dir.glob(pattern)):
            with h5py.File(cxi, "r") as f:
                n_frames = int(f[label_key].shape[0])
            sid = f"{detector}_{cxi.stem}"
            sessions.append(
                {"session_id": sid, "detector": detector, "frame_count": n_frames}
            )
            session_map[sid] = cxi

    return sessions, session_map


def _make_loader(
    split_artifact: dict,
    split_name: str,
    session_map: dict[str, Path],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    label_key: str = "entry_1/labels/hit",
):
    ids = [
        sid
        for sid, s in split_artifact["splits"].items()
        if s == split_name
    ]
    return cxi_session_loader(session_map, ids, batch_size, num_workers, shuffle, label_key=label_key)


def _train_fold(
    fold: dict,
    split_artifact: dict,
    session_map: dict[str, Path],
    cfg: dict,
    device: str,
) -> dict:
    """Train one LODO fold from scratch; return cross-detector and in-domain metrics."""
    import wandb

    backbone = cfg["model"]["backbone"]
    seed = cfg["seed"]
    fold_id = fold["fold_id"]
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]
    epochs = cfg["training"]["epochs"]
    patience = cfg["training"].get("early_stopping_patience", 10)
    run_name = f"{backbone}-lodo-fold{fold_id}-seed{seed}"

    label_key = cfg["lodo"].get("label_key", "entry_1/labels/hit")
    train_dl     = _make_loader(split_artifact, SPLIT_TRAIN,          session_map, batch_size, num_workers, shuffle=True,  label_key=label_key)
    val_dl       = _make_loader(split_artifact, SPLIT_VAL,            session_map, batch_size, num_workers, shuffle=False, label_key=label_key)
    in_domain_dl = _make_loader(split_artifact, SPLIT_IN_DOMAIN_TEST, session_map, batch_size, num_workers, shuffle=False, label_key=label_key)
    cross_dl     = _make_loader(split_artifact, SPLIT_CROSS_DETECTOR, session_map, batch_size, num_workers, shuffle=False, label_key=label_key)

    n_train   = len(train_dl.dataset)
    n_val     = len(val_dl.dataset)
    n_indomain = len(in_domain_dl.dataset)
    n_cross   = len(cross_dl.dataset)

    print(
        f"\n{'='*60}\n"
        f"Fold {fold_id}  |  held-out: {fold['test_detector']}\n"
        f"  train={n_train}  val={n_val}  in_domain_test={n_indomain}  cross={n_cross}\n"
        f"{'='*60}"
    )

    _set_seeds(seed)
    model = build_supervised_model(
        backbone=backbone,
        pretrained=cfg["model"]["pretrained"],
        num_classes=cfg["model"]["num_classes"],
    ).to(device)

    ckpt_dir = Path("checkpoints") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / "best.pt"
    resume_eval_only = ckpt_path.exists()

    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"].get("entity"),
        id=run_name,        # deterministic ID so resume="allow" finds the same run
        name=run_name,
        config={**cfg, "fold_id": fold_id, "test_detector": fold["test_detector"]},
        tags=cfg["wandb"].get("tags", []),
        resume="allow",
    )

    if resume_eval_only:
        print(f"  Checkpoint found at {ckpt_path} — skipping training, resuming from evaluation.")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"],
        )
        criterion = nn.CrossEntropyLoss()

        best_f1 = -1.0
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            train_m = train_one_epoch(model, train_dl, optimizer, criterion, device)
            val_m = evaluate(model, val_dl, criterion, device)

            print(
                f"  Epoch {epoch:3d}/{epochs}  "
                f"train_loss={train_m['loss']:.4f}  "
                f"val_loss={val_m['loss']:.4f}  "
                f"val_AP={val_m['ap']:.4f}  val_F1={val_m['f1']:.4f}"
            )
            wandb.log({
                "epoch": epoch,
                "train/loss": train_m["loss"],
                "val/loss": val_m["loss"],
                "val/ap": val_m["ap"],
                "val/auc": val_m["auc"],
                "val/f1": val_m["f1"],
            })

            if val_m["f1"] > best_f1:
                best_f1 = val_m["f1"]
                epochs_no_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_f1": best_f1,
                        "backbone": backbone,
                        "num_classes": cfg["model"]["num_classes"],
                    },
                    ckpt_path,
                )
                print(f"    → checkpoint saved (val F1={best_f1:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                    break

    # Evaluate best checkpoint on in-domain and cross-detector test sets
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    ckpt_backbone = ckpt.get("backbone")
    ckpt_num_classes = ckpt.get("num_classes")
    if ckpt_backbone is not None and ckpt_backbone != backbone:
        raise RuntimeError(
            f"Checkpoint backbone={ckpt_backbone!r} does not match config backbone={backbone!r}. "
            "Delete the checkpoint or update the config."
        )
    if ckpt_num_classes is not None and ckpt_num_classes != cfg["model"]["num_classes"]:
        raise RuntimeError(
            f"Checkpoint num_classes={ckpt_num_classes} does not match config "
            f"num_classes={cfg['model']['num_classes']}. Delete the checkpoint or update the config."
        )
    model.load_state_dict(ckpt["model_state_dict"])

    in_domain_m = run_on_loader(model, in_domain_dl, device)
    cross_m = run_on_loader(model, cross_dl, device)

    print(
        f"  In-domain test:    AP={in_domain_m['ap']:.4f}  AUC={in_domain_m['auc_roc']:.4f}  F1={in_domain_m['f1']:.4f}"
    )
    print(
        f"  Cross-detector:    AP={cross_m['ap']:.4f}  AUC={cross_m['auc_roc']:.4f}  F1={cross_m['f1']:.4f}"
    )

    wandb.log({
        "in_domain/ap":    in_domain_m["ap"],
        "in_domain/auc":   in_domain_m["auc_roc"],
        "in_domain/f1":    in_domain_m["f1"],
        "cross/ap":        cross_m["ap"],
        "cross/auc":       cross_m["auc_roc"],
        "cross/f1":        cross_m["f1"],
    })
    wandb.finish()

    result = {
        "fold_id": fold_id,
        "test_detector": fold["test_detector"],
        "cross": {
            "ap":      cross_m["ap"],
            "auc_roc": cross_m["auc_roc"],
            "f1":      cross_m["f1"],
            "threshold": cross_m["threshold"],
        },
        "in_domain": {
            "ap":      in_domain_m["ap"],
            "auc_roc": in_domain_m["auc_roc"],
            "f1":      in_domain_m["f1"],
            "threshold": in_domain_m["threshold"],
        },
    }
    results_path = ckpt_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved → {results_path}")

    return {
        "test_detector": fold["test_detector"],
        "ap": cross_m["ap"],
        "in_domain_ap": in_domain_m["ap"],
        "auc_roc": cross_m["auc_roc"],
        "f1": cross_m["f1"],
    }


def main(config_path: str | Path, folds: list[int] | None = None) -> None:
    cfg = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Config: {config_path}")

    sessions, session_map = build_sessions(cfg["lodo"])
    total_frames = sum(s["frame_count"] for s in sessions)
    print(f"Sessions: {len(sessions)}  total frames: {total_frames}")
    for det in cfg["lodo"]["detector_dirs"]:
        det_sessions = [s for s in sessions if s["detector"] == det]
        print(f"  {det}: {len(det_sessions)} sessions")

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

    # Save split artifacts alongside checkpoints for reproducibility
    artifacts_dir = Path("checkpoints") / "lodo_splits"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    fold_results: dict[str, dict] = {}

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

        result = _train_fold(fold, split_artifact, session_map, cfg, device)
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
    parser = argparse.ArgumentParser(description="LODO training for SFX hitfinder")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=None,
        help="Fold IDs to run (1–4). Omit to run all four.",
    )
    args = parser.parse_args()
    main(args.config, folds=args.folds)
