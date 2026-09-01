"""Shared LODO fold-construction and train/eval loop (Track 1 + Track 2).

`_train_fold` accepts an optional `model_builder` callable so both the ResNet
asymmetric pipeline (Track 1) and the ViT fine-tune pipeline (Track 2) can
reuse the same loop without duplication.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

from src.data.dataloader import asymmetric_loader
from src.evaluation.benchmark import (
    SPLIT_CROSS_DETECTOR,
    SPLIT_IN_DOMAIN_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    run_patch_agg,
    save_split_artifact,
)
from src.hitfinders.base import Hitfinder
from src.models.supervised import build_supervised_model
from src.training.train_supervised import _set_seeds, train_one_epoch


def _build_intra_split(sessions: list[dict]) -> dict:
    """80/10/10 greedy split within a single detector — no cross-detector held-out set.

    Mirrors the greedy algorithm in build_session_stratified_split() but assigns
    every session to train/val/in_domain_test instead of segregating a test detector.
    """
    sorted_sessions = sorted(sessions, key=lambda s: s["frame_count"], reverse=True)
    bucket_names = [SPLIT_TRAIN, SPLIT_VAL, SPLIT_IN_DOMAIN_TEST]
    ratios = [0.80, 0.10, 0.10]
    bucket_targets = [r * len(sorted_sessions) for r in ratios]
    bucket_counts = [0, 0, 0]
    splits: dict[str, str] = {}
    for s in sorted_sessions:
        deficits = [bucket_targets[i] - bucket_counts[i] for i in range(3)]
        chosen = int(np.argmax(deficits))
        splits[s["session_id"]] = bucket_names[chosen]
        bucket_counts[chosen] += 1
    detector = sorted_sessions[0]["detector"] if sorted_sessions else "unknown"
    return {"fold": 0, "variant": "intra", "test_detector": detector, "splits": splits}


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


def _train_fold(
    fold: dict,
    split_artifact: dict,
    session_map: dict[str, Path],
    cfg: dict,
    hitfinder: Hitfinder,
    device: str,
    num_workers_override: int | None = None,
    resume_training: bool = False,
    model_builder: Callable[[], nn.Module] | None = None,
    run_name_prefix: str | None = None,
    extra_results: dict | None = None,
) -> dict:
    """Train one LODO fold and return metrics.

    `model_builder`: when provided, replaces the default ResNet construction.
      The callable takes no arguments and returns an uninitialised `nn.Module`.
      Backbone/num_classes checkpoint validation is skipped for custom builders.
    `run_name_prefix`: overrides the default `{backbone}-asymmetric` prefix in
      the wandb run name and checkpoint directory.
    `extra_results`: extra keys merged into the per-fold results.json.
    """
    import wandb

    backbone = cfg["model"].get("backbone", "vit_small_mae")
    seed = cfg["seed"]
    fold_id = fold["fold_id"]
    batch_size = cfg["training"]["batch_size"]
    num_workers = (
        num_workers_override
        if num_workers_override is not None
        else cfg["training"]["num_workers"]
    )
    epochs = cfg["training"]["epochs"]
    patience = cfg["training"].get("early_stopping_patience", 10)

    prefix = run_name_prefix or f"{backbone}-asymmetric"
    run_name = f"{prefix}-fold{fold_id}-seed{seed}"

    label_key = cfg["lodo"].get("label_key", "entry_1/labels/hit")

    train_ids = [sid for sid, s in split_artifact["splits"].items() if s == SPLIT_TRAIN]

    train_dl = asymmetric_loader(
        session_map=session_map,
        session_ids=train_ids,
        hitfinder=hitfinder,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        label_key=label_key,
    )

    bench_cfg = cfg.get("benchmark", {})
    patch_stride = bench_cfg.get("patch_stride", 224)
    min_hit_patches = bench_cfg.get("min_hit_patches", 3)
    aggregation = bench_cfg.get("aggregation", "vote")

    val_ids = [sid for sid, s in split_artifact["splits"].items() if s == SPLIT_VAL]
    in_domain_ids = [
        sid for sid, s in split_artifact["splits"].items() if s == SPLIT_IN_DOMAIN_TEST
    ]
    cross_ids = [
        sid for sid, s in split_artifact["splits"].items() if s == SPLIT_CROSS_DETECTOR
    ]

    n_train = len(train_dl.dataset)
    n_val = len(val_ids)
    n_indomain = len(in_domain_ids)
    n_cross = len(cross_ids)

    print(
        f"\n{'='*60}\n"
        f"Fold {fold_id}  |  held-out: {fold['test_detector']}\n"
        f"  train={n_train} patches  val={n_val} sessions  in_domain_test={n_indomain} sessions  cross={n_cross} sessions\n"
        f"{'='*60}"
    )

    _set_seeds(seed)
    if model_builder is not None:
        model = model_builder().to(device)
    else:
        model = build_supervised_model(
            backbone=backbone,
            pretrained=cfg["model"]["pretrained"],
            num_classes=cfg["model"]["num_classes"],
        ).to(device)

    ckpt_dir = Path("checkpoints") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / "best.pt"
    resume_eval_only = ckpt_path.exists() and not resume_training
    resume_training_from_ckpt = ckpt_path.exists() and resume_training

    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"].get("entity"),
        id=run_name,
        name=run_name,
        config={**cfg, "fold_id": fold_id, "test_detector": fold["test_detector"]},
        tags=cfg["wandb"].get("tags", []),
        resume="allow",
    )

    wandb.log({"hitfinder/backend": cfg["hitfinder"]["backend"]})

    if resume_eval_only:
        print(
            f"  Checkpoint found at {ckpt_path} — skipping training, resuming from evaluation."
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"],
        )
        criterion = nn.CrossEntropyLoss()

        best_f1 = -1.0
        epochs_no_improve = 0
        start_epoch = 1

        if resume_training_from_ckpt:
            # weights_only=False needed to restore optimizer_state_dict (our own file).
            _ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(_ckpt["model_state_dict"])
            if "optimizer_state_dict" in _ckpt:
                optimizer.load_state_dict(_ckpt["optimizer_state_dict"])
            else:
                print(
                    "  Warning: checkpoint has no optimizer state — starting with fresh optimizer."
                )
            _saved_f1 = _ckpt.get("val_f1", -1.0)
            best_f1 = -1.0 if np.isnan(_saved_f1) else _saved_f1
            epochs_no_improve = _ckpt.get("epochs_no_improve", 0)
            start_epoch = _ckpt.get("epoch", 0) + 1
            print(
                f"  Resuming training from epoch {start_epoch} "
                f"(best val F1 so far: {best_f1:.4f})"
            )
            if start_epoch > epochs:
                print(
                    f"  Warning: checkpoint epoch {start_epoch - 1} >= config epochs {epochs}. "
                    "Nothing left to train — proceeding to evaluation."
                )

        for epoch in range(start_epoch, epochs + 1):
            train_m = train_one_epoch(model, train_dl, optimizer, criterion, device)
            val_m = run_patch_agg(
                model,
                session_map,
                val_ids,
                label_key=label_key,
                patch_stride=patch_stride,
                min_hit_patches=min_hit_patches,
                device=device,
                aggregation=aggregation,
            )

            print(
                f"  Epoch {epoch:3d}/{epochs}  "
                f"train_loss={train_m['loss']:.4f}  "
                f"val_AP={val_m['ap']:.4f}  val_F1={val_m['f1']:.4f}"
            )
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_m["loss"],
                    "val/ap": val_m["ap"],
                    "val/auc": val_m["auc_roc"],
                    "val/f1": val_m["f1"],
                    "hitfinder/n_peaks_mean": float("nan"),
                }
            )

            if not np.isnan(val_m["f1"]) and val_m["f1"] > best_f1:
                best_f1 = val_m["f1"]
                epochs_no_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_f1": best_f1,
                        "epochs_no_improve": 0,
                        "inference_threshold": val_m["threshold"],
                        "backbone": backbone,
                        "num_classes": cfg["model"]["num_classes"],
                    },
                    ckpt_path,
                )
                print(f"    → checkpoint saved (val F1={best_f1:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(
                        f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)"
                    )
                    break

        if not ckpt_path.exists():
            print(
                "  No val-F1 improvement recorded — saving final epoch as checkpoint."
            )
            torch.save(
                {
                    "epoch": epochs,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": float("nan"),
                    "epochs_no_improve": patience,
                    "inference_threshold": float("nan"),
                    "backbone": backbone,
                    "num_classes": cfg["model"]["num_classes"],
                },
                ckpt_path,
            )

    # Evaluate best checkpoint on in-domain and cross-detector test sets.
    # weights_only=False: checkpoints contain optimizer_state_dict; all values are
    # tensors/primitives and we own the files.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Skip backbone/num_classes sanity check when caller provides a custom model_builder
    # (ViT fine-tune checkpoints use a different schema).
    if model_builder is None:
        ckpt_backbone = ckpt.get("backbone")
        ckpt_num_classes = ckpt.get("num_classes")
        if ckpt_backbone is not None and ckpt_backbone != backbone:
            raise RuntimeError(
                f"Checkpoint backbone={ckpt_backbone!r} does not match config backbone={backbone!r}. "
                "Delete the checkpoint or update the config."
            )
        if (
            ckpt_num_classes is not None
            and ckpt_num_classes != cfg["model"]["num_classes"]
        ):
            raise RuntimeError(
                f"Checkpoint num_classes={ckpt_num_classes} does not match config "
                f"num_classes={cfg['model']['num_classes']}. Delete the checkpoint or update the config."
            )

    model.load_state_dict(ckpt["model_state_dict"])

    _saved_thresh = ckpt.get("inference_threshold", float("nan"))
    inference_threshold: float = _saved_thresh if not np.isnan(_saved_thresh) else 0.5
    print(
        f"  Inference threshold: {inference_threshold:.4f} (option {'1 — val-set' if not np.isnan(_saved_thresh) else '2 — fixed 0.5'})"
    )

    in_domain_m = run_patch_agg(
        model,
        session_map,
        in_domain_ids,
        label_key=label_key,
        patch_stride=patch_stride,
        min_hit_patches=min_hit_patches,
        device=device,
        aggregation=aggregation,
    )
    cross_m = run_patch_agg(
        model,
        session_map,
        cross_ids,
        label_key=label_key,
        patch_stride=patch_stride,
        min_hit_patches=min_hit_patches,
        device=device,
        aggregation=aggregation,
    )

    print(
        f"  In-domain test:    AP={in_domain_m['ap']:.4f}  AUC={in_domain_m['auc_roc']:.4f}  F1={in_domain_m['f1']:.4f}"
    )
    print(
        f"  Cross-detector:    AP={cross_m['ap']:.4f}  AUC={cross_m['auc_roc']:.4f}  F1={cross_m['f1']:.4f}"
    )

    wandb.log(
        {
            "in_domain/ap": in_domain_m["ap"],
            "in_domain/auc": in_domain_m["auc_roc"],
            "in_domain/f1": in_domain_m["f1"],
            "cross/ap": cross_m["ap"],
            "cross/auc": cross_m["auc_roc"],
            "cross/f1": cross_m["f1"],
            "inference_threshold": inference_threshold,
        }
    )
    wandb.finish()

    result: dict = {
        "fold_id": fold_id,
        "test_detector": fold["test_detector"],
        "inference_threshold": inference_threshold,
        "cross": {
            "ap": cross_m["ap"],
            "auc_roc": cross_m["auc_roc"],
            "f1": cross_m["f1"],
            "threshold": cross_m["threshold"],
        },
        "in_domain": {
            "ap": in_domain_m["ap"],
            "auc_roc": in_domain_m["auc_roc"],
            "f1": in_domain_m["f1"],
            "threshold": in_domain_m["threshold"],
        },
    }
    result.update(extra_results or {})
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
