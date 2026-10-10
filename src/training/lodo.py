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
from src.data.frame_cache import FrameCache, verify_cache_or_raise
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
from src.training.inference_results import (
    INFERENCE_RESULTS_NAME,
    RESULTS_NAME,
    assess_training,
    fetch_wandb_history,
    inference_block,
    inference_result_path,
    summary_updates,
    wandb_enabled,
)
from src.training.train_supervised import _set_seeds, train_one_epoch
from src.training.wandb_identity import resolve_wandb_id, wandb_id_for_training


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


def _write_inference_summary(
    cfg: dict, run_name: str, updates: dict, wandb_id: str | None = None
) -> None:
    """Record an inference pass in the closed-out W&B run's summary, if W&B is on.

    Runs after the result file is written, so a W&B failure can never lose the
    metrics: it only warns. The run is touched only now that the metrics exist, so
    a failed evaluation cannot leave it marked crashed. No config or tags are sent,
    and the settings stop the resumed run from re-uploading metadata, console
    output, system stats or code.
    """
    import wandb

    if not wandb_enabled():
        return
    try:
        run = wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"].get("entity"),
            id=wandb_id or run_name,
            name=run_name,
            resume="allow",
            settings=wandb.Settings(
                console="off",
                x_disable_stats=True,
                x_disable_meta=True,
                save_code=False,
            ),
        )
        try:
            for key, value in updates.items():
                run.summary[key] = value
        finally:
            wandb.finish()
    except Exception as exc:  # the local result file is already written
        print(f"  [wandb] could not record the inference summary: {exc!r}")


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
    frame_cache: FrameCache | None = None,
    inference_only: bool = False,
) -> dict:
    """Train one LODO fold and return metrics.

    `model_builder`: when provided, replaces the default ResNet construction.
      The callable takes no arguments and returns an uninitialised `nn.Module`.
      Backbone/num_classes checkpoint validation is skipped for custom builders.
    `run_name_prefix`: overrides the default `{backbone}-asymmetric` prefix in
      the wandb run name and checkpoint directory.
    `extra_results`: extra keys merged into the per-fold results.json.
    `frame_cache`: optional two-tier FrameCache. When supplied, its manifest is
      verified against `cfg` before any training starts, and it is forwarded to
      the training loader and to every run_patch_agg call.
    `inference_only`: evaluate an existing best.pt on the in-domain and
      cross-detector sets without training. The training loader, optimizer and
      epoch loop are never built; W&B (when enabled) only receives run-summary
      entries under `inference/`; the result goes to results.json if the run
      has none, else to results.inference.json (see inference_results.py).
    """
    import wandb

    verify_cache_or_raise(frame_cache, cfg)

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
    run_suffix = cfg.get("wandb", {}).get("run_suffix", "")
    run_name = f"{prefix}-fold{fold_id}-seed{seed}{run_suffix}"

    label_key = cfg["lodo"].get("label_key", "entry_1/labels/hit")
    hit_frac = cfg.get("asymmetric", {}).get("hit_frac", 0.5)
    hard_neg_max_attempts = cfg.get("asymmetric", {}).get("hard_neg_max_attempts", 50)
    crops_per_frame = cfg.get("asymmetric", {}).get("crops_per_frame", 1)

    ckpt_dir = Path("checkpoints") / run_name
    # Must run AFTER check_checkpoint_collisions' real (non-dry) pass, which may
    # rotate wandb_id.txt; resolving earlier would log to the superseded run.
    wandb_id = resolve_wandb_id(ckpt_dir, run_name)
    ckpt_path = ckpt_dir / "best.pt"
    if inference_only:
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"--inference-only needs an existing checkpoint: {ckpt_path}"
            )
    else:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if ckpt_path.exists() and not resume_training:
            raise RuntimeError(
                f"{ckpt_path} already exists: pass resume_training or inference_only "
                "(the entry points gate this before calling _train_fold)."
            )
    resume_training_from_ckpt = ckpt_path.exists() and resume_training

    train_dl = None
    if not inference_only:
        train_ids = [
            sid for sid, s in split_artifact["splits"].items() if s == SPLIT_TRAIN
        ]
        train_dl = asymmetric_loader(
            session_map=session_map,
            session_ids=train_ids,
            hitfinder=hitfinder,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            label_key=label_key,
            frame_cache=frame_cache,
            hit_frac=hit_frac,
            hard_neg_max_attempts=hard_neg_max_attempts,
            crops_per_frame=crops_per_frame,
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

    if inference_only:
        train_desc = "train=skipped (inference only)"
    else:
        n_train_frames = len(train_dl.dataset)
        n_train = n_train_frames * crops_per_frame
        train_desc = (
            f"train={n_train} crops ({n_train_frames} frames x {crops_per_frame} "
            "crops/frame)"
        )
    n_val = len(val_ids)
    n_indomain = len(in_domain_ids)
    n_cross = len(cross_ids)

    print(
        f"\n{'='*60}\n"
        f"Fold {fold_id}  |  held-out: {fold['test_detector']}\n"
        f"  {train_desc}  "
        f"val={n_val} sessions  in_domain_test={n_indomain} sessions  cross={n_cross} sessions\n"
        f"{'='*60}"
    )

    _set_seeds(seed)
    if model_builder is not None:
        model = model_builder().to(device)
    else:
        model = build_supervised_model(
            backbone=backbone,
            # best.pt supplies every weight; skip the pretrained-weights download.
            pretrained=cfg["model"]["pretrained"] and not inference_only,
            num_classes=cfg["model"]["num_classes"],
        ).to(device)

    if not inference_only:
        wandb_id = wandb_id_for_training(
            ckpt_dir,
            run_name,
            cfg["wandb"]["project"],
            cfg["wandb"].get("entity"),
            resuming=resume_training_from_ckpt,
        )
        wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"].get("entity"),
            id=wandb_id,
            name=run_name,
            config={**cfg, "fold_id": fold_id, "test_detector": fold["test_detector"]},
            tags=cfg["wandb"].get("tags", []),
            resume="allow",
        )
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")

        wandb.log({"hitfinder/backend": cfg["hitfinder"]["backend"]})

    if inference_only:
        print(f"  --inference-only: evaluating {ckpt_path}; no training.")
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
            train_dl.dataset.set_epoch(epoch)
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
                frame_cache=frame_cache,
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
                    "train/realized_hit_frac": train_m["hit_frac"],
                    "val/ap": val_m["ap"],
                    "val/auc": val_m["auc_roc"],
                    "val/f1": val_m["f1"],
                    "hitfinder/n_peaks_mean": float("nan"),
                },
                step=epoch,
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

    training_check = None
    if inference_only:
        training_check = assess_training(
            configured_epochs=epochs,
            checkpoint_epoch=ckpt.get("epoch"),
            history=fetch_wandb_history(
                cfg["wandb"]["project"], cfg["wandb"].get("entity"), wandb_id
            ),
        )
        for warning in training_check["warnings"]:
            print(f"  [inference] WARNING: {warning}")

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
        frame_cache=frame_cache,
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
        frame_cache=frame_cache,
    )

    print(
        f"  In-domain test:    AP={in_domain_m['ap']:.4f}  AUC={in_domain_m['auc_roc']:.4f}  F1={in_domain_m['f1']:.4f}"
    )
    print(
        f"  Cross-detector:    AP={cross_m['ap']:.4f}  AUC={cross_m['auc_roc']:.4f}  F1={cross_m['f1']:.4f}"
    )

    if not inference_only:
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
    if inference_only:
        results_path = inference_result_path(ckpt_dir)
        # Always recorded, so a results.json completed by an inference pass is
        # distinguishable from one written by a finished training run.
        result["inference"] = inference_block(
            aggregation=aggregation,
            patch_stride=patch_stride,
            min_hit_patches=min_hit_patches,
            checkpoint=ckpt,
            training_check=training_check,
        )
    else:
        results_path = ckpt_dir / RESULTS_NAME
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved → {results_path}")

    if inference_only:
        _write_inference_summary(
            cfg,
            run_name,
            summary_updates(in_domain_m, cross_m, inference_threshold),
            wandb_id=wandb_id,
        )

    return {
        "test_detector": fold["test_detector"],
        "ap": cross_m["ap"],
        "in_domain_ap": in_domain_m["ap"],
        "auc_roc": cross_m["auc_roc"],
        "f1": cross_m["f1"],
    }
