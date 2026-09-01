"""MAE pretraining loop — Track 2, strict LODO (one run per fold).

Usage:
    python -m src.training.train_ssl_pretrain --config configs/ssl/mae_pretrain.yaml --fold 1
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import wandb

from src.data.dataloader import ssl_crop_loader
from src.hitfinders import get_hitfinder
from src.models.ssl import MASKING_PEAK_AWARE, build_mae_model
from src.training.lodo import build_sessions
from src.training.train_supervised import _set_seeds
from src.utils.config import load_config

CHECKPOINT_DIR_DEFAULT = "checkpoints"


def _cosine_lr(base_lr: float, epoch: int, warmup: int, total: int) -> float:
    if epoch < warmup:
        return base_lr * (epoch + 1) / max(warmup, 1)
    t = (epoch - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


def run_pretrain(
    cfg: dict,
    session_map: dict[str, Path],
    session_ids: list[str],
    run_name: str,
    device: str,
    resume: bool = False,
) -> dict:
    _set_seeds(cfg["seed"])
    ssl_cfg = cfg["ssl"]
    tr = cfg["training"]
    masking = ssl_cfg.get("masking", "random")
    hitfinder = get_hitfinder(cfg) if masking == MASKING_PEAK_AWARE else None

    dl = ssl_crop_loader(
        session_map=session_map,
        session_ids=session_ids,
        batch_size=tr["batch_size"],
        num_workers=tr.get("num_workers", 8),
        shuffle=True,
        seed=cfg["seed"],
        crops_per_frame=ssl_cfg.get("crops_per_frame", 1),
        hitfinder=hitfinder,
        min_valid_frac=ssl_cfg.get("min_valid_frac", 0.5),
    )
    model = build_mae_model(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=tr["learning_rate"], weight_decay=tr["weight_decay"]
    )
    ckpt_dir = Path(cfg.get("checkpoint_dir", CHECKPOINT_DIR_DEFAULT)) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_dir / "last.pt"

    start_epoch = 1
    if resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        opt.load_state_dict(state["optimizer_state_dict"])
        start_epoch = state["epoch"] + 1

    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"].get("entity"),
        id=run_name,
        name=run_name,
        config=cfg,
        tags=cfg["wandb"].get("tags", []),
        resume="allow",
    )

    epochs = tr["epochs"]
    final_loss = float("nan")
    epochs_run = 0
    for epoch in range(start_epoch, epochs + 1):
        lr = _cosine_lr(
            tr["learning_rate"], epoch - 1, tr.get("warmup_epochs", 0), epochs
        )
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        losses = []
        for crops, peak_patches in dl:
            crops = crops.to(device)
            loss, _, _ = model(
                crops,
                mask_ratio=ssl_cfg.get("mask_ratio", 0.6),
                masking=masking,
                peak_patches=peak_patches.to(device),
                peak_mask_frac=ssl_cfg.get("peak_mask_frac", 1.0),
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr.get("grad_clip", 1.0))
            opt.step()
            losses.append(loss.item())
        final_loss = float(np.mean(losses)) if losses else float("nan")
        epochs_run += 1
        wandb.log({"epoch": epoch, "pretrain/loss": final_loss, "pretrain/lr": lr})
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "loss": final_loss,
                "backbone": cfg.get("model", {}).get(
                    "backbone", "mae_vit_small_patch16"
                ),
                "ssl": ssl_cfg,
            },
            last_path,
        )
        if epoch % tr.get("checkpoint_every", 20) == 0:
            epoch_ckpt = ckpt_dir / f"epoch{epoch}.pt"
            torch.save(
                torch.load(last_path, map_location="cpu", weights_only=True), epoch_ckpt
            )
    wandb.finish()
    return {
        "epochs_run": epochs_run,
        "final_loss": final_loss,
        "checkpoint": str(last_path),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument(
        "--fold",
        type=int,
        required=True,
        help="LODO fold ID (1-4); the fold's test detector is EXCLUDED from pretraining",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sessions, session_map = build_sessions(cfg["lodo"])

    from src.evaluation.benchmark import build_lodo_folds

    fold = next(f for f in build_lodo_folds() if f["fold_id"] == args.fold)
    held_out = fold["test_detector"]
    pretrain_ids = [s["session_id"] for s in sessions if s["detector"] != held_out]
    run_name = f"mae-vits16-fold{args.fold}-seed{cfg['seed']}"
    print(f"Fold {args.fold}: excluding {held_out}; {len(pretrain_ids)} sessions")
    summary = run_pretrain(
        cfg, session_map, pretrain_ids, run_name, device, resume=args.resume
    )
    print(summary)


if __name__ == "__main__":
    main()
