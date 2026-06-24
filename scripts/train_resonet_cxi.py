"""Train ResNet18 on Resonet-generated CXI data with embedded labels.

Loads a single multi-frame CXI file, splits into train/val via random_split,
and trains using the same loop, optimizer, and checkpoint format as
train_supervised.py.

Usage:
    /home/gketawal/.conda/envs/sfx-hitfinder/bin/python \\
        scripts/train_resonet_cxi.py \\
        --config configs/supervised/resnet18_resonet.yaml

SLURM:
    sbatch scripts/submit_resonet_train.sh
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import MultiFrameCXIDataset
from src.models.supervised import build_supervised_model
from src.training.train_supervised import _set_seeds, evaluate, train_one_epoch
from src.utils.config import load_config


def main(config_path: str | Path) -> None:
    import wandb

    cfg = load_config(config_path)
    _set_seeds(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone = cfg["model"]["backbone"]
    run_name = f"{backbone}-resonet-seed{cfg['seed']}"

    cxi_path = Path(cfg["data"]["cxi_file"])
    val_fraction = float(cfg["data"].get("val_fraction", 0.2))
    label_key = cfg["data"].get("label_key", "entry_1/labels/hit")

    full_dataset = MultiFrameCXIDataset([cxi_path], label_key=label_key)
    n_total = len(full_dataset)
    test_fraction = float(cfg["data"].get("test_fraction", 0.1))
    n_test = int(n_total * test_fraction)
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val - n_test

    generator = torch.Generator().manual_seed(cfg["seed"])
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    print(f"Run:      {run_name}")
    print(f"Device:   {device}")
    print(f"Data:     {cxi_path.name}  —  {n_train} train / {n_val} val / {n_test} test frames")
    print(f"Backbone: {backbone}  pretrained={cfg['model']['pretrained']}")
    print(f"Epochs:   {cfg['training']['epochs']}  lr={cfg['training']['learning_rate']}")

    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"].get("entity"),
        name=run_name,
        config=cfg,
        tags=cfg["wandb"].get("tags", []),
    )

    model = build_supervised_model(
        backbone=backbone,
        pretrained=cfg["model"]["pretrained"],
        num_classes=cfg["model"]["num_classes"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = Path("checkpoints") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    patience = cfg["training"].get("early_stopping_patience", 10)
    epochs_without_improvement = 0

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_m = train_one_epoch(model, train_dl, optimizer, criterion, device)
        val_m = evaluate(model, val_dl, criterion, device)

        print(
            f"Epoch {epoch:3d}/{cfg['training']['epochs']}  "
            f"train_loss={train_m['loss']:.4f}  "
            f"val_loss={val_m['loss']:.4f}  "
            f"val_AP={val_m['ap']:.4f}  val_AUC={val_m['auc']:.4f}  val_F1={val_m['f1']:.4f}"
        )
        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_m["loss"],
                "val/loss": val_m["loss"],
                "val/ap": val_m["ap"],
                "val/auc": val_m["auc"],
                "val/f1": val_m["f1"],
            }
        )

        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_f1": best_f1,
                },
                ckpt_dir / "best.pt",
            )
            print(f"  → checkpoint saved (val F1={best_f1:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Final evaluation on held-out test set using best checkpoint
    print("\nEvaluating best checkpoint on test set...")
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    criterion_eval = nn.CrossEntropyLoss()
    test_m = evaluate(model, test_dl, criterion_eval, device)

    print(
        f"Test results (epoch {ckpt['epoch']})  "
        f"AP={test_m['ap']:.4f}  AUC={test_m['auc']:.4f}  F1={test_m['f1']:.4f}"
    )
    wandb.log({
        "test/ap": test_m["ap"],
        "test/auc": test_m["auc"],
        "test/f1": test_m["f1"],
        "test/loss": test_m["loss"],
    })

    wandb.finish()
    print(f"\nDone. Best val F1={best_f1:.4f}  checkpoint: {ckpt_dir}/best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
