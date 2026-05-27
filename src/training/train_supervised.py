"""Supervised training entry point.

Usage: python src/training/train_supervised.py --config configs/supervised/resnet18.yaml
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.evaluation.metrics import average_precision, auc_roc, f1_at_optimal_threshold
from src.models.supervised import build_supervised_model


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str | torch.device,
) -> dict[str, float]:
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.float().to(device), y.long().to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return {"loss": total_loss / max(n, 1)}


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str | torch.device,
) -> dict[str, float]:
    model.train(False)
    criterion = nn.CrossEntropyLoss()
    total_loss, n = 0.0, 0
    all_scores: list[float] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.float().to(device), y.long().to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * len(y)
            n += len(y)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_scores.extend(probs.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_score = np.array(all_scores)
    return {
        "loss": total_loss / max(n, 1),
        "ap": average_precision(y_true, y_score),
        "auc": auc_roc(y_true, y_score),
        "f1": f1_at_optimal_threshold(y_true, y_score)[0],
    }


def main(config_path: str | Path) -> None:
    import wandb  # lazy import — not needed for smoke tests or HPC imports without network

    from src.data.dataloader import supervised_loader
    from src.utils.config import load_config

    cfg = load_config(config_path)
    _set_seeds(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = cfg["model"]["backbone"]
    run_name = f"{backbone}-seed{cfg['seed']}"

    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"].get("entity"),
        name=run_name,
        config=cfg,
        tags=cfg["wandb"].get("tags", []),
    )

    train_dl = supervised_loader(
        split_file=cfg["data"]["train_split"],
        labels_file=cfg["data"]["labels_file"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        shuffle=True,
    )
    val_dl = supervised_loader(
        split_file=cfg["data"]["val_split"],
        labels_file=cfg["data"]["labels_file"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        shuffle=False,
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

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_m = train_one_epoch(model, train_dl, optimizer, criterion, device)
        val_m = evaluate(model, val_dl, device)
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
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_f1": best_f1,
                },
                ckpt_dir / "best.pt",
            )

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to model YAML config")
    args = parser.parse_args()
    main(args.config)
