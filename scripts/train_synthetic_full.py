"""Full supervised training run on all 10k hitfinder_10k synthetic frames.

Dataset : hitfinder_10k_merged.h5 — 10,000 frames, 512×512, pre-assembled Eiger-like.
Labels  : labels[:, -1] (`bg_only`) — 1.0 → non-hit (0), 0.0 → hit (1).
Pipeline: GCN → LCN (window=9) → resize 224×224 (Reborn skipped; images pre-assembled).
Training: ResNet18 pretrained, AdamW, up to 200 epochs, early stop on val F1 (patience=20).
Logging : wandb project "sfx-hitfinder", tag "synthetic-full". No login needed (netrc saved).

Usage:
    conda run -n sfx-hitfinder python3 scripts/train_synthetic_full.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import wandb
from skimage.transform import resize as sk_resize
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.supervised import build_supervised_model
from src.preprocessing.normalize import LCN_WINDOW_DEFAULT, gcn, lcn
from src.training.train_supervised import evaluate, train_one_epoch

DATA_FILE = Path(
    "/data/bioxfel/user/gihan/Resonet/hitfinder_10k/hitfinder_10k_merged.h5"
)
BG_ONLY_COL = -1  # labels[:, -1] is `bg_only`: 1.0 → non-hit, 0.0 → hit
TRAIN_FRAC = 0.8
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-2
MAX_EPOCHS = 200
PATIENCE = 20  # early-stop: epochs without val F1 improvement
MIN_DELTA = 1e-4  # minimum improvement to reset patience counter
SEED = 42
TARGET_SIZE = (224, 224)
WANDB_PROJECT = "sfx-hitfinder"
RUN_NAME = f"resnet18-10k-full-seed{SEED}"
CKPT_DIR = Path("checkpoints") / RUN_NAME


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _preprocess(frame: np.ndarray) -> np.ndarray:
    image_gcn = gcn(frame)
    image_lcn = lcn(image_gcn, window=LCN_WINDOW_DEFAULT)
    resized = sk_resize(image_lcn, TARGET_SIZE, anti_aliasing=True, preserve_range=True)
    return resized.astype(np.float32)


class SyntheticHitDataset(Dataset):
    """Reads frames and binary labels from a hitfinder_10k HDF5 file."""

    def __init__(self, h5_path: Path, indices: list[int]) -> None:
        self._h5_path = h5_path
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        real_idx = self._indices[idx]
        with h5py.File(self._h5_path, "r") as f:
            frame = f["images"][real_idx].astype(np.float32)
            bg_only = float(f["labels"][real_idx, BG_ONLY_COL])
        label = 0 if bg_only == 1.0 else 1
        tensor = torch.from_numpy(_preprocess(frame)).unsqueeze(0)  # (1, 224, 224)
        return tensor, label


def _split_indices(n: int, train_frac: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n).tolist()
    cut = int(n * train_frac)
    return idx[:cut], idx[cut:]


def main() -> None:
    _set_seeds(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    with h5py.File(DATA_FILE, "r") as f:
        n_frames = f["images"].shape[0]
    print(f"Total frames in {DATA_FILE.name}: {n_frames}")

    train_idx, val_idx = _split_indices(n_frames, TRAIN_FRAC, SEED)
    print(f"Split — train: {len(train_idx)}  val: {len(val_idx)}")

    with h5py.File(DATA_FILE, "r") as f:
        all_bg = f["labels"][:, BG_ONLY_COL]
    n_hits = int((all_bg != 1.0).sum())
    n_nonhits = n_frames - n_hits
    print(f"Label balance — hits: {n_hits}  non-hits: {n_nonhits}\n")

    wandb.init(
        project=WANDB_PROJECT,
        name=RUN_NAME,
        tags=["supervised", "synthetic-full"],
        config={
            "backbone": "resnet18",
            "pretrained": True,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "min_delta": MIN_DELTA,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
            "train_frac": TRAIN_FRAC,
            "data_file": str(DATA_FILE),
            "n_frames": n_frames,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_hits": n_hits,
            "n_nonhits": n_nonhits,
            "lcn_window": LCN_WINDOW_DEFAULT,
        },
    )

    train_dl = DataLoader(
        SyntheticHitDataset(DATA_FILE, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )
    val_dl = DataLoader(
        SyntheticHitDataset(DATA_FILE, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )

    model = build_supervised_model(
        backbone="resnet18",
        pretrained=True,
        num_classes=2,
    ).to(device)
    print(f"Model: ResNet18 (pretrained)  num_classes=2")
    print(f"Max epochs: {MAX_EPOCHS}  Early stop patience: {PATIENCE}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / "best.pt"

    header = f"{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>8}  {'AP':>6}  {'AUC':>6}  {'F1':>6}  {'No-imp':>6}"
    print(header)
    print("-" * len(header))

    best_f1, epochs_no_improve, stopped_epoch = -1.0, 0, MAX_EPOCHS

    for epoch in range(1, MAX_EPOCHS + 1):
        train_m = train_one_epoch(model, train_dl, optimizer, criterion, device)
        val_m = evaluate(model, val_dl, criterion, device)

        print(
            f"{epoch:>5}  {train_m['loss']:>10.4f}  {val_m['loss']:>8.4f}"
            f"  {val_m['ap']:>6.4f}  {val_m['auc']:>6.4f}  {val_m['f1']:>6.4f}  {epochs_no_improve:>6}"
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

        if val_m["f1"] > best_f1 + MIN_DELTA:
            best_f1 = val_m["f1"]
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_f1": best_f1,
                },
                ckpt_path,
            )
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            stopped_epoch = epoch
            print(
                f"\nEarly stop at epoch {epoch} — no improvement for {PATIENCE} epochs."
            )
            break

    wandb.run.summary["best_val_f1"] = best_f1
    wandb.run.summary["stopped_epoch"] = stopped_epoch
    wandb.finish()

    print(f"\nBest val F1: {best_f1:.4f}  |  Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
