"""Evaluate a trained supervised model on held-out HDF5 test data.

The test files contain pre-assembled images (no Reborn geometry step needed).
Label encoding matches training: labels[:, -1] bg_only==1.0 → non-hit (0), else → hit (1).
Preprocessing matches training: GCN → LCN (window=9) → resize 224×224.

Usage:
    conda run -n sfx-hitfinder python3 scripts/evaluate_supervised.py \\
        --data-dir /data/bioxfel/user/gihan/Resonet/hitfinder_val \\
        --checkpoint checkpoints/resnet18-10k-full-seed42/best.pt
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from skimage.transform import resize as sk_resize
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.supervised import build_supervised_model
from src.preprocessing.normalize import LCN_WINDOW_DEFAULT, gcn, lcn
from src.evaluation.metrics import average_precision, auc_roc, f1_at_optimal_threshold

TARGET_SIZE = (224, 224)
BATCH_SIZE = 64
BG_ONLY_COL = -1


def _preprocess(frame: np.ndarray) -> np.ndarray:
    image_gcn = gcn(frame.astype(np.float32))
    image_lcn = lcn(image_gcn, window=LCN_WINDOW_DEFAULT)
    resized = sk_resize(image_lcn, TARGET_SIZE, anti_aliasing=True, preserve_range=True)
    return resized.astype(np.float32)


class ValDataset(Dataset):
    """Loads frames from multiple HDF5 files; same label encoding as training."""

    def __init__(self, h5_files: list[Path]) -> None:
        self._entries: list[tuple[Path, int]] = []
        for path in h5_files:
            with h5py.File(path, "r") as f:
                n = f["images"].shape[0]
            for i in range(n):
                self._entries.append((path, i))

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, frame_idx = self._entries[idx]
        with h5py.File(path, "r") as f:
            frame = f["images"][frame_idx].astype(np.float32)
            bg_only = float(f["labels"][frame_idx, BG_ONLY_COL])
        label = 0 if bg_only == 1.0 else 1
        tensor = torch.from_numpy(_preprocess(frame)).unsqueeze(0)  # (1, 224, 224)
        return tensor, label


def run_evaluation(data_dir: Path, checkpoint: Path) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    h5_files = sorted(data_dir.glob("compressed*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No compressed*.h5 files found in {data_dir}")
    print(f"Test files: {len(h5_files)}  ({data_dir})")

    dataset = ValDataset(h5_files)
    n_total = len(dataset)
    print(f"Total frames: {n_total}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )

    model = build_supervised_model(backbone="resnet18", pretrained=False, num_classes=2)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Checkpoint: {checkpoint}  (saved at epoch {ckpt['epoch']}, val F1={ckpt['val_f1']:.4f})\n")

    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.float().to(device)
            logits = model(images)
            scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_scores.append(scores)
            all_labels.append(labels.numpy())

    y_score = np.concatenate(all_scores)
    y_true = np.concatenate(all_labels)

    ap = average_precision(y_true, y_score)
    auc = auc_roc(y_true, y_score)
    f1, threshold = f1_at_optimal_threshold(y_true, y_score)

    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    n_hits = int(y_true.sum())
    n_nonhits = n_total - n_hits

    print(f"{'='*45}")
    print(f"  Test set results — {n_total} frames")
    print(f"  Hits: {n_hits}  Non-hits: {n_nonhits}  ({n_hits/n_total:.1%} hit rate)")
    print(f"{'='*45}")
    print(f"  AP (avg precision): {ap:.4f}")
    print(f"  AUC-ROC:            {auc:.4f}")
    print(f"  F1 (optimal thresh={threshold:.3f}): {f1:.4f}")
    print(f"{'='*45}")
    print(f"  Confusion matrix (threshold={threshold:.3f}):")
    print(f"    TP={tp}  FP={fp}")
    print(f"    FN={fn}  TN={tn}")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    print(f"  Precision: {precision:.4f}  Recall: {recall:.4f}")
    print(f"{'='*45}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate supervised hitfinder on test data")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/data/bioxfel/user/gihan/Resonet/hitfinder_val"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/resnet18-10k-full-seed42/best.pt"),
    )
    args = parser.parse_args()
    run_evaluation(args.data_dir, args.checkpoint)


if __name__ == "__main__":
    main()
