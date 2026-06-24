"""Evaluate the trained supervised model on Resonet-generated CXI files.

Handles the Resonet CXI format:
  - Data key:  entry_1/data_1/data  (N, 5632, 384) uint16
  - Label key: entry_1/labels/hit   (N,) float32  — 1.0=hit, 0.0=non-hit

Preprocessing uses EigerRESoNeT geometry (extract_panels_from_canvas) or the
preprocess_assembled bypass (identical output for this detector layout).

Usage:
    /home/gketawal/.conda/envs/sfx-hitfinder/bin/python \\
        scripts/evaluate_resonet_cxi.py \\
        --cxi-dir /data/bioxfel/user/gihan/Resonet/cxi_100 \\
        --checkpoint checkpoints/resnet18-10k-full-seed42/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import MultiFrameCXIDataset
from src.evaluation.metrics import average_precision, auc_roc, f1_at_optimal_threshold
from src.models.supervised import build_supervised_model
from src.preprocessing.geometry import extract_panels_from_canvas, load_pad_geometry
from src.preprocessing.pipeline import preprocess

BATCH_SIZE = 64
_PADS = None  # loaded once on first use


def _preprocess_with_geometry(frame: np.ndarray) -> np.ndarray:
    global _PADS
    if _PADS is None:
        _PADS = load_pad_geometry("EigerRESoNeT")
    panels = extract_panels_from_canvas(frame, _PADS)
    return preprocess(panels, _PADS)


def run_normalization_preview(cxi_paths: list[Path], n_frames: int = 5) -> None:
    """Print normalization statistics for the first n_frames frames."""
    from src.preprocessing.io import read_frame
    from src.preprocessing.pipeline import preprocess_assembled

    print(f"\n{'='*50}")
    print("  Normalization pipeline preview")
    print(f"  File: {cxi_paths[0]}")
    print(f"{'='*50}")
    print(f"  {'Frame':<6} {'Raw shape':<14} {'Raw min':<10} {'Raw max':<10} "
          f"{'Norm min':<10} {'Norm max':<10} {'Finite'}")
    print(f"  {'-'*72}")

    for i in range(min(n_frames, 50)):
        raw = read_frame(cxi_paths[0], i)
        norm = preprocess_assembled(raw)
        print(f"  {i:<6} {str(raw.shape):<14} {raw.min():<10.1f} {raw.max():<10.1f} "
              f"{norm.min():<10.4f} {norm.max():<10.4f} {np.isfinite(norm).all()}")

    print(f"  → Output shape: (224, 224) float32\n")


def run_model_evaluation(
    cxi_paths: list[Path],
    checkpoint: Path,
    use_geometry: bool = False,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{'='*50}")
    print(f"  Supervised model evaluation")
    print(f"  Device: {device}")
    print(f"  Files:  {len(cxi_paths)} CXI file(s)")
    print(f"  Geometry assembly: {'EigerRESoNeT' if use_geometry else 'preprocess_assembled bypass'}")
    print(f"{'='*50}")

    preprocess_fn = _preprocess_with_geometry if use_geometry else None
    dataset = MultiFrameCXIDataset(cxi_paths, preprocess_fn=preprocess_fn)
    n_total = len(dataset)
    print(f"  Total frames: {n_total}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    model = build_supervised_model(backbone="resnet18", pretrained=False, num_classes=2)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Saved epoch={ckpt['epoch']}, val F1={ckpt['val_f1']:.4f}\n")

    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.float().to(device)
            logits = model(images)
            scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_scores.append(scores)
            all_labels.append(labels.numpy() if hasattr(labels, "numpy") else np.array(labels))
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) * BATCH_SIZE >= n_total:
                done = min((batch_idx + 1) * BATCH_SIZE, n_total)
                print(f"  Processed {done}/{n_total} frames...")

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

    print(f"\n{'='*50}")
    print(f"  Results — {n_total} frames")
    print(f"  Hits: {n_hits}  Non-hits: {n_nonhits}  ({n_hits/n_total:.1%} hit rate)")
    print(f"{'='*50}")
    print(f"  AP  (avg precision):              {ap:.4f}")
    print(f"  AUC-ROC:                          {auc:.4f}")
    print(f"  F1  (optimal thresh={threshold:.3f}): {f1:.4f}")
    print(f"{'='*50}")
    print(f"  Confusion matrix (threshold={threshold:.3f}):")
    print(f"    TP={tp:<5} FP={fp}")
    print(f"    FN={fn:<5} TN={tn}")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    print(f"  Precision: {precision:.4f}  Recall: {recall:.4f}")
    print(f"{'='*50}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Resonet CXI frames through normalization + supervised model"
    )
    parser.add_argument(
        "--cxi-dir",
        type=Path,
        default=Path("/data/bioxfel/user/gihan/Resonet/cxi_100"),
        help="Directory containing compressed*.cxi files",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/resnet18-10k-full-seed42/best.pt"),
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--use-geometry",
        action="store_true",
        default=False,
        help="Use EigerRESoNeT geometry assembly (default: preprocess_assembled bypass)",
    )
    parser.add_argument(
        "--preview-frames",
        type=int,
        default=5,
        help="Number of frames to show in normalization preview",
    )
    args = parser.parse_args()

    cxi_paths = sorted(args.cxi_dir.glob("compressed*.cxi"))
    if not cxi_paths:
        raise FileNotFoundError(f"No compressed*.cxi files found in {args.cxi_dir}")
    print(f"\nFound {len(cxi_paths)} CXI file(s) in {args.cxi_dir}")

    run_normalization_preview(cxi_paths, n_frames=args.preview_frames)
    run_model_evaluation(cxi_paths, args.checkpoint, use_geometry=args.use_geometry)


if __name__ == "__main__":
    main()
