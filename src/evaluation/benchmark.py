"""Cross-detector leave-one-detector-out evaluation protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import average_precision, auc_roc, f1_at_optimal_threshold

__all__ = [
    "DETECTORS",
    "SPLIT_TRAIN",
    "SPLIT_VAL",
    "SPLIT_IN_DOMAIN_TEST",
    "SPLIT_CROSS_DETECTOR",
    "build_lodo_folds",
    "build_session_stratified_split",
    "save_split_artifact",
    "load_split_artifact",
    "run_on_loader",
    "run_fold",
    "run_benchmark",
]

DETECTORS: list[str] = ["AGIPD", "JUNGFRAU_4M", "ePix10k", "Eiger4M"]

SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_IN_DOMAIN_TEST = "in_domain_test"
SPLIT_CROSS_DETECTOR = "cross_detector_eval"


def build_lodo_folds() -> list[dict]:
    """Return the 4 leave-one-detector-out fold definitions.

    Each fold: {"fold_id": int, "test_detector": str, "train_detectors": list[str]}
    """
    return [
        {
            "fold_id": i + 1,
            "test_detector": detector,
            "train_detectors": [d for d in DETECTORS if d != detector],
        }
        for i, detector in enumerate(DETECTORS)
    ]


def build_session_stratified_split(
    sessions: list[dict],
    test_detector: str,
    ratios: tuple[float, float, float] = (0.80, 0.10, 0.10),
    fold: int | None = None,
    variant: str | None = None,
    seed: int = 42,
) -> dict:
    """Build a session-level split artifact.

    Sessions from test_detector get SPLIT_CROSS_DETECTOR.
    Remaining sessions are assigned greedy train/val/in_domain_test
    (sorted by frame_count descending, bucket fills in ratio order).

    The algorithm is fully deterministic from the sorted order of sessions by
    frame_count. The ``seed`` parameter is accepted for API forward-compatibility
    but has no effect on the current implementation.
    """
    held_out = [s for s in sessions if s["detector"] == test_detector]
    train_pool = sorted(
        [s for s in sessions if s["detector"] != test_detector],
        key=lambda s: s["frame_count"],
        reverse=True,
    )

    bucket_names = [SPLIT_TRAIN, SPLIT_VAL, SPLIT_IN_DOMAIN_TEST]
    bucket_targets = [r * len(train_pool) for r in ratios]
    bucket_counts = [0, 0, 0]
    splits: dict[str, str] = {}

    for s in train_pool:
        deficits = [bucket_targets[i] - bucket_counts[i] for i in range(3)]
        chosen = int(np.argmax(deficits))
        splits[s["session_id"]] = bucket_names[chosen]
        bucket_counts[chosen] += 1

    for s in held_out:
        splits[s["session_id"]] = SPLIT_CROSS_DETECTOR

    return {
        "fold": fold,
        "variant": variant,
        "test_detector": test_detector,
        "splits": splits,
    }


def save_split_artifact(artifact: dict, path: str | Path) -> None:
    """Write split artifact to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2)


def load_split_artifact(path: str | Path) -> dict:
    """Load split artifact from a JSON file."""
    with open(path) as f:
        return json.load(f)


def run_on_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> dict[str, float]:
    """Run model on a DataLoader; return ap, auc_roc, f1, threshold.

    Model must output logits (pre-sigmoid) with shape [batch, 1] or [batch].
    """
    model.to(device)
    model.eval()
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images).squeeze(-1)
            scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(scores)
            all_labels.append(labels.numpy())

    y_score = np.concatenate(all_scores)
    y_true = np.concatenate(all_labels)

    best_f1, threshold = f1_at_optimal_threshold(y_true, y_score)
    return {
        "ap": average_precision(y_true, y_score),
        "auc_roc": auc_roc(y_true, y_score),
        "f1": best_f1,
        "threshold": threshold,
    }


def run_fold(
    model: torch.nn.Module,
    split_artifact: dict,
    dataloader_factory: Callable[[list[str]], DataLoader],
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate model on the held-out test-detector sessions for one fold.

    dataloader_factory(session_ids) must return a DataLoader over those sessions.
    """
    held_out_ids = [
        sid
        for sid, split in split_artifact["splits"].items()
        if split == SPLIT_CROSS_DETECTOR
    ]
    loader = dataloader_factory(held_out_ids)
    metrics = run_on_loader(model, loader, device)
    metrics["test_detector"] = split_artifact["test_detector"]
    return metrics


def run_benchmark(
    model: torch.nn.Module,
    split_artifacts: list[dict],
    dataloader_factory: Callable[[list[str]], DataLoader],
    device: str = "cpu",
) -> dict:
    """Run all folds and return per-fold results plus mean_ap and std_ap."""
    results: dict = {}
    for artifact in split_artifacts:
        fold_id = artifact["fold"]
        results[f"fold_{fold_id}"] = run_fold(
            model, artifact, dataloader_factory, device
        )

    ap_values = [v["ap"] for v in results.values()]
    results["mean_ap"] = float(np.mean(ap_values))
    results["std_ap"] = float(np.std(ap_values))
    return results
