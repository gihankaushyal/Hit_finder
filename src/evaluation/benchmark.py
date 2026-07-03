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
    "run_patch_agg",
    "run_fold",
    "run_benchmark",
    "format_results_table",
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
    if variant == "semi_supervised_lodo":
        raise NotImplementedError(
            "semi_supervised_lodo is not yet implemented. "
            "Phase 5 will add unlabeled held-out frames to the MAE pretraining pool."
        )

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

    Model must output logits with shape [batch, 1], [batch], or [batch, 2].
    For [batch, 2], softmax is applied and the positive-class (index 1) column
    is used as the score. For [batch, 1] or [batch], sigmoid is applied.
    """
    model.to(device)
    model.eval()
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            if logits.ndim == 2 and logits.shape[1] == 2:
                scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            else:
                scores = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            all_scores.append(scores)
            all_labels.append(labels.numpy())

    if not all_scores:
        return {
            "ap": float("nan"),
            "auc_roc": float("nan"),
            "f1": float("nan"),
            "threshold": float("nan"),
        }

    y_score = np.concatenate(all_scores)
    y_true = np.concatenate(all_labels)

    best_f1, threshold = f1_at_optimal_threshold(y_true, y_score)
    return {
        "ap": average_precision(y_true, y_score),
        "auc_roc": auc_roc(y_true, y_score),
        "f1": best_f1,
        "threshold": threshold,
    }


def run_patch_agg(
    model: torch.nn.Module,
    session_map: dict[str, Path],
    session_ids: list[str],
    label_key: str = "entry_1/labels/hit",
    patch_size: int = 224,
    patch_stride: int = 224,
    min_hit_patches: int = 3,
    device: str = "cpu",
    inference_batch_size: int = 64,
    aggregation: str = "max",
) -> dict[str, float]:
    """Evaluate a model on full frames using patch-grid aggregation.

    For each frame: tile into complete patch_size×patch_size patches → GCN →
    LCN each patch → run all patches through the model in mini-batches →
    reduce to a single frame-level score (max softmax over patches or vote).

    Replaces run_on_loader for all evaluation steps (validation during training,
    in-domain test, and cross-detector test). The gradient-update training loop
    is unaffected.

    Args:
        model: Trained classifier with 2-class head.
        session_map: Mapping from session_id to CXI file path.
        session_ids: Subset of sessions to evaluate.
        label_key: HDF5 key for per-frame labels.
        patch_size: Patch side length in pixels (default 224).
        patch_stride: Step between patches in pixels.
        min_hit_patches: Vote mode only — frame is logged as binary hit if
            vote_count >= min_hit_patches. Not used in AP/AUC/F1 computation.
        device: Torch device string ('cpu' or 'cuda').
        inference_batch_size: Patches per forward pass.
        aggregation: Frame-score reduction over patches. "max" (default,
            backward-compatible): max softmax across patches. "vote":
            hit_count/n_patches where hit_count = patches with softmax[:,1] > 0.5.

    Returns:
        dict with keys: ap, auc_roc, f1, threshold.
    """
    import h5py

    from src.preprocessing.pipeline import preprocess_eval_patches

    model.to(device)
    model.eval()

    all_scores: list[float] = []
    all_labels: list[int] = []

    for sid in session_ids:
        path = Path(session_map[sid])
        with h5py.File(path, "r") as f:
            labels_arr = f[label_key][()]
            frames_arr = f["entry_1/data_1/data"][()]

        for frame_idx in range(len(frames_arr)):
            frame = frames_arr[frame_idx].astype(np.float32)
            try:
                patches_np = preprocess_eval_patches(
                    frame, patch_size=patch_size, stride=patch_stride
                )
            except ValueError:
                continue

            patch_tensors = torch.from_numpy(patches_np).unsqueeze(1).to(device)
            patch_scores_list: list[np.ndarray] = []
            with torch.no_grad():
                for i in range(0, len(patch_tensors), inference_batch_size):
                    batch = patch_tensors[i : i + inference_batch_size]
                    logits = model(batch)
                    if logits.ndim == 2 and logits.shape[1] == 2:
                        s = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    else:
                        s = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                    patch_scores_list.append(s)

            patch_scores = np.concatenate(patch_scores_list)
            n_patches = len(patch_scores)
            if aggregation == "vote":
                hit_count = int((patch_scores > 0.5).sum())
                frame_score = float(hit_count) / max(n_patches, 1)
            else:
                frame_score = float(patch_scores.max())
            all_scores.append(frame_score)
            all_labels.append(int(round(float(labels_arr[frame_idx]))))

    if not all_scores:
        nan = float("nan")
        return {"ap": nan, "auc_roc": nan, "f1": nan, "threshold": nan}

    y_score = np.array(all_scores)
    y_true = np.array(all_labels)
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
    session_map: dict[str, Path],
    device: str = "cpu",
    patch_stride: int = 224,
    min_hit_patches: int = 3,
    label_key: str = "entry_1/labels/hit",
) -> dict[str, float]:
    """Evaluate model on the held-out cross-detector sessions for one fold.

    Uses patch-grid aggregation: tiles each frame into 224×224 patches, runs
    all patches through the model, and reduces to a frame-level max score.
    """
    held_out_ids = [
        sid
        for sid, split in split_artifact["splits"].items()
        if split == SPLIT_CROSS_DETECTOR
    ]
    metrics = run_patch_agg(
        model,
        session_map,
        held_out_ids,
        label_key=label_key,
        patch_stride=patch_stride,
        min_hit_patches=min_hit_patches,
        device=device,
    )
    metrics["test_detector"] = split_artifact["test_detector"]
    return metrics


def run_benchmark(
    model: torch.nn.Module,
    split_artifacts: list[dict],
    session_map: dict[str, Path],
    device: str = "cpu",
    patch_stride: int = 224,
    min_hit_patches: int = 3,
    label_key: str = "entry_1/labels/hit",
) -> dict:
    """Run all folds and return per-fold results plus mean_ap and std_ap."""
    results: dict = {}
    for artifact in split_artifacts:
        fold_id = artifact.get("fold")
        if fold_id is None:
            raise ValueError(
                f"split_artifact for test_detector='{artifact.get('test_detector')}' "
                "has fold=None. Pass fold=<int> to build_session_stratified_split."
            )
        results[f"fold_{fold_id}"] = run_fold(
            model,
            artifact,
            session_map,
            device,
            patch_stride=patch_stride,
            min_hit_patches=min_hit_patches,
            label_key=label_key,
        )

    ap_values = [v["ap"] for v in results.values()]
    results["mean_ap"] = float(np.mean(ap_values))
    results["std_ap"] = float(np.std(ap_values, ddof=1))
    return results


def format_results_table(
    results: dict,
    oracle_ap: dict[str, float] | None = None,
) -> str:
    """Format a results table showing per-fold AP and optional oracle comparison.

    oracle_ap: {detector_name: oracle_AP} — within-detector upper bound.
    """
    has_oracle = oracle_ap is not None
    header = f"{'Fold':<8} {'Detector':<15} {'AP':>8}"
    if has_oracle:
        header += f"  {'Oracle AP':>10}  {'Rel. Gap':>9}"

    sep = "-" * len(header)
    lines = [header, sep]

    for fold_id in range(1, len(DETECTORS) + 1):
        fold_key = f"fold_{fold_id}"
        fold_data = results.get(fold_key, {})
        ap = fold_data.get("ap", float("nan"))
        detector = fold_data.get("test_detector", DETECTORS[fold_id - 1])
        line = f"{fold_key:<8} {detector:<15} {ap:>8.4f}"
        if has_oracle:
            o_ap = oracle_ap.get(detector, float("nan"))
            if o_ap is not None and o_ap > 0:
                gap = (o_ap - ap) / o_ap * 100
                line += f"  {o_ap:>10.4f}  {gap:>8.1f}%"
            else:
                line += f"  {'N/A':>10}  {'N/A':>9}"
        lines.append(line)

    lines.append(sep)
    mean_ap = results.get("mean_ap", float("nan"))
    std_ap = results.get("std_ap", float("nan"))
    mean_line = f"{'Mean':<8} {'':<15} {mean_ap:>8.4f}"
    if has_oracle:
        mean_line += f"  {'':>10}  {'+/-' + f' {std_ap:.4f}':>9}"
    else:
        mean_line += f"  +/- {std_ap:.4f}"
    lines.append(mean_line)
    return "\n".join(lines)
