"""Evaluation metrics for SFX hitfinder: AP, AUC-ROC, F1."""

from __future__ import annotations

import numpy as np

__all__ = ["average_precision", "auc_roc", "f1_at_optimal_threshold"]


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision-recall curve (interpolation-free, sklearn-compatible).

    Tie-breaking convention: ``np.argsort(y_score, kind='stable')[::-1]``, which
    matches sklearn's ``average_precision_score``.  For equal scores, higher-indexed
    samples appear first (pessimistic ordering — positives are NOT promoted to the
    front of a tie group).  This is the same stable-sort reversal sklearn uses
    internally in ``precision_recall_curve``.

    Returns 0.0 if there are no positive examples.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    sorted_idx = np.argsort(y_score, kind="stable")[::-1]
    y_sorted = y_true[sorted_idx]

    n_pos = y_sorted.sum()
    if n_pos == 0:
        return 0.0

    tp = np.cumsum(y_sorted)
    precision = tp / np.arange(1, len(y_sorted) + 1)
    recall = tp / n_pos
    recall_diff = np.diff(recall, prepend=0.0)
    return float(np.sum(precision * recall_diff))


def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve via trapezoidal integration.

    Returns 0.5 if all labels are the same class (AUC undefined).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    sorted_idx = np.lexsort((-y_true, -y_score))
    y_sorted = y_true[sorted_idx]

    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    tpr = np.r_[0.0, np.cumsum(y_sorted) / n_pos]
    fpr = np.r_[0.0, np.cumsum(1 - y_sorted) / n_neg]
    return float(np.trapezoid(tpr, fpr))


def f1_at_optimal_threshold(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[float, float]:
    """F1 score and the threshold that maximises it.

    Returns (best_f1, best_threshold). Iterates over all unique score values.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.sum() == 0:
        return 0.0, float("nan")

    thresholds = np.unique(y_score)
    best_f1, best_thresh = 0.0, float(thresholds[0])

    for t in thresholds:
        y_pred = (y_score >= t).astype(float)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        denom_p = tp + fp
        denom_r = tp + fn
        if denom_p == 0 or denom_r == 0:
            continue

        p = tp / denom_p
        r = tp / denom_r
        if p + r == 0:
            continue

        f1 = 2 * p * r / (p + r)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)

    return float(best_f1), best_thresh
