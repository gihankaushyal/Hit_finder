"""Names of the recall-at-fixed-FPR metrics, free of heavy imports.

src/training/inference_results.py (imported by wandb_identity, which must not pull in
numpy or torch) needs the keys; metrics.py computes them and re-exports these names.
"""

from __future__ import annotations

# Fixed-FPR operating points: key suffix -> FPR. Recall at 1e-3 needs thousands of
# negatives per split (scripts/recall_fpr_stability.py decides what is reportable).
RECALL_FPR_TARGETS: dict[str, float] = {"1e-3": 1e-3, "1e-2": 1e-2}
# Result keys per split: on the aggregation score, then on the max-tile score.
RECALL_FPR_KEYS: tuple[str, ...] = tuple(
    key
    for t in RECALL_FPR_TARGETS
    for key in (
        f"recall_at_fpr_{t}",
        f"fpr_at_fpr_{t}",
        f"max_recall_at_fpr_{t}",
        f"max_fpr_at_fpr_{t}",
    )
)
