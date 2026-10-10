"""Result-file and W&B helpers for ``--inference-only`` runs (see lodo._train_fold).

Kept as small pure functions so the rules (never clobber a closed-out
``results.json``; never send training-step metrics for an inference pass) are
unit-testable without evaluating anything.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path

RESULTS_NAME = "results.json"
INFERENCE_RESULTS_NAME = "results.inference.json"
WANDB_OFF_MODES = frozenset({"offline", "disabled"})

# summary key -> key in the dict returned by run_patch_agg
_SUMMARY_METRICS = {"ap": "ap", "auc": "auc_roc", "f1": "f1", "threshold": "threshold"}


def inference_result_path(ckpt_dir: str | Path) -> Path:
    """`results.json` if the run has none (this completes the run), else the side file."""
    ckpt_dir = Path(ckpt_dir)
    primary = ckpt_dir / RESULTS_NAME
    return ckpt_dir / INFERENCE_RESULTS_NAME if primary.exists() else primary


def inference_block(
    *,
    aggregation: str,
    patch_stride: int,
    min_hit_patches: int,
    checkpoint: dict,
) -> dict:
    """Provenance recorded in `results.inference.json`."""
    return {
        "aggregation": aggregation,
        "patch_stride": patch_stride,
        "min_hit_patches": min_hit_patches,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_f1": checkpoint.get("val_f1"),
        "evaluated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
    }


def summary_updates(
    in_domain_m: dict, cross_m: dict, threshold: float
) -> dict[str, float]:
    """W&B run-summary entries for an inference pass (not step-indexed)."""
    updates: dict[str, float] = {"inference/threshold": threshold}
    for label, metrics in (("in_domain", in_domain_m), ("cross", cross_m)):
        for name, key in _SUMMARY_METRICS.items():
            updates[f"inference/{label}/{name}"] = metrics[key]
    return updates


def wandb_enabled() -> bool:
    """False when W&B is offline or disabled, so nothing is sent.

    Checks WANDB_MODE and W&B's own resolved settings, which also cover a mode set
    with the `wandb offline` / `wandb disabled` commands (stored in a settings file,
    not in the environment). If the settings cannot be read, W&B is assumed on.
    """
    if os.environ.get("WANDB_MODE", "").lower() in WANDB_OFF_MODES:
        return False
    try:
        import wandb

        mode = str(wandb.setup().settings.mode).lower()
    except Exception:
        return True
    return mode not in WANDB_OFF_MODES
