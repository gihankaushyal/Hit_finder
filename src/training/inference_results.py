"""Result-file and W&B helpers for ``--inference-only`` runs (see lodo._train_fold).

Kept as small pure functions so the rules (never clobber a closed-out
``results.json``; never send training-step metrics for an inference pass) are
unit-testable without evaluating anything.
"""

from __future__ import annotations

import datetime
import math
import os
from pathlib import Path

RESULTS_NAME = "results.json"
INFERENCE_RESULTS_NAME = "results.inference.json"
WANDB_OFF_MODES = frozenset({"offline", "disabled"})
WANDB_API_TIMEOUT_S = 60
WANDB_STATE_FINISHED = "finished"
# Training history logged by lodo._train_fold (one row per epoch).
HISTORY_KEYS = ["epoch", "train/loss", "val/f1"]
# "The loss has flattened": it improved by less than LOSS_FLAT_REL_TOL, relative to
# its value at the start of the window, over the last LOSS_FLAT_WINDOW logged epochs.
LOSS_FLAT_WINDOW = 5
LOSS_FLAT_REL_TOL = 0.01

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
    training_check: dict,
) -> dict:
    """Provenance recorded in every result file an inference pass writes."""
    return {
        "aggregation": aggregation,
        "patch_stride": patch_stride,
        "min_hit_patches": min_hit_patches,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_f1": checkpoint.get("val_f1"),
        "training_check": training_check,
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


def loss_flatness(
    losses: list[float],
    window: int = LOSS_FLAT_WINDOW,
    rel_tol: float = LOSS_FLAT_REL_TOL,
) -> dict | None:
    """Did the loss flatten over the last `window` finite values?

    Flat means the relative improvement from the first to the last value of the
    window is below `rel_tol` (a rising loss counts as flat: it is no longer
    improving). Returns None when fewer than `window` finite values exist.
    """
    finite = [x for x in losses if x is not None and math.isfinite(x)]
    if len(finite) < window:
        return None
    start, end = finite[-window], finite[-1]
    improvement = (start - end) / abs(start) if start != 0 else 0.0
    return {
        "window": window,
        "relative_improvement": improvement,
        "flat": improvement < rel_tol,
    }


def fetch_wandb_history(project: str, entity: str | None, run_id: str) -> dict | None:
    """Read-only copy of a run's per-epoch history, or None if unavailable.

    Uses the public API, so the run is neither resumed nor modified. Returns
    None (after printing a note) when W&B is offline/disabled, unreachable, or the
    run does not exist: the caller then falls back to epoch counts only.
    """
    if not wandb_enabled():
        return None
    try:
        import wandb

        path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
        run = wandb.Api(timeout=WANDB_API_TIMEOUT_S).run(path)
        rows = sorted(run.scan_history(keys=HISTORY_KEYS), key=lambda r: r["epoch"])
        state = run.state
    except Exception as exc:
        print(
            f"  [inference] could not read the W&B history ({exc!r}); epoch counts only."
        )
        return None
    return {
        "state": state,
        "epochs": [r["epoch"] for r in rows],
        "train_loss": [r["train/loss"] for r in rows],
        "val_f1": [r["val/f1"] for r in rows],
    }


def assess_training(
    configured_epochs: int, checkpoint_epoch: int | None, history: dict | None
) -> dict:
    """Was the run behind this checkpoint finished, and had its loss flattened?

    `status`: "complete" (W&B says the run finished; early stopping can end a run
    before `configured_epochs`), "incomplete" (any other W&B state, e.g. crashed or
    killed) or "unknown" (no W&B history). `warnings` explains every non-complete
    case, so results from a provisional checkpoint are never mistaken for a
    closed-out run.
    """
    check: dict = {
        "status": "unknown",
        "checkpoint_epoch": checkpoint_epoch,
        "configured_epochs": configured_epochs,
        "wandb_state": None,
        "last_logged_epoch": None,
        "train_loss_flat": None,
        "warnings": [],
    }
    warnings: list[str] = check["warnings"]
    if history is None:
        if checkpoint_epoch is not None and checkpoint_epoch < configured_epochs:
            warnings.append(
                f"cannot verify that training finished: the checkpoint is from epoch "
                f"{checkpoint_epoch} of {configured_epochs} and the W&B history is "
                "unavailable (early stopping would also explain it)."
            )
        return check

    state = history["state"]
    check["wandb_state"] = state
    check["last_logged_epoch"] = history["epochs"][-1] if history["epochs"] else None
    check["train_loss_flat"] = loss_flatness(history["train_loss"])
    if state == WANDB_STATE_FINISHED:
        check["status"] = "complete"
        return check

    check["status"] = "incomplete"
    warnings.append(
        f"training did not finish: W&B state '{state}', last logged epoch "
        f"{check['last_logged_epoch']}/{configured_epochs}; the checkpoint is from "
        f"epoch {checkpoint_epoch}. These results are provisional."
    )
    flat = check["train_loss_flat"]
    if flat is None:
        warnings.append(
            f"too few logged epochs (< {LOSS_FLAT_WINDOW}) to judge the loss trend."
        )
    elif flat["flat"]:
        warnings.append(
            f"train loss had flattened (improved {flat['relative_improvement']:.2%} "
            f"over the last {flat['window']} epochs): the checkpoint is probably "
            "close to what training would reach."
        )
    else:
        warnings.append(
            f"train loss was still decreasing ({flat['relative_improvement']:.2%} over "
            f"the last {flat['window']} epochs): this checkpoint is under-trained."
        )
    return check
