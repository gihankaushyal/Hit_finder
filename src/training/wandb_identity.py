"""W&B run identity across ``--override-training`` attempts.

Logging to an existing W&B run id with ``step=epoch`` after an override drops or
splices the new attempt's metrics (measured on wandb 0.27.0), and a deleted run's id
can never be reused. So each override attempt gets a NEW run id:

    attempt 1: <run_name>     attempt 2: <run_name>-o2     attempt 3: <run_name>-o3

The display name stays ``<run_name>``. The live id is stored in
``<run_dir>/wandb_id.txt``; it is absent for first attempts and legacy runs, whose id
is the run name. The previous attempt's run is kept and tagged ``overridden``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from src.training.inference_results import WANDB_API_TIMEOUT_S, wandb_enabled

WANDB_ID_FILE = "wandb_id.txt"
OVERRIDDEN_TAG = "overridden"  # scripts/plot_hit_frac.py keeps its own copy


def resolve_wandb_id(run_dir: str | Path, run_name: str) -> str:
    """The W&B run id for this run directory (the run name unless an override rotated it)."""
    id_file = Path(run_dir) / WANDB_ID_FILE
    if id_file.is_file():
        text = id_file.read_text().strip()
        if text:
            return text
    return run_name


def next_wandb_id(run_name: str, current: str) -> str:
    """Id of the attempt after `current`: ``<run_name>`` -> ``-o2`` -> ``-o3`` ..."""
    m = re.fullmatch(re.escape(run_name) + r"-o([0-9]+)", current)
    attempt = int(m.group(1)) + 1 if m else 2
    return f"{run_name}-o{attempt}"


def rotate_wandb_id(run_dir: str | Path, run_name: str) -> tuple[str, str]:
    """Write the next attempt's id to the id file (atomically); return ``(old_id, new_id)``."""
    old = resolve_wandb_id(run_dir, run_name)
    new = next_wandb_id(run_name, old)
    id_file = Path(run_dir) / WANDB_ID_FILE
    tmp_file = id_file.with_name(id_file.name + ".tmp")
    tmp_file.write_text(new + "\n")
    try:
        os.replace(tmp_file, id_file)  # atomic: a kill never leaves a truncated id
    except OSError:
        tmp_file.unlink(missing_ok=True)
        raise
    return old, new


def tag_overridden(project: str | None, entity: str | None, run_id: str) -> bool:
    """Add the ``overridden`` tag to a W&B run. Never raises: a failure only warns."""
    if project is None or not wandb_enabled():
        return False
    try:
        import wandb

        path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
        run = wandb.Api(timeout=WANDB_API_TIMEOUT_S).run(path)
        tags = list(run.tags)
        if OVERRIDDEN_TAG not in tags:
            run.tags = tags + [OVERRIDDEN_TAG]
            run.update()
        return True
    except Exception as exc:
        print(
            f"  [wandb] could not tag {run_id!r} as overridden ({exc!r}); continuing."
        )
        return False


class OverrideHook:
    """`check_checkpoint_collisions(on_override=...)` callback: rotate the id, tag the old run."""

    def __init__(self, project: str | None, entity: str | None) -> None:
        self.project = project
        self.entity = entity
        self.rotations: list[tuple[str, str, str]] = []

    def __call__(self, run_name: str, run_dir: Path) -> None:
        old, new = rotate_wandb_id(run_dir, run_name)
        self.rotations.append((run_name, old, new))
        print(f"  [wandb] override: logging to new run {new!r}; {old!r} is kept.")
        tag_overridden(self.project, self.entity, old)


def override_hook_from_cfg(cfg: dict) -> OverrideHook:
    wandb_cfg = cfg.get("wandb", {})
    return OverrideHook(wandb_cfg.get("project"), wandb_cfg.get("entity"))
