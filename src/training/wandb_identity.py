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
from collections.abc import Callable
from pathlib import Path

from src.training.inference_results import WANDB_API_TIMEOUT_S, wandb_enabled

WANDB_ID_FILE = "wandb_id.txt"
OVERRIDDEN_TAG = "overridden"


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


def _set_overridden_tag(
    project: str | None, entity: str | None, run_id: str, present: bool
) -> bool:
    """Add (`present`) or remove the ``overridden`` tag. Never raises: a failure only warns."""
    if project is None or not wandb_enabled():
        return False
    try:
        import wandb

        run = wandb.Api(timeout=WANDB_API_TIMEOUT_S).run(
            _run_path(project, entity, run_id)
        )
        tags = list(run.tags)
        if present and OVERRIDDEN_TAG not in tags:
            run.tags = tags + [OVERRIDDEN_TAG]
            run.update()
        elif not present and OVERRIDDEN_TAG in tags:
            run.tags = [t for t in tags if t != OVERRIDDEN_TAG]
            run.update()
        return True
    except Exception as exc:
        verb = "tag" if present else "untag"
        print(
            f"  [wandb] could not {verb} {run_id!r} as overridden ({exc!r}); continuing."
        )
        return False


def _attempt_number(name: str, run_id: str) -> int:
    m = re.fullmatch(re.escape(name) + r"-o([0-9]+)", run_id)
    return int(m.group(1)) if m else 1


def select_current_runs(runs):
    """Runs to plot: drop `overridden`-tagged ones, then keep the highest attempt per name.

    The attempt number comes from the id (``<name>`` = 1, ``<name>-oN`` = N), so the
    superseded attempt is dropped even when its tag was never written (offline mode).
    """
    best: dict[str, object] = {}
    for run in runs:
        if OVERRIDDEN_TAG in run.tags:
            continue
        cur = best.get(run.name)
        if cur is None or _attempt_number(run.name, run.id) > _attempt_number(
            cur.name, cur.id
        ):
            best[run.name] = run
    return list(best.values())


def _run_path(project: str, entity: str | None, run_id: str) -> str:
    return f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"


MAX_FRESH_ID_ATTEMPTS = 20


def ensure_fresh_wandb_id(
    run_dir: str | Path, run_name: str, project: str | None, entity: str | None
) -> str:
    """The id for a FRESH training start: never one whose W&B run already exists.

    Reusing an existing run id with ``step=epoch`` drops or splices metrics, which
    happens when the checkpoint dir was removed by hand or a crash left no
    checkpoint. Existing runs on the chain are tagged ``overridden`` and the id file
    is rotated past them. If W&B cannot be queried the current id is kept.
    """
    if project is None or not wandb_enabled():
        return resolve_wandb_id(run_dir, run_name)
    import wandb

    for _ in range(MAX_FRESH_ID_ATTEMPTS):
        run_id = resolve_wandb_id(run_dir, run_name)
        try:
            wandb.Api(timeout=WANDB_API_TIMEOUT_S).run(
                _run_path(project, entity, run_id)
            )
        except Exception as exc:
            if "not found" in str(exc).lower():
                return run_id
            print(
                f"  [wandb] could not check whether run {run_id!r} exists ({exc!r}); "
                "keeping it."
            )
            return run_id
        tag_overridden(project, entity, run_id)
        _, new = rotate_wandb_id(run_dir, run_name)
        print(f"  [wandb] run {run_id!r} already exists; fresh start uses {new!r}.")
    return resolve_wandb_id(run_dir, run_name)


def wandb_id_for_training(
    run_dir: str | Path,
    run_name: str,
    project: str | None,
    entity: str | None,
    resuming: bool,
) -> str:
    """Id for the training `wandb.init`: current id when resuming, else a fresh one."""
    if resuming:
        return resolve_wandb_id(run_dir, run_name)
    return ensure_fresh_wandb_id(run_dir, run_name, project, entity)


def tag_overridden(project: str | None, entity: str | None, run_id: str) -> bool:
    """Add the ``overridden`` tag to a W&B run. Never raises: a failure only warns."""
    return _set_overridden_tag(project, entity, run_id, True)


class OverrideHook:
    """`check_checkpoint_collisions(on_override=...)` callback: rotate the id, tag the old run."""

    def __init__(self, project: str | None, entity: str | None) -> None:
        self.project = project
        self.entity = entity
        self.rotations: list[tuple[str, str, str]] = []

    def __call__(self, run_name: str, run_dir: Path) -> Callable[[], None]:
        """Rotate the id and tag the old run; return a callable undoing both."""
        id_file = Path(run_dir) / WANDB_ID_FILE
        previous = id_file.read_text() if id_file.is_file() else None
        old, new = rotate_wandb_id(run_dir, run_name)
        self.rotations.append((run_name, old, new))
        print(f"  [wandb] override: logging to new run {new!r}; {old!r} is kept.")
        tag_overridden(self.project, self.entity, old)

        def rollback() -> None:
            if previous is None:
                id_file.unlink(missing_ok=True)
            else:
                id_file.write_text(previous)
            if self.rotations and self.rotations[-1] == (run_name, old, new):
                self.rotations.pop()
            _set_overridden_tag(self.project, self.entity, old, False)
            print(f"  [wandb] override rolled back: {old!r} is the live run again.")

        return rollback


def override_hook_from_cfg(cfg: dict) -> OverrideHook:
    wandb_cfg = cfg.get("wandb", {})
    return OverrideHook(wandb_cfg.get("project"), wandb_cfg.get("entity"))
