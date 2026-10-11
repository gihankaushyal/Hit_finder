"""Durable record that ``--override-training`` was already applied by this launch.

``--override-training`` deletes the existing checkpoint, so re-running the same launch
after the new attempt wrote one would destroy it. A requeued SLURM job keeps its
``SLURM_JOB_ID``; a new ``sbatch`` gets a new one. After a run is overridden the gate
writes ``<run_dir>/override_applied.json`` with the launching job id, and a later
override request from the SAME job id is treated as a resume.

Outside SLURM there is no job id, so no marker is written and the flag is honoured on
every launch (each direct invocation is deliberate).
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

OVERRIDE_MARKER_FILE = "override_applied.json"
JOB_ID_ENV = "SLURM_JOB_ID"


def current_job_id() -> str | None:
    """The SLURM job id of this process, or None outside SLURM."""
    return os.environ.get(JOB_ID_ENV) or None


def write_override_marker(run_dir: str | Path, run_name: str) -> bool:
    """Record that this job id applied the override. False (nothing written) outside SLURM.

    Written atomically. Raises OSError on failure: a missing marker would leave the new
    attempt's checkpoint unprotected against a requeue, so the launch must fail loudly.
    """
    job_id = current_job_id()
    if job_id is None:
        return False
    marker = Path(run_dir) / OVERRIDE_MARKER_FILE
    tmp = marker.with_name(marker.name + ".tmp")
    tmp.write_text(
        json.dumps(
            {
                "job_id": job_id,
                "run_name": run_name,
                "applied_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
                    timespec="seconds"
                ),
            }
        )
        + "\n"
    )
    try:
        os.replace(tmp, marker)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise
    return True


def override_already_applied(run_dir: str | Path) -> bool:
    """True when the marker names this process's job id (a requeue of the same launch)."""
    job_id = current_job_id()
    if job_id is None:
        return False
    try:
        data = json.loads((Path(run_dir) / OVERRIDE_MARKER_FILE).read_text())
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and data.get("job_id") == job_id
