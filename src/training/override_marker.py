"""Durable record that ``--override-training`` was already applied by this launch.

``--override-training`` deletes the existing checkpoint, so re-running the same launch
after the new attempt wrote one would destroy it. A requeued SLURM job keeps its
``SLURM_JOB_ID`` and gets a higher ``SLURM_RESTART_COUNT``; a new ``sbatch`` gets a new
job id. After a run is overridden the gate writes ``<run_dir>/override_applied.json``
with the launching job id and restart count, and a later override request is treated as
a resume only from the SAME job id with a HIGHER restart count (a requeue). Two launches
inside one allocation (``salloc``, ``srun --pty``) share both, so the second is a
deliberate override.

Outside SLURM there is no job id, so no marker is written and the flag is honoured on
every launch (each direct invocation is deliberate).

Known limit, by design: the marker is keyed on the launch, not on the checkpoint. It is
never cleared, so copying a different checkpoint into the run directory and requeueing
the same job resumes from that file, and a recycled job id (cluster counter reset)
could skip a requested override.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

OVERRIDE_MARKER_FILE = "override_applied.json"
JOB_ID_ENV = "SLURM_JOB_ID"
RESTART_COUNT_ENV = "SLURM_RESTART_COUNT"


def current_job_id() -> str | None:
    """The SLURM job id of this process, or None outside SLURM."""
    return os.environ.get(JOB_ID_ENV) or None


def current_restart_count() -> int:
    """How many times SLURM has requeued this job (unset, empty or garbage -> 0)."""
    try:
        return int(os.environ.get(RESTART_COUNT_ENV, "0"))
    except ValueError:
        return 0


def write_override_marker(run_dir: str | Path, run_name: str) -> bool:
    """Record that this job id applied the override. False (nothing written) outside SLURM.

    Written atomically. Raises OSError on failure; the gate catches it and warns, because
    it runs after the destructive steps and a missing marker only costs requeue protection.
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
                "restart_count": current_restart_count(),
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
    """True when this process is a requeue of the launch that wrote the marker.

    Same job id and a higher restart count than the one recorded (older markers without
    a count read as 0).
    """
    job_id = current_job_id()
    if job_id is None:
        return False
    try:
        data = json.loads((Path(run_dir) / OVERRIDE_MARKER_FILE).read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict) or data.get("job_id") != job_id:
        return False
    recorded = data.get("restart_count", 0)
    return isinstance(recorded, int) and current_restart_count() > recorded
