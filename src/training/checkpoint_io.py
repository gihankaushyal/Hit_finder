"""Checkpoint writes that a SLURM preemption cannot truncate."""

from __future__ import annotations

import os
from pathlib import Path

import torch


def atomic_torch_save(obj: object, path: str | Path) -> None:
    """`torch.save` to ``<path>.tmp``, fsync, then rename over `path`.

    A kill mid-write leaves the previous file intact (a resume would otherwise
    crash-loop on a truncated checkpoint). The fsync makes the data durable before
    the rename, so a node crash cannot leave a renamed but empty file. The directory
    is not fsynced: a node crash right after the rename may keep the previous file,
    which is still a complete checkpoint. An exception removes the partial tmp file;
    a SIGTERM or SIGKILL does not, and --override-training deletes it (``*.tmp``).
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
