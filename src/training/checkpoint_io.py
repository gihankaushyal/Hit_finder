"""Checkpoint writes that a SLURM preemption cannot truncate."""

from __future__ import annotations

import os
from pathlib import Path

import torch


def atomic_torch_save(obj: object, path: str | Path) -> None:
    """`torch.save` to ``<path>.tmp``, then rename over `path`.

    A kill mid-write leaves the previous file intact (a resume would otherwise
    crash-loop on a truncated checkpoint); the partial tmp file is removed.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
