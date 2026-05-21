"""Unified image reader: dispatches .img (fabio) and .h5/.cxi (h5py)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

_SUPPORTED = {".img", ".h5", ".cxi"}


def read_image(path: str | Path) -> np.ndarray:
    """Return a 2D float32 array from a diffraction image file.

    Supports:
      .img        → fabio (ADSC, MAR, and similar crystallography formats)
      .h5 / .cxi  → h5py, key 'entry/data/data'

    Args:
        path: Path to the image file.

    Returns:
        2D numpy array, shape (H, W), dtype float32. No channel dimension.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. Supported: {sorted(_SUPPORTED)}"
        )

    if suffix == ".img":
        import fabio

        with fabio.open(str(path)) as img_file:
            return img_file.data.astype(np.float32)

    with h5py.File(path, "r") as f:
        return f["entry/data/data"][()].astype(np.float32)
