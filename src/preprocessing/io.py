"""Unified image reader: dispatches .img (fabio) and .h5/.cxi (h5py)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

_SUPPORTED = {".img", ".h5", ".cxi"}

# Keys tried in order for HDF5/CXI files. First match wins.
# Confirmed against real detector files (2026-05-22):
#   entry/data/data                                          → Eiger4M (.h5)
#   entry_1/instrument_1/detector_1/detector_corrected/data → AGIPD (.cxi)
#   entry_1/data_1/data                                     → ePix10k (.cxi)
#   entry_0000/instrument/Simulator/data                    → JUNGFRAU 4M (.h5)
_HDF5_CANDIDATE_KEYS = [
    "entry/data/data",
    "entry_1/instrument_1/detector_1/detector_corrected/data",
    "entry_1/data_1/data",
    "entry_0000/instrument/Simulator/data",
]


def _read_hdf5_frame(f: h5py.File) -> np.ndarray:
    """Return the first 2D frame from an open HDF5 file.

    Tries candidate keys in order. Handles a leading batch dimension
    (N, H, W) by returning frame 0.

    Raises:
        KeyError: If no candidate key is found in the file.
    """
    for key in _HDF5_CANDIDATE_KEYS:
        if key in f:
            data = f[key][()]
            if data.ndim == 3:
                return data[0]
            return data
    raise KeyError(
        f"No known data key found in HDF5 file. Tried: {_HDF5_CANDIDATE_KEYS}. "
        "Run scripts/probe_hdf5.py to inspect the file structure."
    )


def read_image(path: str | Path) -> np.ndarray:
    """Return a single 2D float32 frame from a diffraction image file.

    Supports:
      .img        → fabio (ADSC, MAR, and similar crystallography formats)
      .h5 / .cxi  → h5py, tries candidate keys; returns frame 0 for multi-frame files

    Args:
        path: Path to the image file.

    Returns:
        2D numpy array, shape (H, W), dtype float32. No channel dimension.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported.
        KeyError: If no known HDF5 data key is found in the file.
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
        return _read_hdf5_frame(f).astype(np.float32)
