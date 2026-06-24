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
#   entry_1/data_1/data                                     → ePix10k (.cxi) and Resonet Eiger CXI
#   entry_0000/instrument/Simulator/data                    → JUNGFRAU 4M (.h5)
_HDF5_CANDIDATE_KEYS = [
    "entry/data/data",
    "entry_1/instrument_1/detector_1/detector_corrected/data",
    "entry_1/data_1/data",
    "entry_0000/instrument/Simulator/data",
]

# Default key for embedded hit labels in multi-frame CXI files (Resonet format).
_DEFAULT_LABEL_KEY = "entry_1/labels/hit"


def _find_hdf5_data_key(f: h5py.File) -> str:
    """Return the first matching candidate key in an open HDF5 file.

    Raises:
        KeyError: If no candidate key is found.
    """
    for key in _HDF5_CANDIDATE_KEYS:
        if key in f:
            return key
    raise KeyError(
        f"No known data key found in HDF5 file. Tried: {_HDF5_CANDIDATE_KEYS}. "
        "Run scripts/probe_hdf5.py to inspect the file structure."
    )


def count_frames(path: str | Path) -> int:
    """Return the number of frames in a multi-frame HDF5/CXI file.

    For files with a leading batch dimension (N, H, W) the batch size N is
    returned. For 2D files (H, W) returns 1.

    Args:
        path: Path to an HDF5 or CXI file.

    Returns:
        Number of frames (int ≥ 1).

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If no known HDF5 data key is found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    with h5py.File(path, "r") as f:
        key = _find_hdf5_data_key(f)
        shape = f[key].shape
    return shape[0] if len(shape) == 3 else 1


def read_frame(path: str | Path, frame_idx: int = 0) -> np.ndarray:
    """Return a specific 2D float32 frame from an HDF5/CXI file.

    Generalises read_image() with a configurable frame index. Reads only the
    requested frame from disk (via HDF5 chunk slicing) rather than loading the
    full dataset into RAM.

    Args:
        path: Path to an HDF5 or CXI file.
        frame_idx: Zero-based frame index for multi-frame (N, H, W) files.

    Returns:
        2D numpy array, shape (H, W), dtype float32.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If no known HDF5 data key is found.
        IndexError: If frame_idx is out of range.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    with h5py.File(path, "r") as f:
        key = _find_hdf5_data_key(f)
        dataset = f[key]
        if dataset.ndim == 3:
            if frame_idx >= dataset.shape[0]:
                raise IndexError(
                    f"frame_idx {frame_idx} out of range for dataset with "
                    f"{dataset.shape[0]} frames."
                )
            data = dataset[frame_idx]
        else:
            data = dataset[()]
    return data.astype(np.float32)


def read_embedded_labels(
    path: str | Path,
    label_key: str = _DEFAULT_LABEL_KEY,
) -> np.ndarray:
    """Return the embedded label array from a multi-frame CXI file.

    Labels are stored as float32 (1.0 = hit, 0.0 = non-hit) in Resonet CXI
    files. The returned array has one entry per frame.

    Args:
        path: Path to an HDF5 or CXI file.
        label_key: HDF5 key for the label dataset.

    Returns:
        1D numpy array of float32 labels, shape (N,).

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If label_key is not found in the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    with h5py.File(path, "r") as f:
        if label_key not in f:
            raise KeyError(
                f"Label key '{label_key}' not found in {path}. "
                "Run scripts/probe_hdf5.py to inspect available keys."
            )
        return f[label_key][()].astype(np.float32)


def _read_hdf5_frame(f: h5py.File) -> np.ndarray:
    key = _find_hdf5_data_key(f)
    data = f[key][()]
    if data.ndim == 3:
        return data[0]
    return data


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
