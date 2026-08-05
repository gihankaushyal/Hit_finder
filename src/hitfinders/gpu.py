"""GPU-accelerated hitfinder wrapper.

Delegates to a user-provided Python script that exposes:
    find_peaks(frame: np.ndarray) -> np.ndarray

The script is imported once on the first find_peaks() call.

Worker safety: GPU CUDA context is NOT shareable across forked
DataLoader worker processes. When using this backend, set
num_workers=0 in asymmetric_loader() (enforced automatically).
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Callable

import numpy as np


class GPUHitfinder:
    """Wrapper around a user-provided GPU hitfinder script.

    The script must expose a top-level function:
        find_peaks(frame: np.ndarray) -> np.ndarray

    where frame is a 2D float32 array (H, W) — raw assembled image
    BEFORE GCN/LCN — and the return is float32 of shape (N_peaks, 2)
    with columns [x, y] in assembled-frame pixel coordinates.

    Args:
        script_path: Absolute path to the GPU hitfinder Python script.
        device: PyTorch device string (e.g. "cuda", "cuda:0", "cpu").
            Passed as the env variable HITFINDER_DEVICE before import so
            the script can pick it up via os.environ if it needs it.
    """

    def __init__(self, script_path: str | Path, device: str = "cuda") -> None:
        self._script_path = Path(script_path)
        self._device = device
        self._fn: Callable[[np.ndarray], np.ndarray] | None = None
        self._mod: object | None = None

    def _load(self) -> None:
        if not self._script_path.exists():
            raise FileNotFoundError(
                f"gpu_hitfinder script not found: {self._script_path}"
            )
        os.environ["HITFINDER_DEVICE"] = self._device
        spec = importlib.util.spec_from_file_location(
            "gpu_hitfinder_script", self._script_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {self._script_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        if not hasattr(mod, "find_peaks"):
            raise AttributeError(
                f"GPU hitfinder script at {self._script_path} must expose "
                "a top-level 'find_peaks(frame: np.ndarray) -> np.ndarray' function."
            )
        self._mod = mod
        self._fn = mod.find_peaks

    def set_geometry(self, **kwargs) -> None:
        """Forward geometry parameters to the script's set_geometry function if present."""
        if self._fn is None:
            self._load()
        if hasattr(self._mod, "set_geometry"):
            self._mod.set_geometry(**kwargs)  # type: ignore[union-attr]

    def find_peaks(self, assembled: np.ndarray) -> np.ndarray:
        """Locate peaks using the user-provided GPU hitfinder script.

        Args:
            assembled: 2D float32 array (H, W) — raw assembled image.

        Returns:
            float32 array of shape (N_peaks, 2): [x, y] columns.
        """
        if self._fn is None:
            self._load()
        result = self._fn(assembled)  # type: ignore[misc]
        return np.asarray(result, dtype=np.float32)
