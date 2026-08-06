# src/hitfinders/base.py
"""Hitfinder Protocol and MockHitfinder for testing."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Hitfinder(Protocol):
    """Interface for peak-finding algorithms.

    Implementations must be stateless with respect to frame content —
    the same assembled frame must always produce the same centroids
    (deterministic). Implementations may hold configuration state
    (thresholds, device handles) set at __init__ time.
    """

    def find_peaks(self, assembled: np.ndarray) -> np.ndarray:
        """Locate Bragg peak centroids in an assembled detector frame.

        Args:
            assembled: 2D float32 array (H, W) — assembled detector image.

        Returns:
            float32 array of shape (N_peaks, 2) where each row is [x, y]
            in assembled-frame pixel coordinates (x=column, y=row).
            Returns shape (0, 2) when no peaks are found.
        """
        ...


class MockHitfinder:
    """Deterministic hitfinder that returns a fixed set of centroids.

    Used in unit tests where real peak detection is not needed.
    """

    def __init__(self, peaks: np.ndarray | None = None) -> None:
        if peaks is None:
            peaks = np.zeros((0, 2), dtype=np.float32)
        self._peaks = np.asarray(peaks, dtype=np.float32)

    def find_peaks(self, assembled: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return self._peaks.copy()
