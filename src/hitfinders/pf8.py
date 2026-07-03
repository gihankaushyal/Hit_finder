# src/hitfinders/pf8.py
"""PeakFinder8 C++ wrapper.

PF8Hitfinder is a stub. Implement find_peaks() once the C++ wrapper
interface is confirmed (subprocess args or ctypes binding).

Worker safety: PF8 C++ subprocess is safe across DataLoader fork workers.
"""

from __future__ import annotations

import numpy as np


class PF8Hitfinder:
    """Wraps the PeakFinder8 C++ binary installed on Sol HPC.

    Args:
        threshold_snr: Signal-to-noise threshold for peak acceptance.
        min_peaks: Minimum number of peaks required to report any peaks.
    """

    def __init__(self, threshold_snr: float = 5.0, min_peaks: int = 1) -> None:
        self.threshold_snr = threshold_snr
        self.min_peaks = min_peaks

    def find_peaks(self, assembled: np.ndarray) -> np.ndarray:
        """Call PF8 C++ wrapper and return centroids.

        Not yet implemented — awaiting C++ wrapper interface details.
        Replace this body with the actual subprocess/ctypes call.
        """
        raise NotImplementedError(
            "PF8Hitfinder.find_peaks is not yet implemented. "
            "Provide the C++ PF8 wrapper interface to complete this method."
        )
