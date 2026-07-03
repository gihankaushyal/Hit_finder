# src/hitfinders/gpu.py
"""GPU-accelerated hitfinder stub.

Worker safety: GPU hitfinder CANNOT run inside DataLoader workers with
num_workers > 0 (no shared CUDA context across forked processes).
When using GPUHitfinder, set num_workers=0 in asymmetric_loader().
"""

from __future__ import annotations

import numpy as np


class GPUHitfinder:
    """GPU-accelerated peak finder.

    Implementation to be provided by user. Replace find_peaks() body
    with the actual GPU inference call once the hitfinder script is
    integrated.

    WARNING: num_workers must be 0 when using this backend — GPU context
    is not shareable across forked DataLoader worker processes.
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = device

    def find_peaks(self, assembled: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "GPUHitfinder.find_peaks is not yet implemented. "
            "Integrate the GPU hitfinder script to complete this method. "
            "Remember to set num_workers=0 in asymmetric_loader()."
        )
