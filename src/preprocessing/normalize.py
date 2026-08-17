"""GCN and LCN normalization. Order: GCN → LCN. Never reversed."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

GCN_EPSILON: float = 1e-6
# Chosen via eps ablation (notebooks/pipeline_debug.ipynb, 2026-08-17): floors the
# denominator at sqrt(1e-2)=0.1 GCN units — suppresses background readout noise
# (out_std 1.0 → 0.13) while preserving Bragg peak amplitude and moderate texture.
LCN_EPSILON: float = 1e-2
LCN_WINDOW_DEFAULT: int = 9


def gcn(image: np.ndarray, eps: float = GCN_EPSILON) -> np.ndarray:
    """Global Contrast Normalization: (I - μ) / (σ + ε).

    Subtracts the global mean and divides by the global standard deviation.
    ε prevents division by zero on uniform images.

    Args:
        image: 2D float array (H, W).
        eps: Stability term added to the denominator.

    Returns:
        Normalized array, same shape and dtype float64.
    """
    image = image.astype(np.float64)
    mu = image.mean()
    sigma = image.std()
    return (image - mu) / (sigma + eps)


def lcn(
    image: np.ndarray,
    window: int = LCN_WINDOW_DEFAULT,
    eps: float = LCN_EPSILON,
) -> np.ndarray:
    """Local Contrast Normalization: (I(x,y) - μ_W(x,y)) / sqrt(σ²_W(x,y) + ε).

    Subtracts a local mean and divides by a local standard deviation computed
    over a square window of side `window`. Uses uniform (box) filtering for
    speed. Window size is a Phase 3 ablation parameter.

    ε is added to the local *variance* (not the std), flooring the denominator
    at sqrt(ε). This prevents noise explosion in low-variance background
    regions, where σ_W ≈ 0 and a std-form ε of 1e-6 provides no stabilization
    (observed as salt-and-pepper static on JUNGFRAU non-hit patches).

    Args:
        image: 2D float array (H, W). Typically the output of gcn().
        window: Side length of the local neighbourhood (must be odd ≥ 1).
        eps: Stability term added to the local variance in the denominator.

    Returns:
        Locally normalized array, same shape and dtype float64.
    """
    image = image.astype(np.float64)
    local_mean = uniform_filter(image, size=window)
    local_sq_mean = uniform_filter(image**2, size=window)
    local_var = np.maximum(local_sq_mean - local_mean**2, 0.0)
    return (image - local_mean) / np.sqrt(local_var + eps)
