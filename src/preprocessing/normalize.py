"""GCN and LCN normalization. Order: GCN → LCN. Never reversed."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

GCN_EPSILON: float = 1e-6
LCN_EPSILON: float = 1e-6
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
    """Local Contrast Normalization: (I(x,y) - μ_W(x,y)) / (σ_W(x,y) + ε).

    Subtracts a local mean and divides by a local standard deviation computed
    over a square window of side `window`. Uses uniform (box) filtering for
    speed. Window size is a Phase 3 ablation parameter.

    Args:
        image: 2D float array (H, W). Typically the output of gcn().
        window: Side length of the local neighbourhood (must be odd ≥ 1).
        eps: Stability term added to the denominator.

    Returns:
        Locally normalized array, same shape and dtype float64.
    """
    image = image.astype(np.float64)
    local_mean = uniform_filter(image, size=window)
    local_sq_mean = uniform_filter(image**2, size=window)
    local_var = np.maximum(local_sq_mean - local_mean**2, 0.0)
    local_std = np.sqrt(local_var)
    return (image - local_mean) / (local_std + eps)
