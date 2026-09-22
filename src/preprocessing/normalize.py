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
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Local Contrast Normalization: (I(x,y) - μ_W(x,y)) / sqrt(σ²_W(x,y) + ε).

    Subtracts a local mean and divides by a local standard deviation computed
    over a square window of side `window`. Uses uniform (box) filtering for
    speed. Window size is a Phase 3 ablation parameter.

    ε is added to the local *variance* (not the std), flooring the denominator
    at sqrt(ε). This prevents noise explosion in low-variance background
    regions, where σ_W ≈ 0 and a std-form ε of 1e-6 provides no stabilization
    (observed as salt-and-pepper static on JUNGFRAU non-hit patches).

    When ``mask`` is given, invalid pixels (detector gaps, padding, eroded
    panel edges) are excluded from μ_W and σ_W via normalized convolution, so
    windows straddling a panel boundary see only real pixels — this removes
    the halo/ringing artifact LCN otherwise produces at gap edges. Invalid
    pixels are set to 0 in the output.

    Args:
        image: 2D float array (H, W). Typically the output of gcn().
        window: Side length of the local neighbourhood (must be odd ≥ 1).
        eps: Stability term added to the local variance in the denominator.
        mask: Optional boolean array (H, W); True = valid detector pixel.

    Returns:
        Locally normalized array, same shape and dtype float64.
    """
    image = image.astype(np.float64)
    if mask is None:
        local_mean = uniform_filter(image, size=window)
        local_sq_mean = uniform_filter(image**2, size=window)
    else:
        m = mask.astype(np.float64)
        # Zero invalid pixels before convolution: NaN * 0.0 = NaN (IEEE 754),
        # so multiplying by m is unsafe when image has NaN at gap locations.
        # np.where(mask, image, 0.0) correctly returns 0.0 where mask=False
        # regardless of the image value there (no NaN propagation).
        image_clean = np.where(mask, image, 0.0)
        count = np.maximum(
            uniform_filter(m, size=window, mode="constant", cval=0), 1e-12
        )
        local_mean = (
            uniform_filter(image_clean, size=window, mode="constant", cval=0) / count
        )
        local_sq_mean = (
            uniform_filter(image_clean**2, size=window, mode="constant", cval=0) / count
        )
    local_var = np.maximum(local_sq_mean - local_mean**2, 0.0)
    out = (image - local_mean) / np.sqrt(local_var + eps)
    if mask is not None:
        out[~mask] = 0.0
    return out


def lcn_torch(
    images: "torch.Tensor",
    window: int = LCN_WINDOW_DEFAULT,
    eps: float = LCN_EPSILON,
    masks: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """Batched masked Local Contrast Normalization on the caller's device.

    GPU counterpart of lcn(), used by the evaluation patch path. With assembly
    and GCN served from the frame cache, LCN is nearly all that remains of the
    eval cost, so it runs on the GPU alongside inference.

    Implements the *masked* branch of lcn() — a normalized convolution: a
    zero-padded box filter divided by the per-pixel count of valid neighbours.
    When masks is None an all-ones mask is used, which is NOT identical to
    lcn(image, mask=None): that branch uses scipy's reflect padding. Callers
    that need reflect semantics must stay on the NumPy path.

    Args:
        images: (B, H, W) or (B, 1, H, W) float tensor, typically GCN'd patches.
        window: Side length of the local neighbourhood (odd, >= 1).
        eps: Added to the local *variance*, flooring the denominator at sqrt(eps).
        masks: Optional bool/float tensor broadcastable to images' spatial shape;
            True = valid detector pixel. Invalid pixels are excluded from the
            local statistics and set to 0 in the output.

    Returns:
        float32 tensor with the same shape as `images`.
    """
    import torch
    import torch.nn.functional as F

    squeeze_channel = images.ndim == 4
    x = images if squeeze_channel else images.unsqueeze(1)
    x = x.to(torch.float32)

    if masks is None:
        m = torch.ones_like(x)
    else:
        m = masks if masks.ndim == 4 else masks.unsqueeze(1)
        m = m.to(torch.float32)

    pad = window // 2

    def _box(t: "torch.Tensor") -> "torch.Tensor":
        # Zero padding + count_include_pad=True reproduces scipy's
        # uniform_filter(mode="constant", cval=0) up to the 1/window**2 factor,
        # which cancels in the sum/count ratio below.
        return F.avg_pool2d(
            F.pad(t, (pad, pad, pad, pad), mode="constant", value=0.0),
            kernel_size=window,
            stride=1,
        )

    x_clean = torch.where(m > 0, x, torch.zeros_like(x))
    count = _box(m).clamp_min(1e-12)
    local_mean = _box(x_clean) / count
    local_sq_mean = _box(x_clean * x_clean) / count
    local_var = (local_sq_mean - local_mean * local_mean).clamp_min(0.0)

    out = (x - local_mean) / torch.sqrt(local_var + eps)
    out = torch.where(m > 0, out, torch.zeros_like(out))
    return out if squeeze_channel else out.squeeze(1)
