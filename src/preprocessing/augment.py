"""Augmentation primitives for SFX diffraction images.

All functions operate on 2D float32 numpy arrays (H, W) and accept an explicit
np.random.Generator so callers control reproducibility. They are stateless and
produce no side effects — safe for DataLoader multiprocessing workers.
"""

from __future__ import annotations

import numpy as np


def random_crop(image: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """Extract a random (size × size) patch from a 2D image.

    Args:
        image: 2D float32 array (H, W). Must have H >= size and W >= size.
        size: Side length of the square crop in pixels.
        rng: Numpy Generator (e.g. np.random.default_rng()).

    Returns:
        float32 array of shape (size, size).

    Raises:
        ValueError: If image dimensions are smaller than size.
    """
    h, w = image.shape
    if h < size or w < size:
        raise ValueError(
            f"random_crop: image ({h}×{w}) is smaller than requested crop ({size}×{size})."
        )
    top = int(rng.integers(0, h - size + 1))
    left = int(rng.integers(0, w - size + 1))
    return image[top : top + size, left : left + size]


def center_crop(image: np.ndarray, size: int) -> np.ndarray:
    """Extract the centre (size × size) patch from a 2D image.

    Deterministic — no RNG needed. Used on the eval/test path.

    Args:
        image: 2D float32 array (H, W). Must have H >= size and W >= size.
        size: Side length of the square crop in pixels.

    Returns:
        float32 array of shape (size, size).

    Raises:
        ValueError: If image dimensions are smaller than size.
    """
    h, w = image.shape
    if h < size or w < size:
        raise ValueError(
            f"center_crop: image ({h}×{w}) is smaller than requested crop ({size}×{size})."
        )
    top = (h - size) // 2
    left = (w - size) // 2
    return image[top : top + size, left : left + size]


def random_rot90(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rotate image by a random multiple of 90°.

    k is drawn uniformly from {0, 1, 2, 3} — rotation by k × 90° counter-clockwise.

    Args:
        image: 2D float32 array (H, W).
        rng: Numpy Generator.

    Returns:
        float32 array (same shape as input for square inputs; transposed dims otherwise).
    """
    k = int(rng.integers(0, 4))
    return np.rot90(image, k=k)


def random_flip(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply independent 50% horizontal and 50% vertical flips.

    Args:
        image: 2D float32 array (H, W).
        rng: Numpy Generator.

    Returns:
        float32 array of the same shape.
    """
    bits = rng.integers(0, 2, size=2)
    if bits[0]:
        image = np.fliplr(image)
    if bits[1]:
        image = np.flipud(image)
    return image


def patch_grid(
    image: np.ndarray,
    patch_size: int = 224,
    stride: int | None = None,
) -> list[np.ndarray]:
    """Tile a 2D image into a grid of (patch_size × patch_size) patches.

    Partial patches at the right/bottom edge are discarded — no padding.
    Patches are returned in row-major order (left-to-right, top-to-bottom).

    Args:
        image: 2D float32 array (H, W).
        patch_size: Side length of each square patch in pixels.
        stride: Step between patch origins. Defaults to patch_size
            (non-overlapping). A stride < patch_size produces overlapping
            patches for better edge coverage at the cost of more forward passes.

    Returns:
        List of float32 arrays each of shape (patch_size, patch_size).
        Empty list if the image is smaller than patch_size in either dim.
    """
    if stride is None:
        stride = patch_size
    h, w = image.shape
    patches = []
    for top in range(0, h - patch_size + 1, stride):
        for left in range(0, w - patch_size + 1, stride):
            patches.append(image[top : top + patch_size, left : left + patch_size])
    return patches


def random_cutout(
    image: np.ndarray,
    rng: np.random.Generator,
    n_holes: int = 3,
    hole_size: int = 32,
) -> np.ndarray:
    """Zero-fill n_holes random (hole_size × hole_size) rectangular patches.

    Simulates detector panel gaps or masked regions. Applied after GCN → LCN so
    masked pixels are set to 0.0, which is the approximate post-normalisation mean.

    Args:
        image: 2D float32 array (H, W).
        rng: Numpy Generator.
        n_holes: Number of rectangular cutout patches to apply.
        hole_size: Side length of each square cutout patch in pixels.

    Returns:
        float32 array of the same shape, with n_holes regions zeroed out.
    """
    result = image.copy()
    h, w = image.shape
    for _ in range(n_holes):
        top = int(rng.integers(0, max(1, h - hole_size + 1)))
        left = int(rng.integers(0, max(1, w - hole_size + 1)))
        result[top : top + hole_size, left : left + hole_size] = 0.0
    return result
