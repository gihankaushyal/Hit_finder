"""Augmentation primitives for SFX diffraction images.

All functions operate on 2D float32 numpy arrays (H, W) and accept an explicit
np.random.Generator so callers control reproducibility. They are stateless and
produce no side effects — safe for DataLoader multiprocessing workers.
"""

from __future__ import annotations

import numpy as np

PAD_BORDER_DEFAULT = 112


def pad_border(image: np.ndarray, pad_size: int = PAD_BORDER_DEFAULT) -> np.ndarray:
    """Symmetrically pad all four edges with zeros.

    Args:
        image: float32 array of shape (H, W).
        pad_size: Number of zero pixels to add on each edge.

    Returns:
        float32 array of shape (H + 2*pad_size, W + 2*pad_size).
    """
    return np.pad(
        image,
        ((pad_size, pad_size), (pad_size, pad_size)),
        mode="constant",
        constant_values=0.0,
    )


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

    Simulates detector panel gaps or masked regions. Applied before GCN → LCN;
    zeroed pixels will be pulled toward the normalised mean during normalisation.

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
