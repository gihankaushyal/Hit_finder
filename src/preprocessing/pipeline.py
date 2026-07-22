"""Full preprocessing pipeline: geometry → crop 224×224 → augment → GCN → LCN."""

from __future__ import annotations

import numpy as np

from reborn.detector import PADAssembler, PADGeometryList


from src.preprocessing.geometry import extract_panels_from_canvas
from src.preprocessing.normalize import LCN_WINDOW_DEFAULT, gcn, lcn

TARGET_SIZE: tuple[int, int] = (224, 224)


def _to_2d(
    image: np.ndarray,
    pad_ss: int | None = None,
    pad_fs: int | None = None,
) -> np.ndarray:
    """Reshape any assembled detector output to 2D (H, W).

    - 2D input: returned unchanged.
    - 3D input (AGIPD modules): row-stacked → (n_modules * n_ss, n_fs).
    - 1D input (Eiger4M monolithic): reshaped using pad_ss × pad_fs.

    Args:
        image: Assembled array from assemble_image().
        pad_ss: Slow-scan dimension of a single Eiger4M panel (required for 1D).
        pad_fs: Fast-scan dimension of a single Eiger4M panel (required for 1D).

    Raises:
        ValueError: If input is 1D and pad_ss/pad_fs are not provided.
    """
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        return image.reshape(-1, image.shape[-1])
    if image.ndim == 1:
        if pad_ss is None or pad_fs is None:
            raise ValueError(
                "pad_ss and pad_fs are required to reshape 1D (Eiger4M) output."
            )
        return image.reshape(pad_ss, pad_fs)
    raise ValueError(f"Unexpected image ndim {image.ndim}; expected 1, 2, or 3.")



def assemble_only(
    frame: np.ndarray,
    pads: PADGeometryList,
    detector_desc: str,
    assembler: PADAssembler | None = None,
) -> np.ndarray:
    """Assemble raw detector frame to native-resolution 2D without normalisation.

    Mirrors the assembly logic in preprocess_with_geometry but stops before
    GCN/LCN/resize — used by the augmentation pipeline to get the full-size
    image for random cropping.

    Args:
        frame: Raw frame from CXI file (detector-native shape).
        pads: PADGeometryList from get_geometry(detector_desc).
        detector_desc: Detector description string from CXI metadata.
        assembler: Optional pre-built PADAssembler (avoids recomputing flat_indices).

    Returns:
        float32 array of shape (H, W) at native detector resolution.

    Raises:
        ValueError: If detector_desc is unrecognised.
    """
    if detector_desc in ("AGIPD 1M", "ePix10k 2.2M"):
        flat = frame.ravel().astype(np.float32)
    elif detector_desc == "EIGER 4M":
        panels = extract_panels_from_canvas(frame.astype(np.float32), pads)
        flat = np.concatenate([p.ravel() for p in panels])
    else:
        raise ValueError(
            f"assemble_only: unrecognised detector_desc '{detector_desc}'. "
            "For Jungfrau 4M use _to_2d() directly (pre-assembled canvas)."
        )
    if assembler is None:
        assembler = PADAssembler(pad_geometry=pads)
    return assembler.assemble_data(flat).astype(np.float32)



def preprocess_eval_patches(
    assembled: np.ndarray,
    patch_size: int = TARGET_SIZE[0],
    stride: int | None = None,
    lcn_window: int = LCN_WINDOW_DEFAULT,
) -> np.ndarray:
    """GCN → LCN each patch from a patch_grid tiling of the assembled image.

    Used for all evaluation paths (validation, in-domain test, cross-detector
    test). The full native-resolution assembled frame is tiled into complete
    (patch_size × patch_size) patches; each patch is normalised independently.

    Args:
        assembled: float32 array (H, W) at native detector resolution.
        patch_size: Patch side length in pixels (default 224).
        stride: Step between patch origins (default = patch_size, non-overlapping).
        lcn_window: LCN neighbourhood size (default 9, Phase 3 ablation).

    Returns:
        float32 array of shape (N, patch_size, patch_size) where N ≥ 1.

    Raises:
        ValueError: If the image produces zero complete patches.
    """
    from src.preprocessing.augment import patch_grid

    patches = patch_grid(assembled.astype(np.float32), patch_size, stride)
    if not patches:
        raise ValueError(
            f"preprocess_eval_patches: no complete {patch_size}×{patch_size} "
            f"patch fits in image of shape {assembled.shape}."
        )
    normed = [lcn(gcn(p), window=lcn_window) for p in patches]
    return np.stack(normed, axis=0).astype(np.float32)
