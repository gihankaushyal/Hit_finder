"""Full preprocessing pipeline: geometry → crop 224×224 → augment → GCN → LCN."""

from __future__ import annotations

import warnings

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


# Valid-pixel masks cached per detector description (built once per process).
_MASK_CACHE: dict[str, np.ndarray] = {}
_MASK_WARNED: set[str] = set()
# Per-(desc, frame_shape) cache so two files with the same detector description
# but different assembled canvas sizes each get their own validated entry.
_FRAME_MASK_CACHE: dict[tuple[str | None, tuple[int, ...]], np.ndarray | None] = {}

# Panel-edge pixels are physically larger on JUNGFRAU/Eiger sensors and collect
# more charge (measured: JF border pixels ~33% brighter than interior). They are
# real signal but not representative — standard SFX practice is to exclude a
# small panel border, so the valid mask is eroded by this many pixels.
EDGE_EROSION_PX: int = 2


def valid_pixel_mask(detector_desc: str) -> np.ndarray:
    """Boolean mask of assembled-canvas pixels covered by real detector panels.

    Built by assembling an all-ones flat array through the same PADAssembler
    used for data, so coverage exactly matches the runtime assembly. Gap and
    padding pixels (never written by any panel) come out False. Pixel-value
    heuristics (e.g. ``pixel == 0``) are deliberately avoided — zero is a
    legitimate value in photon-counting backgrounds.

    Args:
        detector_desc: CXI detector description, e.g. 'AGIPD 1M', 'EIGER 4M',
            'Jungfrau 4M' (routed through its CrystFEL geometry since the
            frame itself arrives pre-assembled).

    The mask is then eroded by EDGE_EROSION_PX so physically double-size
    panel-edge pixels (genuinely brighter, but unrepresentative) are also
    treated as invalid.

    Returns:
        Boolean array with the assembled canvas shape, cached per description.

    Raises:
        ValueError: If detector_desc is unrecognised.
    """
    from scipy.ndimage import binary_erosion

    from src.preprocessing.geometry import DETECTOR_LOADERS, get_assembler, get_geometry

    if detector_desc in _MASK_CACHE:
        return _MASK_CACHE[detector_desc]

    if detector_desc == "Jungfrau 4M":
        # Pre-assembled canvas: PADAssembler is not usable for this geometry
        # (its flat_indices/n_pixels disagree), but each CrystFEL panel carries
        # parent_data_slice — its slab in the canvas the frames arrive in.
        pads = DETECTOR_LOADERS["JUNGFRAU_4M"]()
        slices = [p.parent_data_slice for p in pads]
        h = max(s[0].stop for s in slices)
        w = max(s[1].stop for s in slices)
        mask = np.zeros((h, w), dtype=bool)
        for s in slices:
            mask[s] = True
    else:
        pads = get_geometry(detector_desc)
        assembler = get_assembler(detector_desc)
        n_pixels = int(sum(int(p.n_fs) * int(p.n_ss) for p in pads))
        coverage = assembler.assemble_data(np.ones(n_pixels, dtype=np.float32))
        mask = np.asarray(coverage) > 0.5
    if EDGE_EROSION_PX > 0:
        mask = binary_erosion(mask, iterations=EDGE_EROSION_PX)
    _MASK_CACHE[detector_desc] = mask
    return mask


def get_valid_mask_for_frame(
    detector_desc: str | None,
    frame_shape: tuple[int, ...],
) -> np.ndarray | None:
    """Return the valid-pixel mask for a frame, or None when unavailable.

    Wraps valid_pixel_mask() with the safety checks every call site needs:
    returns None (warning once per detector) if the description is missing,
    the geometry is unavailable, or the mask shape does not match the frame.
    Masks are never guessed from pixel values.

    Results are cached per (detector_desc, frame_shape) so two CXI files with
    the same detector description but different assembled canvas sizes each get
    their own validated entry rather than colliding on the shape-mismatch path.
    """
    cache_key: tuple[str | None, tuple[int, ...]] = (detector_desc, frame_shape)
    if cache_key in _FRAME_MASK_CACHE:
        return _FRAME_MASK_CACHE[cache_key]

    result: np.ndarray | None
    if detector_desc is None:
        result = None
    else:
        try:
            mask = valid_pixel_mask(detector_desc)
        except (ValueError, KeyError, OSError) as exc:
            if detector_desc not in _MASK_WARNED:
                _MASK_WARNED.add(detector_desc)
                warnings.warn(
                    f"get_valid_mask_for_frame: no valid-pixel mask for "
                    f"'{detector_desc}' ({exc}); mask-aware steps skipped.",
                    stacklevel=2,
                )
            result = None
        else:
            if mask.shape != frame_shape:
                if detector_desc not in _MASK_WARNED:
                    _MASK_WARNED.add(detector_desc)
                    warnings.warn(
                        f"get_valid_mask_for_frame: mask shape {mask.shape} != frame "
                        f"shape {frame_shape} for '{detector_desc}'; mask-aware steps skipped.",
                        stacklevel=2,
                    )
                result = None
            else:
                result = mask

    _FRAME_MASK_CACHE[cache_key] = result
    return result


def fill_gaps_after_gcn(
    gcn_frame: np.ndarray,
    detector_desc: str | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Set detector-gap/padding pixels of a GCN'd frame to 0 (the global mean).

    After GCN the global mean is 0 by construction, so filling invalid pixels
    with 0 removes the step-function transition at panel/gap boundaries that
    otherwise produces LCN halo/ringing artifacts — and matches the value used
    by pad_border. Modifies gcn_frame in place and returns it.

    If no mask is given it is derived from the detector geometry via
    get_valid_mask_for_frame(). The fill is skipped (frame returned unchanged,
    one warning per detector) when geometry is unavailable or the mask shape
    does not match the frame — never guessed from pixel values.
    """
    if mask is None:
        mask = get_valid_mask_for_frame(detector_desc, gcn_frame.shape)
        if mask is None:
            return gcn_frame
    if mask.shape != gcn_frame.shape:
        key = detector_desc or "<explicit mask>"
        if key not in _MASK_WARNED:
            _MASK_WARNED.add(key)
            warnings.warn(
                f"fill_gaps_after_gcn: mask shape {mask.shape} != frame shape "
                f"{gcn_frame.shape} for '{key}'; gap fill skipped.",
                stacklevel=2,
            )
        return gcn_frame
    gcn_frame[~mask] = 0.0
    return gcn_frame


def preprocess_eval_patches(
    assembled: np.ndarray,
    patch_size: int = TARGET_SIZE[0],
    stride: int | None = None,
    lcn_window: int = LCN_WINDOW_DEFAULT,
    detector_desc: str | None = None,
) -> np.ndarray:
    """GCN the full assembled frame, tile into patches, then LCN each patch.

    Used for all evaluation paths (validation, in-domain test, cross-detector
    test). GCN is applied once to the full native-resolution frame so all patches
    share the same global scale; then the frame is tiled into complete
    (patch_size × patch_size) patches and each patch is LCN-normalised.

    Args:
        assembled: float32 array (H, W) at native detector resolution.
        patch_size: Patch side length in pixels (default 224).
        stride: Step between patch origins (default = patch_size, non-overlapping).
        lcn_window: LCN neighbourhood size (default 9, Phase 3 ablation).
        detector_desc: CXI detector description for gap handling; when given,
            gap/padding/edge pixels are set to 0 after GCN (fill_gaps_after_gcn)
            and excluded from LCN local statistics (masked LCN), so windows
            straddling a panel boundary see only real pixels.

    Returns:
        float32 array of shape (N, patch_size, patch_size) where N ≥ 1.

    Raises:
        ValueError: If the image produces zero complete patches.
    """
    from src.preprocessing.augment import patch_grid

    mask = get_valid_mask_for_frame(detector_desc, assembled.shape)
    gcn_frame = gcn(assembled.astype(np.float32))
    gcn_frame = fill_gaps_after_gcn(gcn_frame, detector_desc, mask=mask)
    patches = patch_grid(gcn_frame, patch_size, stride)
    if not patches:
        raise ValueError(
            f"preprocess_eval_patches: no complete {patch_size}×{patch_size} "
            f"patch fits in image of shape {assembled.shape}."
        )
    if mask is not None:
        mask_patches = patch_grid(mask, patch_size, stride)
        normed = [
            lcn(p, window=lcn_window, mask=mp) for p, mp in zip(patches, mask_patches)
        ]
    else:
        normed = [lcn(p, window=lcn_window) for p in patches]
    return np.stack(normed, axis=0).astype(np.float32)
