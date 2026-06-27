"""Full preprocessing pipeline: geometry → GCN → LCN → resize to 224×224."""

from __future__ import annotations

import numpy as np
from skimage.transform import resize as sk_resize

from reborn.detector import PADAssembler, PADGeometryList

from src.preprocessing.geometry import assemble_image, extract_panels_from_canvas
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


def preprocess_assembled(
    image_2d: np.ndarray,
    lcn_window: int = LCN_WINDOW_DEFAULT,
) -> np.ndarray:
    """GCN → LCN → resize on a pre-assembled image, skipping geometry.

    Use for formats where the stored array is already a spatial image:
    Resonet Eiger/ePix10k CXI (5632×384 stacked panels), .img files, or any
    case where Reborn geometry assembly has already been applied externally.
    3D input (AGIPD modules, shape N×ss×fs) is row-stacked to 2D via _to_2d.

    Pipeline order is identical to preprocess() from step 3 onward, so both
    paths produce comparable outputs after normalization.

    Args:
        image_2d: float32 array, shape (H, W) or (N, ss, fs) for AGIPD modules.
        lcn_window: LCN neighbourhood size (default 9, from Phase 3 ablation).

    Returns:
        float32 array of shape (224, 224).
    """
    image_2d = _to_2d(image_2d)
    image_gcn = gcn(image_2d.astype(np.float32))
    image_lcn = lcn(image_gcn, window=lcn_window)
    resized = sk_resize(
        image_lcn,
        TARGET_SIZE,
        anti_aliasing=True,
        preserve_range=True,
    )
    return resized.astype(np.float32)


def preprocess(
    panel_data: list[np.ndarray],
    pads: object,
    lcn_window: int = LCN_WINDOW_DEFAULT,
) -> np.ndarray:
    """Full pipeline: assemble → GCN → LCN → resize to 224×224.

    This pipeline is shared and identical for both Track 1 (supervised) and
    Track 2 (SSL/MAE). Do not modify one track's preprocessing without
    applying the same change to both.

    Steps (order is non-negotiable):
        1. Reborn geometry assembly (multi-panel → single array)
        2. Flatten to 2D if needed (AGIPD 3D → 2D, Eiger4M 1D → 2D)
        3. GCN: global mean/std normalization
        4. LCN: local mean/std normalization
        5. Resize to TARGET_SIZE (224 × 224), anti-aliased

    Args:
        panel_data: List of 2D arrays (one per detector panel).
        pads: PADGeometryList from load_pad_geometry().
        lcn_window: LCN neighbourhood size (ablation parameter, default 9).

    Returns:
        float32 array of shape (224, 224).
    """
    assembled = assemble_image(pads, panel_data)

    pad_ss = pads[0].n_ss if len(pads) == 1 else None
    pad_fs = pads[0].n_fs if len(pads) == 1 else None
    image_2d = _to_2d(assembled, pad_ss=pad_ss, pad_fs=pad_fs)

    image_gcn = gcn(image_2d)
    image_lcn = lcn(image_gcn, window=lcn_window)

    resized = sk_resize(
        image_lcn,
        TARGET_SIZE,
        anti_aliasing=True,
        preserve_range=True,
    )
    return resized.astype(np.float32)


def preprocess_with_geometry(
    frame: np.ndarray,
    pads: PADGeometryList,
    detector_desc: str,
    lcn_window: int = LCN_WINDOW_DEFAULT,
    assembler: PADAssembler | None = None,
) -> np.ndarray:
    """Geometry-aware preprocessing: flatten pixels → PADAssembler → GCN → LCN → resize.

    Assembly strategy (confirmed by visual inspection 2026-06-26):
      - AGIPD 1M:     Reborn standard pads + PADAssembler(frame.ravel())
      - ePix10k 2.2M: Reborn standard pads + PADAssembler(frame.ravel())
      - EIGER 4M:     CrystFEL geom pads  + PADAssembler(concat panel ravels)

    Jungfrau 4M is pre-assembled — use preprocess_assembled() for that detector.

    Args:
        frame: Raw frame read from CXI file (float32 array, detector-native shape).
        pads: PADGeometryList loaded via get_geometry(detector_desc).
        detector_desc: Detector description string from CXI metadata
            (entry_1/instrument_1/detector_1/description).
        lcn_window: LCN neighbourhood size (default 9, from Phase 3 ablation).
        assembler: Optional pre-constructed PADAssembler (pass get_assembler() result
            to avoid recreating flat_indices on every call in tight loops).

    Returns:
        float32 array of shape (224, 224).

    Raises:
        ValueError: If detector_desc is unrecognised.
    """
    if detector_desc in ("AGIPD 1M", "ePix10k 2.2M"):
        # Reborn standard pads — flat pixel order matches raw array ravel.
        flat = frame.ravel().astype(np.float32)
    elif detector_desc == "EIGER 4M":
        # CrystFEL geom — extract panels via parent_data_slice, then flatten.
        panels = extract_panels_from_canvas(frame.astype(np.float32), pads)
        flat = np.concatenate([p.ravel() for p in panels])
    else:
        raise ValueError(
            f"preprocess_with_geometry: unrecognised detector_desc '{detector_desc}'. "
            "Use preprocess_assembled() for Jungfrau 4M."
        )

    if assembler is None:
        assembler = PADAssembler(pad_geometry=pads)
    assembled = assembler.assemble_data(flat)  # always 2D
    image_gcn = gcn(assembled.astype(np.float32))
    image_lcn = lcn(image_gcn, window=lcn_window)
    resized = sk_resize(
        image_lcn,
        TARGET_SIZE,
        anti_aliasing=True,
        preserve_range=True,
    )
    return resized.astype(np.float32)
