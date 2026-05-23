"""Reborn geometry handling — assembles multi-panel detector images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from reborn import detector

_JUNGFRAU_4M_GEOM_JSON = Path(__file__).parent / "data" / "jungfrau4m_jf4m_103mm.json"


def jungfrau4m_crystfel_pad_geometry_list(
    detector_distance: float = 0.103,
) -> detector.PADGeometryList:
    """Load JUNGFRAU 4M geometry derived from jf4m_103mm_20260408.geom.

    parent_data_shape=[2164, 2068] and parent_data_slice values match the
    pre-assembled canvas stored in Jungfrau.h5 so that panels can be extracted
    manually before passing to PADAssembler.
    """
    pads = detector.load_pad_geometry_list(str(_JUNGFRAU_4M_GEOM_JSON))
    pads.set_average_detector_distance(detector_distance, beam_vec=[0, 0, 1])
    return pads


DETECTOR_LOADERS = {
    "AGIPD": detector.agipd_pad_geometry_list,
    "JUNGFRAU_4M": jungfrau4m_crystfel_pad_geometry_list,
    "ePix10k": detector.epix10k_pad_geometry_list,
    "Eiger4M": detector.eiger4M_pad_geometry_list,
}


def load_pad_geometry(
    detector_type: str,
    detector_distance: float | None = None,
) -> detector.PADGeometryList:
    """Return PADGeometryList for the given detector type.

    Args:
        detector_type: One of "AGIPD", "JUNGFRAU_4M", "ePix10k", "Eiger4M".
        detector_distance: Sample-to-detector distance in metres. When None
            (default) each detector loader uses its own natural default
            (0.103 m for JUNGFRAU_4M, 0.1 m for the others).

    Raises:
        ValueError: If detector_type is not one of the four supported values.
    """
    if detector_type not in DETECTOR_LOADERS:
        raise ValueError(
            f"Unknown detector type '{detector_type}'. "
            f"Supported: {sorted(DETECTOR_LOADERS)}"
        )
    loader = DETECTOR_LOADERS[detector_type]
    if detector_distance is None:
        return loader()
    return loader(detector_distance=detector_distance)


def assemble_image(
    pads: detector.PADGeometryList,
    panel_data: list[np.ndarray],
) -> np.ndarray:
    """Assemble per-panel arrays into a lab-frame image.

    For geometries that define parent canvas slicing (e.g. JUNGFRAU_4M),
    PADAssembler places each panel at its real-space position, producing a
    2D grid that may be larger than n_pixels (gap pixels are zero-filled).
    For other geometries the classic concat_data → reshape path is used:
    - AGIPD:   3D (n_modules, module_ss, module_fs)
    - ePix10k: 2D (n_ss, n_fs)
    - Eiger4M: 1D flat (monolithic)

    Args:
        pads: PADGeometryList for the detector.
        panel_data: List of 2D arrays, one per panel (shape n_ss × n_fs each).

    Returns:
        numpy array — shape depends on detector (see above).
    """
    if pads.defines_slicing():
        canvas_size = int(np.prod(pads[0].parent_data_shape))
        if canvas_size != pads.n_pixels:
            # Canvas contains gap pixels — PADAssembler places panels at their
            # real-space positions; the classic concat→reshape path would fail
            # because the canvas ravel (canvas_size) != n_pixels.
            assembler = detector.PADAssembler(pad_geometry=pads)
            return assembler.assemble_data(panel_data)
    flat = pads.concat_data(panel_data)
    return pads.reshape(flat)
