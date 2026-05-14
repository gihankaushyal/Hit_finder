"""Reborn geometry handling — assembles multi-panel detector images."""

from __future__ import annotations

import numpy as np
from reborn import detector

DETECTOR_LOADERS = {
    "AGIPD": detector.agipd_pad_geometry_list,
    "JUNGFRAU_4M": detector.jungfrau4m_pad_geometry_list,
    "ePix10k": detector.epix10k_pad_geometry_list,
    "Eiger4M": detector.eiger4M_pad_geometry_list,
}


def load_pad_geometry(
    detector_type: str,
    detector_distance: float = 0.1,
) -> detector.PADGeometryList:
    """Return PADGeometryList for the given detector type.

    Args:
        detector_type: One of "AGIPD", "JUNGFRAU_4M", "ePix10k", "Eiger4M".
        detector_distance: Sample-to-detector distance in metres.

    Raises:
        ValueError: If detector_type is not one of the four supported values.
    """
    if detector_type not in DETECTOR_LOADERS:
        raise ValueError(
            f"Unknown detector type '{detector_type}'. "
            f"Supported: {sorted(DETECTOR_LOADERS)}"
        )
    return DETECTOR_LOADERS[detector_type](detector_distance=detector_distance)


def assemble_image(
    pads: detector.PADGeometryList,
    panel_data: list[np.ndarray],
) -> np.ndarray:
    """Assemble per-panel arrays into a lab-frame image via Reborn reshape.

    The returned array shape depends on detector geometry:
    - JUNGFRAU_4M, ePix10k: 2D (n_ss, n_fs)
    - AGIPD: 3D (n_modules, module_ss, module_fs) — modules are not tiled
    - Eiger4M: 1D flat (monolithic, no multi-panel assembly needed)

    Phase 3 will add normalization and resize steps that bring all detectors
    to a uniform 2D 224×224 representation.

    Args:
        pads: PADGeometryList for the detector.
        panel_data: List of 2D arrays, one per panel (shape n_ss × n_fs each).

    Returns:
        numpy array — shape depends on detector (see above).
    """
    flat = pads.concat_data(panel_data)
    return pads.reshape(flat)
