"""Reborn geometry handling — assembles multi-panel detector images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from reborn import detector
from reborn.external.crystfel import geometry_file_to_pad_geometry_list

_JUNGFRAU_4M_GEOM_JSON = Path(__file__).parent / "data" / "jungfrau4m_jf4m_103mm.json"
_EIGER_RESONET_GEOM = Path(__file__).parent / "data" / "eiger_resonet.geom"

# CrystFEL geom file for Eiger4M — only detector that needs it.
# AGIPD and ePix10k use Reborn's built-in standard loaders instead.
_EIGER4M_GEOM = Path(__file__).parent / "data" / "eiger4m.geom"

_GEOM_CACHE: dict[str, detector.PADGeometryList] = {}
_ASSEMBLER_CACHE: dict[str, detector.PADAssembler] = {}


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


def eiger4m_crystfel_pad_geometry_list(
    detector_distance: float | None = None,
) -> detector.PADGeometryList:
    """Load Eiger4M geometry from the CrystFEL .geom file.

    Our Eiger4M data is stored as a stacked LCLS canvas (5632×384), 64 panels of
    176×192 px each. Reborn's built-in eiger4M_pad_geometry_list() expects a
    different pixel count and cannot be used. This loader sets parent_data_slice
    on every panel so extract_panels_from_canvas() and PADAssembler work correctly.
    """
    pads = geometry_file_to_pad_geometry_list(str(_EIGER4M_GEOM))
    if detector_distance is not None:
        pads.set_average_detector_distance(detector_distance, beam_vec=[0, 0, 1])
    return pads


def eiger_resonet_pad_geometry_list() -> detector.PADGeometryList:
    """Load Eiger geometry from the Resonet CrystFEL .geom file.

    The raw canvas shape is (5632, 384): 64 panels of (176, 192) px each.
    parent_data_slice on each PADGeometry gives the ss/fs ranges to extract
    each panel from the raw frame — use extract_panels_from_canvas() before
    calling assemble_image().

    Source: /data/bioxfel/user/gihan/Resonet/geoms/Eigar.geom (copied to
    src/preprocessing/data/eiger_resonet.geom for reproducibility).
    """
    return geometry_file_to_pad_geometry_list(str(_EIGER_RESONET_GEOM))


_KNOWN_DESCS = {"AGIPD 1M", "ePix10k 2.2M", "EIGER 4M"}


def get_geometry(detector_desc: str) -> detector.PADGeometryList:
    """Load and cache PADGeometryList for the given CXI detector description.

    AGIPD 1M and ePix10k 2.2M use Reborn's built-in standard geometry loaders.
    EIGER 4M uses the CrystFEL .geom file in src/preprocessing/data/eiger4m.geom.

    Args:
        detector_desc: Value of entry_1/instrument_1/detector_1/description,
            e.g. 'AGIPD 1M', 'ePix10k 2.2M', 'EIGER 4M'.

    Returns:
        PADGeometryList, cached in _GEOM_CACHE so it is loaded only once per process.

    Raises:
        ValueError: If detector_desc is 'Jungfrau 4M' (pre-assembled — use
            preprocess_assembled() instead) or an unrecognised string.
    """
    if detector_desc == "Jungfrau 4M":
        raise ValueError(
            "Jungfrau 4M arrives pre-assembled. Use preprocess_assembled() directly."
        )
    if detector_desc not in _KNOWN_DESCS:
        raise ValueError(
            f"Unknown detector description '{detector_desc}'. "
            f"Known: {sorted(_KNOWN_DESCS)} (plus 'Jungfrau 4M' which is pre-assembled)."
        )
    if detector_desc not in _GEOM_CACHE:
        if detector_desc == "AGIPD 1M":
            _GEOM_CACHE[detector_desc] = detector.agipd_pad_geometry_list()
        elif detector_desc == "ePix10k 2.2M":
            _GEOM_CACHE[detector_desc] = detector.epix10k_pad_geometry_list()
        else:  # EIGER 4M — stacked LCLS canvas (5632×384), 64 panels via CrystFEL geom
            _GEOM_CACHE[detector_desc] = geometry_file_to_pad_geometry_list(
                str(_EIGER4M_GEOM)
            )
    return _GEOM_CACHE[detector_desc]


def get_assembler(detector_desc: str) -> detector.PADAssembler:
    """Return a cached PADAssembler for the given detector description.

    PADAssembler computes flat_indices once on construction; caching avoids
    recomputing them on every frame.
    """
    if detector_desc not in _ASSEMBLER_CACHE:
        _ASSEMBLER_CACHE[detector_desc] = detector.PADAssembler(
            pad_geometry=get_geometry(detector_desc)
        )
    return _ASSEMBLER_CACHE[detector_desc]


DETECTOR_LOADERS = {
    "AGIPD": detector.agipd_pad_geometry_list,
    "JUNGFRAU_4M": jungfrau4m_crystfel_pad_geometry_list,
    "ePix10k": detector.epix10k_pad_geometry_list,
    "Eiger4M": eiger4m_crystfel_pad_geometry_list,
    "EigerRESoNeT": eiger_resonet_pad_geometry_list,
}


def extract_panels_from_canvas(
    canvas: np.ndarray,
    pads: detector.PADGeometryList,
) -> list[np.ndarray]:
    """Extract per-panel 2D arrays from a raw detector canvas using geometry slices.

    Required for detectors whose CXI files store all panels in a single 2D
    canvas (e.g. EigerRESoNeT with shape (5632, 384), JUNGFRAU_4M with shape
    (2164, 2068)). Each pad's parent_data_slice gives the (ss_slice, fs_slice)
    index pair into the canvas.

    Args:
        canvas: 2D raw frame from the HDF5/CXI file, shape (H, W).
        pads: PADGeometryList loaded via load_pad_geometry() for a detector
            whose geometry defines parent_data_slice (i.e. pads.defines_slicing()
            returns True).

    Returns:
        List of 2D float32 arrays, one per panel, each shaped (n_ss, n_fs).

    Raises:
        ValueError: If the geometry does not define parent_data_slice.
    """
    if not pads.defines_slicing():
        raise ValueError(
            "Geometry does not define parent_data_slice. Use this function only "
            "for detectors loaded from CrystFEL .geom files (EigerRESoNeT, JUNGFRAU_4M)."
        )
    return [canvas[pad.parent_data_slice].astype(np.float32) for pad in pads]


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
