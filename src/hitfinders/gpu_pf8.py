"""pyFAI OCL_PeakFinder GPU hitfinder script.

Consumed by GPUHitfinder via dynamic import.  Exposes two module-level functions:

    set_geometry(**kwargs)      -- call once per CXI file open (geometry update)
    find_peaks(frame)           -- call per frame; returns (N_peaks, 2) [x, y]

pyFAI and pyopencl are imported lazily inside _rebuild() so this module is
importable in CI without GPU hardware.

All tuning knobs are overridable via environment variables set before import.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Tuning parameters — all overridable via env vars set before import
# ---------------------------------------------------------------------------
PIXEL_SIZE: float = float(os.environ.get("HITFINDER_PIXEL_SIZE", 75e-6))
DIST: float = float(os.environ.get("HITFINDER_DIST", 0.2))
WAVELENGTH: float = float(os.environ.get("HITFINDER_WAVELENGTH", 1.3e-10))
NPT: int = int(os.environ.get("HITFINDER_NPT", 1000))
POL_FACTOR: float = float(os.environ.get("HITFINDER_POL_FACTOR", 0.99))
CYCLE: int = int(os.environ.get("HITFINDER_CYCLE", 5))
CUTOFF_PICK: float = float(os.environ.get("HITFINDER_CUTOFF_PICK", 3.0))
CUTOFF_PEAK: float = float(os.environ.get("HITFINDER_CUTOFF_PEAK", 3.0))
NOISE: float = float(os.environ.get("HITFINDER_NOISE", 1.0))
CONNECTED: int = int(os.environ.get("HITFINDER_CONNECTED", 3))
PATCH_SIZE: int = int(os.environ.get("HITFINDER_PATCH_SIZE", 3))

# ---------------------------------------------------------------------------
# Module-level state — rebuilt lazily when geometry or frame shape changes
# ---------------------------------------------------------------------------
_dist: float = DIST
_wavelength: float = WAVELENGTH
_pixel_size: float = PIXEL_SIZE
_poni1: float | None = None   # None → computed from frame shape at _rebuild time
_poni2: float | None = None

_geometry_changed: bool = True
_last_shape: tuple[int, int] | None = None

_ctx: Any = None        # pyopencl.Context — created once
_pf: Any = None         # OCL_PeakFinder instance
_polarization: Any = None   # pyFAI polarization container (.array, .checksum)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def set_geometry(
    dist: float | None = None,
    wavelength: float | None = None,
    pixel_size: float | None = None,
    poni1: float | None = None,
    poni2: float | None = None,
) -> None:
    """Update detector geometry.

    None values leave the current setting unchanged.
    Only flags a rebuild when at least one value actually changes.
    """
    global _dist, _wavelength, _pixel_size, _poni1, _poni2, _geometry_changed
    changed = False
    if dist is not None and dist != _dist:
        _dist = dist
        changed = True
    if wavelength is not None and wavelength != _wavelength:
        _wavelength = wavelength
        changed = True
    if pixel_size is not None and pixel_size != _pixel_size:
        _pixel_size = pixel_size
        changed = True
    if poni1 is not None and poni1 != _poni1:
        _poni1 = poni1
        changed = True
    if poni2 is not None and poni2 != _poni2:
        _poni2 = poni2
        changed = True
    if changed:
        _geometry_changed = True


def find_peaks(frame: np.ndarray) -> np.ndarray:
    """Find Bragg peaks in a raw assembled frame.

    Args:
        frame: 2D float32 array (H, W), raw assembled image before GCN/LCN.

    Returns:
        float32 array of shape (N_peaks, 2), columns [x, y] in pixel coords.
        Returns shape (0, 2) when no peaks are found.
    """
    frame = np.asarray(frame, dtype=np.float32)

    # Zero out sub-zero pixels (detector artefacts / pedestal drift).
    frame_clean = frame.copy()
    frame_clean[frame < 0.0] = 0.0

    if _geometry_changed or _pf is None or frame.shape != _last_shape:
        _rebuild(frame.shape)

    from pyFAI.containers import ErrorModel  # lazy — only reached after _rebuild has run

    res = _pf.peakfinder8(
        data=frame_clean,
        error_model=ErrorModel.parse("azimuthal"),
        polarization=_polarization.array,
        polarization_checksum=_polarization.checksum,
        cycle=CYCLE,
        cutoff_pick=CUTOFF_PICK,
        cutoff_peak=CUTOFF_PEAK,
        noise=NOISE,
        connected=CONNECTED,
        patch_size=PATCH_SIZE,
    )

    if len(res["pos0"]) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # pos0 = row (y), pos1 = col (x) — return as [x, y] to match interface contract.
    return np.column_stack([res["pos1"], res["pos0"]]).astype(np.float32)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _rebuild(frame_shape: tuple[int, int]) -> None:
    """Reconstruct AzimuthalIntegrator and OCL_PeakFinder."""
    global _ctx, _pf, _polarization, _geometry_changed, _last_shape

    import pyFAI
    import pyFAI.detector
    from pyFAI import units
    from pyFAI.opencl.peak_finder import OCL_PeakFinder
    import pyopencl

    nrows, ncols = frame_shape
    poni1 = _poni1 if _poni1 is not None else (nrows / 2.0) * _pixel_size
    poni2 = _poni2 if _poni2 is not None else (ncols / 2.0) * _pixel_size

    # Static mask: zeros (no bad-pixel file; overflow handled by zeroing frame_clean).
    static_mask = np.zeros(frame_shape, dtype=bool)

    det = pyFAI.detector.Detector(pixel1=_pixel_size, pixel2=_pixel_size)
    det.mask = static_mask

    ai = pyFAI.AzimuthalIntegrator(
        dist=_dist,
        poni1=poni1,
        poni2=poni2,
        wavelength=_wavelength,
    )
    ai.detector = det

    if _ctx is None:
        _ctx = pyopencl.create_some_context(interactive=False)

    unit = units.to_unit("r_mm")
    integrator = ai.setup_sparse_integrator(
        frame_shape, NPT, mask=static_mask,
        unit=unit, split="no", algo="CSR", scale=False,
    )

    ai.polarization(factor=POL_FACTOR, shape=frame_shape)
    polarization = ai._cached_array.get("last_polarization")

    _pf = OCL_PeakFinder(
        integrator.lut,
        image_size=nrows * ncols,
        bin_centers=integrator.bin_centers,
        radius=ai._cached_array[unit.name.split("_")[0] + "_center"],
        mask=static_mask,
        ctx=_ctx,
        unit=unit,
    )
    _polarization = polarization
    _last_shape = frame_shape
    _geometry_changed = False
