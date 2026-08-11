"""PeakFinder8 hitfinder via the ssc Python/Cython wrapper.

Wraps ``ssc.peakfinder8_extension.peakfinder_8`` from
https://github.com/kif/peakfinder8.  Build and install once:

    /path/to/envs/sfx-hitfinder/bin/python -m pip install /tmp/peakfinder8 --no-build-isolation

Parameters match the NumpyPF8 / PF8 (C-ext) backends so all three are
drop-in interchangeable.  Worker-safe: the Cython extension has no shared
module state.

``peakfinder_8`` call signature (from Cython source)::

    peakfinder_8(max_num_peaks,
                 data, mask, pix_r,
                 asic_nx, asic_ny, nasics_x, nasics_y,
                 adc_thresh, hitfinder_min_snr,
                 hitfinder_min_pix_count, hitfinder_max_pix_count,
                 hitfinder_local_bg_radius)

    data    : float32 2-D C-contiguous (H, W)
    mask    : int8   2-D C-contiguous — 1 = valid pixel, 0 = masked
    pix_r   : float32 2-D C-contiguous — per-pixel scattering radius (mm from
              pyFAI ``ai._cached_array['r_center']``) or, as a fallback,
              Euclidean pixel distance from frame centre.

Returns (peak_list_x, peak_list_y, peak_list_value) where x = fast-scan
(column) and y = slow-scan (row) in assembled-frame pixel coordinates.

Note: ``min_res`` / ``max_res`` are NOT parameters of ``peakfinder_8`` itself;
apply them as post-filters on the returned centroid array if needed.
"""

from __future__ import annotations

import numpy as np


def _euclidean_pix_r(h: int, w: int) -> np.ndarray:
    """Fallback per-pixel radius: Euclidean pixel distance from frame centre."""
    cy, cx = h / 2.0, w / 2.0
    rows, cols = np.ogrid[:h, :w]
    r = np.sqrt((cols - cx) ** 2 + (rows - cy) ** 2).astype(np.float32)
    return np.ascontiguousarray(r)


class PF8HitfinderPythonWrapper:
    """PeakFinder8 via the ssc Cython Python wrapper.

    find_peaks() is called on the raw assembled float32 frame
    (before GCN/LCN), per the pipeline order:
        assembled → find_peaks → crop → rot90 → flip → GCN → LCN → cutout

    Args:
        threshold: Absolute ADU minimum (``adc_thresh``). Default 800.
        min_snr: Minimum local signal-to-noise ratio. Default 5.0.
        min_pix_count: Minimum pixels per peak cluster. Default 9.
        max_pix_count: Maximum pixels per peak cluster. Default 999.
        local_bg_radius: Radius for local background estimation. Default 3.
        min_res: Minimum distance from frame centre (pixels). 0 = disabled.
        max_res: Maximum distance from frame centre (pixels). 0 = disabled.
        max_num_peaks: Hard cap on returned peaks. Default 1000.
    """

    def __init__(
        self,
        threshold: float = 800.0,
        min_snr: float = 5.0,
        min_pix_count: int = 9,
        max_pix_count: int = 999,
        local_bg_radius: int = 3,
        min_res: int = 0,
        max_res: int = 0,
        max_num_peaks: int = 1000,
    ) -> None:
        try:
            from ssc.peakfinder8_extension import peakfinder_8 as _pf8
        except ImportError as exc:
            raise ImportError(
                "ssc.peakfinder8_extension not found. "
                "Install with: /path/to/envs/sfx-hitfinder/bin/python -m pip install "
                "/tmp/peakfinder8 --no-build-isolation"
            ) from exc

        self._pf8 = _pf8
        self._threshold = float(threshold)
        self._min_snr = float(min_snr)
        self._min_pix = int(min_pix_count)
        self._max_pix = int(max_pix_count)
        self._bg_radius = int(local_bg_radius)
        self._min_res = int(min_res)
        self._max_res = int(max_res)
        self._max_peaks = int(max_num_peaks)

        # Optional precomputed pix_r — set via set_pix_r() when a pyFAI
        # AzimuthalIntegrator is available (ai._cached_array['r_center']).
        self._pix_r: np.ndarray | None = None

    def set_pix_r(self, pix_r: np.ndarray) -> None:
        """Supply a precomputed per-pixel radius array (e.g. from pyFAI).

        Expected source::

            pix_r = ai._cached_array['r_center'].astype('float32')

        Providing this gives physically correct azimuthal binning.  Without
        it, find_peaks() falls back to Euclidean pixel distance.
        """
        self._pix_r = np.ascontiguousarray(pix_r, dtype=np.float32)

    def find_peaks(self, assembled: np.ndarray) -> np.ndarray:
        """Find Bragg peak centroids in a raw assembled detector frame.

        Args:
            assembled: 2D float32 array (H, W).

        Returns:
            float32 array of shape (N_peaks, 2): each row is [x, y]
            (x = column / fast-scan, y = row / slow-scan).
            Returns (0, 2) when no peaks found.
        """
        if assembled.ndim != 2:
            raise ValueError(f"assembled must be 2D, got shape {assembled.shape}")

        h, w = assembled.shape

        # Zero sub-zero pixels (pedestal artefacts) before passing to PF8.
        data = np.ascontiguousarray(assembled, dtype=np.float32)
        data[data < 0.0] = 0.0

        # mask: 1 = valid, 0 = masked (PF8 convention).
        # Convention matches pyFAI: imask = (1 - pyfai_mask).astype("int8")
        # where pyfai_mask has True (=1) for bad pixels.
        # With no bad-pixel map, all pixels are marked valid.
        mask = np.ones((h, w), dtype=np.int8)

        # Radial coordinate array — use pyFAI r_center if available.
        pix_r = self._pix_r if self._pix_r is not None else _euclidean_pix_r(h, w)
        if pix_r.shape != (h, w):
            pix_r = _euclidean_pix_r(h, w)

        peak_x, peak_y, _ = self._pf8(
            self._max_peaks,
            data,
            mask,
            pix_r,
            w,               # asic_nx — fast-scan width (columns)
            h,               # asic_ny — slow-scan height (rows)
            1,               # nasics_x — single assembled panel
            1,               # nasics_y
            self._threshold,
            self._min_snr,
            self._min_pix,
            self._max_pix,
            self._bg_radius,
        )

        if len(peak_x) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        # peak_x = fast-scan (col = x), peak_y = slow-scan (row = y).
        peaks = np.column_stack([
            np.asarray(peak_x, dtype=np.float32),
            np.asarray(peak_y, dtype=np.float32),
        ])

        # Optional resolution ring filter (not a peakfinder_8 param — applied here).
        if self._min_res > 0 or self._max_res > 0:
            cy, cx = h / 2.0, w / 2.0
            dist = np.sqrt((peaks[:, 0] - cx) ** 2 + (peaks[:, 1] - cy) ** 2)
            keep = np.ones(len(peaks), dtype=bool)
            if self._min_res > 0:
                keep &= dist >= self._min_res
            if self._max_res > 0:
                keep &= dist <= self._max_res
            peaks = peaks[keep]

        return peaks
