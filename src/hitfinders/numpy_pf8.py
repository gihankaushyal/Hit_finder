"""Pure Python/NumPy reimplementation of the PeakFinder8 algorithm.

Reference: Barty et al. (2014) Journal of Applied Crystallography.
Algorithm mirrors CrystFEL's implementation: for each pixel compute
local SNR via a box-filter approximation to the local background,
accept candidates above threshold and SNR, cluster with connected
components, filter by size, and return intensity-weighted centroids.

No CrystFEL or C dependencies. Worker-safe.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import label, uniform_filter


class NumpyPF8Hitfinder:
    """Pure NumPy/SciPy implementation of PeakFinder8.

    find_peaks() is called on the raw assembled float32 frame
    (before GCN/LCN), per the pipeline order:
        assembled → find_peaks → crop → rot90 → flip → GCN → LCN → cutout

    Args:
        threshold: Absolute intensity minimum (ADU). Default 800.
        min_snr: Minimum (I - bg_mean) / bg_std. Default 5.0.
        min_pix_count: Minimum pixels per peak cluster. Default 2.
        max_pix_count: Maximum pixels per peak cluster. Default 200.
        local_bg_radius: Half-width of background estimation box. Default 3.
        min_res: Min distance from frame centre (pixels). 0 = disabled.
        max_res: Max distance from frame centre (pixels). 0 = disabled.
    """

    def __init__(
        self,
        threshold: float = 800.0,
        min_snr: float = 5.0,
        min_pix_count: int = 2,
        max_pix_count: int = 200,
        local_bg_radius: int = 3,
        min_res: int = 0,
        max_res: int = 0,
    ) -> None:
        self._threshold = float(threshold)
        self._min_snr = float(min_snr)
        self._min_pix = int(min_pix_count)
        self._max_pix = int(max_pix_count)
        self._bg_radius = int(local_bg_radius)
        self._min_res = int(min_res)
        self._max_res = int(max_res)

    def find_peaks(self, assembled: np.ndarray) -> np.ndarray:
        """Locate Bragg peak centroids in a raw assembled detector frame.

        Args:
            assembled: 2D float32 array (H, W).

        Returns:
            float32 array of shape (N_peaks, 2): each row is [x, y]
            (x = column, y = row). Returns (0, 2) when no peaks found.
        """
        if assembled.ndim != 2:
            raise ValueError(f"assembled must be 2D, got shape {assembled.shape}")

        frame = assembled.astype(np.float32, copy=False)
        h, w = frame.shape
        box_out = 2 * self._bg_radius + 1
        # Inner exclusion box: the outermost 1-pixel border of the outer box
        # is used as background, matching CrystFEL PF8's ring-background model.
        # This ensures the peak pixel does not contaminate its own background.
        box_in = max(1, box_out - 2)

        n_out = box_out * box_out
        n_in = box_in * box_in
        n_ring = max(n_out - n_in, 1)

        sum_out = uniform_filter(frame, size=box_out, mode="reflect") * n_out
        sum_in = uniform_filter(frame, size=box_in, mode="reflect") * n_in
        bg_mean = (sum_out - sum_in) / n_ring

        sq = frame ** 2
        ssq_out = uniform_filter(sq, size=box_out, mode="reflect") * n_out
        ssq_in = uniform_filter(sq, size=box_in, mode="reflect") * n_in
        bg_sq_mean = (ssq_out - ssq_in) / n_ring

        bg_var = np.maximum(bg_sq_mean - bg_mean ** 2, 0.0)
        bg_std = np.sqrt(bg_var)

        above_threshold = frame > self._threshold
        snr = (frame - bg_mean) / (bg_std + 1e-6)
        candidates = above_threshold & (snr > self._min_snr)

        if self._min_res > 0 or self._max_res > 0:
            cy, cx = h / 2.0, w / 2.0
            ys, xs = np.ogrid[:h, :w]
            dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
            if self._min_res > 0:
                candidates &= dist >= self._min_res
            if self._max_res > 0:
                candidates &= dist <= self._max_res

        labeled, n_clusters = label(candidates)
        if n_clusters == 0:
            return np.zeros((0, 2), dtype=np.float32)

        peaks: list[list[float]] = []
        for cluster_id in range(1, n_clusters + 1):
            mask = labeled == cluster_id
            pix_count = int(mask.sum())
            if not (self._min_pix <= pix_count <= self._max_pix):
                continue
            intensity = frame[mask]
            row_coords, col_coords = np.where(mask)
            total = float(intensity.sum())
            if total <= 0.0:
                continue
            cx_peak = float((intensity * col_coords).sum() / total)
            cy_peak = float((intensity * row_coords).sum() / total)
            peaks.append([cx_peak, cy_peak])

        if not peaks:
            return np.zeros((0, 2), dtype=np.float32)
        return np.array(peaks, dtype=np.float32)
