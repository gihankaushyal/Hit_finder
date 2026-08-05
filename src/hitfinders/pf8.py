"""CrystFEL PeakFinder8 hitfinder via a thin C bridge.

Requires _pf8_wrap.so compiled from src/hitfinders/_pf8_wrap.c:
    cd src/hitfinders && make

Worker safety: the C bridge has no shared state — safe across
forked DataLoader worker processes.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

_SO_DEFAULT = Path(__file__).parent / "_pf8_wrap.so"
_MAX_N_PEAKS = 2048  # CrystFEL PF8 hard limit per panel


def _load_lib(so_path: Path) -> ctypes.CDLL:
    if not so_path.exists():
        raise FileNotFoundError(
            f"_pf8_wrap.so not found at {so_path}. "
            "Compile it first: cd src/hitfinders && make"
        )
    lib = ctypes.CDLL(str(so_path))
    lib.pf8_find_peaks.restype = ctypes.c_int
    lib.pf8_find_peaks.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # data
        ctypes.c_int,  # w
        ctypes.c_int,  # h
        ctypes.c_float,  # threshold
        ctypes.c_float,  # min_snr
        ctypes.c_int,  # min_pix_count
        ctypes.c_int,  # max_pix_count
        ctypes.c_int,  # local_bg_radius
        ctypes.c_int,  # min_res
        ctypes.c_int,  # max_res
        ctypes.c_int,  # use_saturated
        ctypes.c_int,  # max_n_peaks
        ctypes.POINTER(ctypes.c_float),  # out_x
        ctypes.POINTER(ctypes.c_float),  # out_y
        ctypes.POINTER(ctypes.c_int),  # out_count
    ]
    return lib


class PF8Hitfinder:
    """CrystFEL PeakFinder8 via ctypes bridge to libcrystfel.so.

    find_peaks() is called on the raw assembled float32 frame
    (before GCN/LCN), per the pipeline order:
        assembled → find_peaks → crop → rot90 → flip → GCN → LCN → cutout

    Args:
        threshold: Absolute intensity threshold in ADU. Default 800.
        min_snr: Minimum local signal-to-noise ratio. Default 5.0.
        min_pix_count: Minimum pixels per connected peak. Default 2.
        max_pix_count: Maximum pixels per connected peak. Default 200.
        local_bg_radius: Radius for local background estimation. Default 3.
        min_res: Minimum distance from frame centre (pixels). 0 = disabled.
        max_res: Maximum distance from frame centre (pixels). 0 = disabled.
        use_saturated: Include saturated pixels. Default False.
        so_path: Path to compiled _pf8_wrap.so. Defaults to same directory.
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
        use_saturated: bool = False,
        so_path: Path | str | None = None,
    ) -> None:
        self._threshold = float(threshold)
        self._min_snr = float(min_snr)
        self._min_pix_count = int(min_pix_count)
        self._max_pix_count = int(max_pix_count)
        self._local_bg_radius = int(local_bg_radius)
        self._min_res = int(min_res)
        self._max_res = int(max_res)
        self._use_saturated = int(use_saturated)
        self._lib = _load_lib(Path(so_path) if so_path is not None else _SO_DEFAULT)

        # Pre-allocate output buffers (reused per call; each DataLoader worker
        # gets its own process copy after fork — no cross-worker aliasing).
        self._out_x = (ctypes.c_float * _MAX_N_PEAKS)()
        self._out_y = (ctypes.c_float * _MAX_N_PEAKS)()
        self._out_count = ctypes.c_int(0)

    def find_peaks(self, assembled: np.ndarray) -> np.ndarray:
        """Find Bragg peak centroids in a raw assembled detector frame.

        Args:
            assembled: 2D float32 array (H, W) — raw assembled image,
                BEFORE GCN/LCN normalization.

        Returns:
            float32 array of shape (N_peaks, 2): each row is [x, y]
            in assembled-frame pixel coordinates (x=column, y=row).
            Returns shape (0, 2) when no peaks found.
        """
        if assembled.ndim != 2:
            raise ValueError(f"assembled must be 2D, got shape {assembled.shape}")
        frame = np.ascontiguousarray(assembled, dtype=np.float32)
        h, w = frame.shape
        data_ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self._out_count.value = 0
        rc = self._lib.pf8_find_peaks(
            data_ptr,
            w,
            h,
            self._threshold,
            self._min_snr,
            self._min_pix_count,
            self._max_pix_count,
            self._local_bg_radius,
            self._min_res,
            self._max_res,
            self._use_saturated,
            _MAX_N_PEAKS,
            self._out_x,
            self._out_y,
            ctypes.byref(self._out_count),
        )
        if rc != 0:
            return np.zeros((0, 2), dtype=np.float32)

        n = self._out_count.value
        if n == 0:
            return np.zeros((0, 2), dtype=np.float32)

        peaks = np.empty((n, 2), dtype=np.float32)
        peaks[:, 0] = np.frombuffer(self._out_x, dtype=np.float32, count=n)
        peaks[:, 1] = np.frombuffer(self._out_y, dtype=np.float32, count=n)
        return peaks
