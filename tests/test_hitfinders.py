# tests/test_hitfinders.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.hitfinders import MockHitfinder, get_hitfinder

_PF8_SO = Path("src/hitfinders/_pf8_wrap.so")


# ── MockHitfinder ─────────────────────────────────────────────────────────────

def test_mock_hitfinder_returns_correct_shape():
    hf = MockHitfinder(peaks=np.array([[100.0, 200.0], [300.0, 400.0]]))
    frame = np.zeros((512, 512), dtype=np.float32)
    result = hf.find_peaks(frame)
    assert result.shape == (2, 2)
    assert result.dtype == np.float32


def test_mock_hitfinder_empty_peaks():
    hf = MockHitfinder(peaks=np.zeros((0, 2), dtype=np.float32))
    result = hf.find_peaks(np.zeros((512, 512), dtype=np.float32))
    assert result.shape == (0, 2)


# ── Factory ───────────────────────────────────────────────────────────────────

def test_get_hitfinder_mock():
    cfg = {"hitfinder": {"backend": "mock"}}
    hf = get_hitfinder(cfg)
    assert isinstance(hf, MockHitfinder)


def test_get_hitfinder_unknown_backend_raises():
    cfg = {"hitfinder": {"backend": "unknown"}}
    with pytest.raises(ValueError, match="Unknown hitfinder backend"):
        get_hitfinder(cfg)


def test_get_hitfinder_missing_backend_raises():
    cfg = {"hitfinder": {}}
    with pytest.raises(ValueError, match="backend.*required"):
        get_hitfinder(cfg)


# ── Protocol ──────────────────────────────────────────────────────────────────

def test_hitfinder_protocol_isinstance():
    """runtime_checkable checks method names only — not signatures."""
    from src.hitfinders.base import Hitfinder
    hf = MockHitfinder()
    assert isinstance(hf, Hitfinder)  # structural check: has find_peaks method


# ── PF8Hitfinder (ctypes) ─────────────────────────────────────────────────────

def _make_sfx_frame(
    seed: int,
    shape: tuple[int, int] = (512, 512),
    bg_mean: float = 50.0,
    bg_std: float = 10.0,
) -> np.ndarray:
    """Synthetic SFX frame: Gaussian noise background (required by PF8 SNR stats)."""
    rng = np.random.default_rng(seed)
    frame = rng.normal(bg_mean, bg_std, shape).astype(np.float32)
    return np.clip(frame, 0.0, None)


@pytest.mark.skipif(
    not _PF8_SO.exists(),
    reason="_pf8_wrap.so not compiled; run 'cd src/hitfinders && make'",
)
def test_pf8_ctypes_finds_bright_spot():
    """PF8Hitfinder finds a 5×5 bright spot on a realistic Gaussian background."""
    from src.hitfinders.pf8 import PF8Hitfinder

    frame = _make_sfx_frame(0)
    frame[254:259, 254:259] = 2000.0
    hf = PF8Hitfinder(threshold=500.0, min_snr=3.0)
    peaks = hf.find_peaks(frame)
    assert peaks.shape[1] == 2
    assert peaks.dtype == np.float32
    assert len(peaks) >= 1
    cx, cy = peaks[0, 0], peaks[0, 1]
    assert abs(cx - 256.0) < 5.0
    assert abs(cy - 256.0) < 5.0


@pytest.mark.skipif(
    not _PF8_SO.exists(),
    reason="_pf8_wrap.so not compiled; run 'cd src/hitfinders && make'",
)
def test_pf8_ctypes_flat_frame_returns_no_peaks():
    """A uniform flat frame (no local peaks) should produce no hits."""
    from src.hitfinders.pf8 import PF8Hitfinder

    frame = np.full((512, 512), 100.0, dtype=np.float32)
    hf = PF8Hitfinder(threshold=500.0, min_snr=5.0)
    peaks = hf.find_peaks(frame)
    assert peaks.shape == (0, 2)
    assert peaks.dtype == np.float32


@pytest.mark.skipif(
    not _PF8_SO.exists(),
    reason="_pf8_wrap.so not compiled; run 'cd src/hitfinders && make'",
)
def test_pf8_ctypes_multiple_spots():
    """PF8Hitfinder finds multiple peaks on a Gaussian background."""
    from src.hitfinders.pf8 import PF8Hitfinder

    frame = _make_sfx_frame(1)
    frame[100:105, 100:105] = 2000.0
    frame[400:405, 400:405] = 2000.0
    hf = PF8Hitfinder(threshold=500.0, min_snr=3.0)
    peaks = hf.find_peaks(frame)
    assert len(peaks) >= 2


# ── NumpyPF8Hitfinder ─────────────────────────────────────────────────────────

def test_numpy_pf8_finds_bright_spot():
    from src.hitfinders.numpy_pf8 import NumpyPF8Hitfinder

    frame = _make_sfx_frame(2)
    # 5×5 spot (half-width 2). bg_radius=6 means the 1-pixel ring sits at
    # distance 6 from each candidate pixel, entirely outside the spot, giving
    # bg_mean≈50 and SNR>>3 for all 25 spot pixels.
    frame[254:259, 254:259] = 2000.0
    hf = NumpyPF8Hitfinder(threshold=500.0, min_snr=3.0, local_bg_radius=6)
    peaks = hf.find_peaks(frame)
    assert peaks.shape[1] == 2
    assert peaks.dtype == np.float32
    assert len(peaks) >= 1
    assert abs(peaks[0, 0] - 256.0) < 5.0
    assert abs(peaks[0, 1] - 256.0) < 5.0


def test_numpy_pf8_empty_frame_returns_no_peaks():
    from src.hitfinders.numpy_pf8 import NumpyPF8Hitfinder

    frame = np.zeros((512, 512), dtype=np.float32)
    hf = NumpyPF8Hitfinder(threshold=500.0, min_snr=3.0)
    peaks = hf.find_peaks(frame)
    assert peaks.shape == (0, 2)
    assert peaks.dtype == np.float32


def test_numpy_pf8_multiple_spots():
    from src.hitfinders.numpy_pf8 import NumpyPF8Hitfinder

    frame = _make_sfx_frame(3)
    frame[100:105, 100:105] = 2000.0
    frame[400:405, 400:405] = 2000.0
    hf = NumpyPF8Hitfinder(threshold=500.0, min_snr=3.0, local_bg_radius=6)
    peaks = hf.find_peaks(frame)
    assert len(peaks) >= 2


def test_numpy_pf8_size_filter_excludes_large_blob():
    from src.hitfinders.numpy_pf8 import NumpyPF8Hitfinder

    frame = _make_sfx_frame(4)
    frame[200:230, 200:230] = 2000.0  # 30×30 = 900 pixels > max_pix_count=200
    hf = NumpyPF8Hitfinder(threshold=500.0, min_snr=3.0, max_pix_count=200)
    peaks = hf.find_peaks(frame)
    assert len(peaks) == 0


def test_get_hitfinder_pf8_numpy_backend():
    from src.hitfinders import get_hitfinder
    from src.hitfinders.numpy_pf8 import NumpyPF8Hitfinder

    cfg = {"hitfinder": {"backend": "pf8_numpy"}}
    hf = get_hitfinder(cfg)
    assert isinstance(hf, NumpyPF8Hitfinder)


# ── GPUHitfinder ─────────────────────────────────────────────────────────────

def test_gpu_hitfinder_delegates_to_callable(tmp_path):
    import textwrap
    script = tmp_path / "my_gpu_hf.py"
    script.write_text(textwrap.dedent("""
        import numpy as np

        def find_peaks(frame: np.ndarray) -> np.ndarray:
            return np.array([[100.0, 200.0]], dtype=np.float32)
    """))

    from src.hitfinders.gpu import GPUHitfinder

    hf = GPUHitfinder(script_path=str(script), device="cpu")
    frame = np.zeros((512, 512), dtype=np.float32)
    peaks = hf.find_peaks(frame)
    assert peaks.shape == (1, 2)
    assert peaks.dtype == np.float32
    assert peaks[0, 0] == pytest.approx(100.0)
    assert peaks[0, 1] == pytest.approx(200.0)


def test_gpu_hitfinder_script_not_found_raises():
    from src.hitfinders.gpu import GPUHitfinder

    hf = GPUHitfinder(script_path="/nonexistent/path/gpu_hf.py", device="cpu")
    with pytest.raises(FileNotFoundError, match="gpu_hitfinder script not found"):
        hf.find_peaks(np.zeros((512, 512), dtype=np.float32))


def test_get_hitfinder_gpu_backend_missing_script_raises():
    from src.hitfinders import get_hitfinder

    cfg = {"hitfinder": {"backend": "gpu", "gpu_script_path": ""}}
    with pytest.raises(ValueError, match="gpu_script_path"):
        get_hitfinder(cfg)


def test_gpu_hitfinder_set_geometry_passes_kwargs(tmp_path):
    """set_geometry on GPUHitfinder calls set_geometry on the loaded module."""
    import textwrap
    script = tmp_path / "hf_with_geom.py"
    script.write_text(textwrap.dedent("""
        import numpy as np

        _last_geom = {}

        def set_geometry(**kwargs):
            _last_geom.update(kwargs)

        def find_peaks(frame):
            return np.zeros((0, 2), dtype=np.float32)
    """))

    from src.hitfinders.gpu import GPUHitfinder

    hf = GPUHitfinder(script_path=str(script), device="cpu")
    hf.set_geometry(dist=0.15, wavelength=1.3e-10)
    # Confirm find_peaks still works after set_geometry has already triggered _load()
    hf.find_peaks(np.zeros((512, 512), dtype=np.float32))
    assert hf._mod._last_geom == {"dist": 0.15, "wavelength": 1.3e-10}


def test_gpu_hitfinder_set_geometry_no_attr_does_not_raise(tmp_path):
    """set_geometry is a no-op when the script lacks set_geometry."""
    import textwrap
    script = tmp_path / "hf_no_geom.py"
    script.write_text(textwrap.dedent("""
        import numpy as np
        def find_peaks(frame):
            return np.zeros((0, 2), dtype=np.float32)
    """))

    from src.hitfinders.gpu import GPUHitfinder

    hf = GPUHitfinder(script_path=str(script), device="cpu")
    hf.set_geometry(dist=0.1)   # must not raise
    hf.find_peaks(np.zeros((512, 512), dtype=np.float32))  # must still work


# ── gpu_pf8 module (CI-safe: pyFAI imported lazily inside _rebuild) ───────────

def test_gpu_pf8_set_geometry_updates_state():
    """set_geometry changes module-level state and marks geometry as changed."""
    import src.hitfinders.gpu_pf8 as gpu_pf8

    # Force a known baseline
    gpu_pf8._dist = 0.2
    gpu_pf8._wavelength = 1.3e-10
    gpu_pf8._pixel_size = 75e-6
    gpu_pf8._geometry_changed = False

    gpu_pf8.set_geometry(dist=0.096, wavelength=1.305e-10, pixel_size=1e-4)

    assert gpu_pf8._dist == pytest.approx(0.096)
    assert gpu_pf8._wavelength == pytest.approx(1.305e-10)
    assert gpu_pf8._pixel_size == pytest.approx(1e-4)
    assert gpu_pf8._geometry_changed is True


def test_gpu_pf8_set_geometry_none_values_are_noop():
    """None arguments to set_geometry leave existing values unchanged."""
    import src.hitfinders.gpu_pf8 as gpu_pf8

    gpu_pf8._dist = 0.15
    gpu_pf8._geometry_changed = False

    gpu_pf8.set_geometry(dist=None, wavelength=None)

    assert gpu_pf8._dist == pytest.approx(0.15)
    assert gpu_pf8._geometry_changed is False


def test_gpu_pf8_find_peaks_returns_correct_shape():
    """find_peaks returns (N, 2) float32 when OCL_PeakFinder is mocked."""
    import sys
    import importlib
    import unittest.mock as mock

    fake_res = {"pos0": np.array([100.0, 200.0]), "pos1": np.array([150.0, 250.0])}
    mock_pf = mock.MagicMock()
    mock_pf.peakfinder8.return_value = fake_res

    with mock.patch.dict("sys.modules", {
        "pyFAI": mock.MagicMock(),
        "pyFAI.detector": mock.MagicMock(),
        "pyFAI.containers": mock.MagicMock(),
        "pyFAI.opencl": mock.MagicMock(),
        "pyFAI.opencl.peak_finder": mock.MagicMock(),
        "pyopencl": mock.MagicMock(),
    }):
        sys.modules.pop("src.hitfinders.gpu_pf8", None)
        import src.hitfinders.gpu_pf8 as gpu_pf8
        importlib.reload(gpu_pf8)

        # Inject pre-built mock objects — skip _rebuild
        gpu_pf8._pf = mock_pf
        gpu_pf8._polarization = mock.MagicMock()
        gpu_pf8._geometry_changed = False
        gpu_pf8._last_shape = (512, 512)

        frame = np.full((512, 512), 50.0, dtype=np.float32)
        peaks = gpu_pf8.find_peaks(frame)

    assert peaks.shape == (2, 2)
    assert peaks.dtype == np.float32
    # columns are [x, y] = [pos1, pos0]
    assert peaks[0, 0] == pytest.approx(150.0)  # x = pos1[0]
    assert peaks[0, 1] == pytest.approx(100.0)  # y = pos0[0]


def test_gpu_pf8_find_peaks_empty_returns_zero_shape():
    """find_peaks returns (0, 2) float32 when OCL_PeakFinder finds no peaks."""
    import sys
    import importlib
    import unittest.mock as mock

    fake_res = {"pos0": np.array([]), "pos1": np.array([])}
    mock_pf = mock.MagicMock()
    mock_pf.peakfinder8.return_value = fake_res

    with mock.patch.dict("sys.modules", {
        "pyFAI": mock.MagicMock(),
        "pyFAI.detector": mock.MagicMock(),
        "pyFAI.containers": mock.MagicMock(),
        "pyFAI.opencl": mock.MagicMock(),
        "pyFAI.opencl.peak_finder": mock.MagicMock(),
        "pyopencl": mock.MagicMock(),
    }):
        sys.modules.pop("src.hitfinders.gpu_pf8", None)
        import src.hitfinders.gpu_pf8 as gpu_pf8
        importlib.reload(gpu_pf8)

        gpu_pf8._pf = mock_pf
        gpu_pf8._polarization = mock.MagicMock()
        gpu_pf8._geometry_changed = False
        gpu_pf8._last_shape = (512, 512)

        frame = np.zeros((512, 512), dtype=np.float32)
        peaks = gpu_pf8.find_peaks(frame)

    assert peaks.shape == (0, 2)
    assert peaks.dtype == np.float32
