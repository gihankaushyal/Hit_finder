# GPU Hitfinder Full Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `src/hitfinders/gpu_pf8.py` (pyFAI OCL_PeakFinder backend), extend `GPUHitfinder` with `set_geometry`, fix the hitfinder/GCN order bug in `AsymmetricCXIDataset`, and document confirmed CXI geometry keys.

**Architecture:** `gpu_pf8.py` is a standalone script dynamically imported by `GPUHitfinder`. It exposes two module-level functions: `set_geometry(**kwargs)` (called once per CXI file open) and `find_peaks(frame)` (called per frame). pyFAI/pyopencl are imported lazily inside `_rebuild()` so the module is importable in CI without GPU hardware. `AsymmetricCXIDataset` caches geometry per unique CXI path in `__init__` and calls `set_geometry` + runs `find_peaks` before `gcn()`.

**Tech Stack:** pyFAI (`OCL_PeakFinder`, `AzimuthalIntegrator`), pyopencl, numpy, h5py, pytest, unittest.mock

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/hitfinders/gpu_pf8.py` | **Create** | OCL_PeakFinder backend script |
| `src/hitfinders/gpu.py` | **Modify** | Add `self._mod` + `set_geometry` pass-through |
| `src/data/dataset.py` | **Modify** | Geometry cache in `__init__`; order fix + `set_geometry` call in `__getitem__` |
| `docs/data_spec.md` | **Modify** | Add confirmed geometry HDF5 key table |
| `tests/test_hitfinders.py` | **Modify** | Add `GPUHitfinder.set_geometry` tests |
| `tests/test_asymmetric_dataset.py` | **Modify** | Add order test + `set_geometry` call test |

---

## Task 1: Extend `GPUHitfinder` with `self._mod` and `set_geometry`

**Files:**
- Modify: `src/hitfinders/gpu.py`
- Modify: `tests/test_hitfinders.py`

- [ ] **Step 1.1 — Write failing tests**

Add after the existing `GPUHitfinder` tests at the bottom of `tests/test_hitfinders.py`:

```python
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
    # Trigger load so module is accessible
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
```

- [ ] **Step 1.2 — Run tests to confirm they fail**

```bash
pytest tests/test_hitfinders.py::test_gpu_hitfinder_set_geometry_passes_kwargs \
       tests/test_hitfinders.py::test_gpu_hitfinder_set_geometry_no_attr_does_not_raise -v
```

Expected: `FAILED` — `GPUHitfinder` has no `set_geometry` method and no `_mod` attribute.

- [ ] **Step 1.3 — Implement the changes in `gpu.py`**

In `src/hitfinders/gpu.py`:

1. Add `_mod` field to `__init__`:
```python
def __init__(self, script_path: str | Path, device: str = "cuda") -> None:
    self._script_path = Path(script_path)
    self._device = device
    self._fn: Callable[[np.ndarray], np.ndarray] | None = None
    self._mod: object | None = None   # add this line
```

2. Store `mod` in `_load()` (add one line after `spec.loader.exec_module(mod)`):
```python
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        if not hasattr(mod, "find_peaks"):
            raise AttributeError(
                f"GPU hitfinder script at {self._script_path} must expose "
                "a top-level 'find_peaks(frame: np.ndarray) -> np.ndarray' function."
            )
        self._mod = mod               # add this line
        self._fn = mod.find_peaks
```

3. Add `set_geometry` method after `_load`:
```python
    def set_geometry(self, **kwargs) -> None:
        """Forward geometry parameters to the script's set_geometry function if present."""
        if self._fn is None:
            self._load()
        if hasattr(self._mod, "set_geometry"):
            self._mod.set_geometry(**kwargs)  # type: ignore[union-attr]
```

- [ ] **Step 1.4 — Run tests to confirm they pass**

```bash
pytest tests/test_hitfinders.py::test_gpu_hitfinder_set_geometry_passes_kwargs \
       tests/test_hitfinders.py::test_gpu_hitfinder_set_geometry_no_attr_does_not_raise -v
```

Expected: `PASSED`

- [ ] **Step 1.5 — Run full hitfinder test suite to check no regressions**

```bash
pytest tests/test_hitfinders.py -v
```

Expected: all tests pass (existing + 2 new).

- [ ] **Step 1.6 — Commit**

```bash
git add src/hitfinders/gpu.py tests/test_hitfinders.py
git commit -m "feat: add set_geometry pass-through to GPUHitfinder"
```

---

## Task 2: Create `src/hitfinders/gpu_pf8.py`

**Files:**
- Create: `src/hitfinders/gpu_pf8.py`
- Modify: `tests/test_hitfinders.py`

- [ ] **Step 2.1 — Write failing tests for `set_geometry` state**

`gpu_pf8.py` uses lazy pyFAI/pyopencl imports (only inside `_rebuild`), so the module is importable in CI without GPU hardware. These tests exploit that.

Add after the `set_geometry` tests in `tests/test_hitfinders.py`:

```python
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
```

- [ ] **Step 2.2 — Run tests to confirm they fail**

```bash
pytest tests/test_hitfinders.py::test_gpu_pf8_set_geometry_updates_state \
       tests/test_hitfinders.py::test_gpu_pf8_set_geometry_none_values_are_noop \
       tests/test_hitfinders.py::test_gpu_pf8_find_peaks_returns_correct_shape \
       tests/test_hitfinders.py::test_gpu_pf8_find_peaks_empty_returns_zero_shape -v
```

Expected: `ERROR` / `ModuleNotFoundError` — `src/hitfinders/gpu_pf8.py` does not exist yet.

- [ ] **Step 2.3 — Create `src/hitfinders/gpu_pf8.py`**

```python
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
```

- [ ] **Step 2.4 — Run tests to confirm they pass**

```bash
pytest tests/test_hitfinders.py::test_gpu_pf8_set_geometry_updates_state \
       tests/test_hitfinders.py::test_gpu_pf8_set_geometry_none_values_are_noop \
       tests/test_hitfinders.py::test_gpu_pf8_find_peaks_returns_correct_shape \
       tests/test_hitfinders.py::test_gpu_pf8_find_peaks_empty_returns_zero_shape -v
```

Expected: `PASSED`

- [ ] **Step 2.5 — Run full hitfinder test suite**

```bash
pytest tests/test_hitfinders.py -v
```

Expected: all pass.

- [ ] **Step 2.6 — Commit**

```bash
git add src/hitfinders/gpu_pf8.py tests/test_hitfinders.py
git commit -m "feat: implement gpu_pf8.py OCL_PeakFinder backend"
```

---

## Task 3: Fix `AsymmetricCXIDataset` — order + geometry call

**Files:**
- Modify: `src/data/dataset.py`
- Modify: `tests/test_asymmetric_dataset.py`

- [ ] **Step 3.1 — Write failing tests**

First, find the `synthetic_cxi` fixture in `tests/test_asymmetric_dataset.py` and add the geometry keys to it. Look for the fixture that creates a CXI file and add these keys inside the `h5py.File` write block:

```python
# Inside the synthetic_cxi fixture, add after creating entry_1/data_1/data:
det_grp = f.require_group("entry_1/instrument_1/detector_1")
det_grp.create_dataset("distance", data=np.float64(0.1))
det_grp.create_dataset("x_pixel_size", data=np.float64(1e-4))
src_grp = f.require_group("entry_1/instrument_1/source_1")
src_grp.create_dataset("wavelength", data=np.float64(1.3e-10))
```

Then add these two tests after the existing `AsymmetricCXIDataset` tests:

```python
def test_hitfinder_runs_before_gcn(synthetic_cxi: Path) -> None:
    """find_peaks must be called on the raw assembled frame, before gcn()."""
    from unittest.mock import patch

    call_order: list[str] = []

    class OrderTrackingHitfinder:
        def find_peaks(self, frame: np.ndarray) -> np.ndarray:
            call_order.append("find_peaks")
            return np.zeros((0, 2), dtype=np.float32)

    def recording_gcn(frame: np.ndarray) -> np.ndarray:
        call_order.append("gcn")
        return frame

    with patch("src.data.dataset.gcn", side_effect=recording_gcn):
        ds = AsymmetricCXIDataset(
            session_ids=["s0"],
            session_map={"s0": synthetic_cxi},
            hitfinder=OrderTrackingHitfinder(),
        )
        ds[0]

    assert "find_peaks" in call_order
    assert "gcn" in call_order
    assert call_order.index("find_peaks") < call_order.index("gcn"), (
        "find_peaks must run before gcn"
    )


def test_set_geometry_called_with_cxi_params(synthetic_cxi: Path) -> None:
    """AsymmetricCXIDataset calls set_geometry with dist/wavelength/pixel_size."""
    set_geom_calls: list[dict] = []

    class GeomCapturingHitfinder:
        def set_geometry(self, **kwargs: float) -> None:
            set_geom_calls.append(kwargs)

        def find_peaks(self, frame: np.ndarray) -> np.ndarray:
            return np.zeros((0, 2), dtype=np.float32)

    ds = AsymmetricCXIDataset(
        session_ids=["s0"],
        session_map={"s0": synthetic_cxi},
        hitfinder=GeomCapturingHitfinder(),
    )
    ds[0]

    assert len(set_geom_calls) >= 1
    call = set_geom_calls[0]
    assert "dist" in call
    assert "wavelength" in call
    assert "pixel_size" in call
    assert call["dist"] == pytest.approx(0.1)
    assert call["wavelength"] == pytest.approx(1.3e-10)
    assert call["pixel_size"] == pytest.approx(1e-4)
```

- [ ] **Step 3.2 — Run tests to confirm they fail**

```bash
pytest tests/test_asymmetric_dataset.py::test_hitfinder_runs_before_gcn \
       tests/test_asymmetric_dataset.py::test_set_geometry_called_with_cxi_params -v
```

Expected: `FAILED` — order test fails (gcn runs before find_peaks); set_geometry test fails (method never called).

- [ ] **Step 3.3 — Modify `dataset.py`: cache geometry in `__init__`**

In `AsymmetricCXIDataset.__init__`, add a `_path_to_geom` dict. Extend the existing `unique_paths` loop (around line 214):

```python
        self._path_to_desc: dict[Path, str] = {}
        self._path_to_geom: dict[Path, dict[str, float]] = {}   # ← add this line
        for p in unique_paths:
            try:
                desc = read_detector_description(p)
                self._path_to_desc[p] = desc
            except (ValueError, KeyError, OSError):
                pass
            try:
                with h5py.File(p, "r") as _f:
                    self._path_to_geom[p] = {
                        "dist": float(_f["entry_1/instrument_1/detector_1/distance"][()]),
                        "wavelength": float(_f["entry_1/instrument_1/source_1/wavelength"][()]),
                        "pixel_size": float(_f["entry_1/instrument_1/detector_1/x_pixel_size"][()]),
                        # x_pixel_size == y_pixel_size for all four supported detectors
                    }
            except (KeyError, OSError):
                pass
```

Note: `h5py` is already imported in `dataset.py` (used elsewhere). Verify with `grep -n "^import h5py" src/data/dataset.py`.

- [ ] **Step 3.4 — Modify `dataset.py`: fix order + call `set_geometry` in `__getitem__`**

In `AsymmetricCXIDataset.__getitem__`, replace the block starting at the assembly section. Current code (around lines 251–255):

```python
        # GCN applied to the full assembled frame before padding/crop.
        assembled = gcn(assembled)

        # --- Run hitfinder ---
        centroids = self._hitfinder.find_peaks(assembled)  # (N, 2) float32 [x, y]
```

Replace with:

```python
        # Update geometry for this CXI file (no-op if hitfinder lacks set_geometry).
        if path in self._path_to_geom and hasattr(self._hitfinder, "set_geometry"):
            self._hitfinder.set_geometry(**self._path_to_geom[path])

        # --- Run hitfinder on raw assembled frame (before GCN) ---
        centroids = self._hitfinder.find_peaks(assembled)  # (N, 2) float32 [x, y]

        # GCN applied to the full assembled frame after hitfinder.
        assembled = gcn(assembled)
```

Also update the docstring inside the class (lines 176–185) to correct the step order:

```
      1. Read + assemble to native resolution
      2. Run hitfinder on raw assembled frame → centroids (N_peaks, 2)
      3. Apply GCN to the full assembled frame
      4. Pad frame by PAD_BORDER_DEFAULT px on each edge; shift centroids
      ...
```

- [ ] **Step 3.5 — Run the new tests to confirm they pass**

```bash
pytest tests/test_asymmetric_dataset.py::test_hitfinder_runs_before_gcn \
       tests/test_asymmetric_dataset.py::test_set_geometry_called_with_cxi_params -v
```

Expected: `PASSED`

- [ ] **Step 3.6 — Run the full asymmetric dataset test suite**

```bash
pytest tests/test_asymmetric_dataset.py -v
```

Expected: all tests pass.

- [ ] **Step 3.7 — Run the complete test suite**

```bash
pytest tests/ -v
```

Expected: all tests pass. Fix any regressions before proceeding.

- [ ] **Step 3.8 — Commit**

```bash
git add src/data/dataset.py tests/test_asymmetric_dataset.py
git commit -m "fix: run hitfinder before GCN in AsymmetricCXIDataset; add set_geometry call"
```

---

## Task 4: Update `docs/data_spec.md` with confirmed geometry keys

**Files:**
- Modify: `docs/data_spec.md`

- [ ] **Step 4.1 — Add geometry keys section**

In `docs/data_spec.md`, after the existing "Confirmed HDF5/CXI Keys" section (at the end of the file), add:

```markdown
---

## Confirmed CXI Geometry Keys (verified 2026-08-04, Resonet production files)

Probed against all four detector CXI files in
`/data/bioxfel/user/gihan/Resonet/production/`.

All four detectors share identical HDF5 paths:

| Parameter | HDF5 key | Units |
|-----------|----------|-------|
| Detector distance | `entry_1/instrument_1/detector_1/distance` | metres |
| X-ray wavelength | `entry_1/instrument_1/source_1/wavelength` | metres |
| Pixel size (x) | `entry_1/instrument_1/detector_1/x_pixel_size` | metres |
| Pixel size (y) | `entry_1/instrument_1/detector_1/y_pixel_size` | metres |

Beam centre is **not stored** in any CXI file.
`gpu_pf8.py` computes it as `(frame_rows / 2) × pixel_size` and
`(frame_cols / 2) × pixel_size`. All four detectors have equal x/y pixel sizes.

Observed pixel sizes per detector:

| Detector | Description in CXI | Pixel size |
|----------|---------------------|-----------|
| ePix10k 2.2M | `b'ePix10k 2.2M'` | 100 µm |
| AGIPD 1M | `b'AGIPD 1M'` | 200 µm |
| EIGER 4M | `b'EIGER 4M'` | 100 µm |
| Jungfrau 4M | `b'Jungfrau 4M'` | 75 µm |

`AsymmetricCXIDataset` reads `x_pixel_size`, `distance`, and `wavelength` at
`__init__` time and passes them to `hitfinder.set_geometry()` before each frame's
`find_peaks` call.
```

- [ ] **Step 4.2 — Commit**

```bash
git add docs/data_spec.md
git commit -m "docs: add confirmed CXI geometry HDF5 keys (all 4 detectors)"
```

---

## Verification

After all tasks are complete, run the full suite end-to-end:

```bash
pytest tests/ -v --tb=short
```

All tests should pass. Then do a final smoke check on a real CXI file to confirm `set_geometry` is called with sensible values:

```python
from src.hitfinders import MockHitfinder
from src.data.dataset import AsymmetricCXIDataset

class PrintingHitfinder:
    def set_geometry(self, **kwargs):
        print("set_geometry:", kwargs)
    def find_peaks(self, frame):
        import numpy as np
        return np.zeros((0, 2), dtype=np.float32)

ds = AsymmetricCXIDataset(
    session_ids=["s0"],
    session_map={"s0": "/data/bioxfel/user/gihan/Resonet/production/epix10k_20k/compressed0.cxi"},
    hitfinder=PrintingHitfinder(),
)
ds[0]
# Expected output: set_geometry: {'dist': 0.096, 'wavelength': 1.305e-10, 'pixel_size': 0.0001}
```
