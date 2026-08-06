# GPU Hitfinder — Full Implementation Design

**Date:** 2026-08-04  
**Branch:** phase-04-augmentation  
**Status:** Approved — ready for implementation plan

---

## Context

`GPUHitfinder` (`src/hitfinders/gpu.py`) is a wrapper that dynamically imports a
user-supplied script and calls `find_peaks(frame)` on it. The stub at
`src/hitfinders/gpu.py` currently raises `NotImplementedError`. This design
specifies the concrete backend script (`src/hitfinders/gpu_pf8.py`) using pyFAI's
`OCL_PeakFinder` (OpenCL-backed PF8 algorithm), plus three supporting changes:
adding `set_geometry` to the wrapper, fixing a hitfinder/GCN ordering bug in
`dataset.py`, and confirming CXI geometry keys in `data_spec.md`.

**Order bug fixed here:** The code in `AsymmetricCXIDataset` was calling
`gcn(assembled)` before `find_peaks(assembled)`, contradicting the pipeline spec.
This implementation fixes the order so hitfinder runs on raw assembled ADU values
(pre-GCN), which is the statistically correct input for OCL_PeakFinder.

---

## Confirmed CXI Geometry Keys (probed 2026-08-04)

All four detectors share identical HDF5 paths:

| Parameter | HDF5 key | Units |
|-----------|----------|-------|
| Distance | `entry_1/instrument_1/detector_1/distance` | metres |
| Wavelength | `entry_1/instrument_1/source_1/wavelength` | metres |
| Pixel size (x) | `entry_1/instrument_1/detector_1/x_pixel_size` | metres |
| Pixel size (y) | `entry_1/instrument_1/detector_1/y_pixel_size` | metres |

Beam centre is **not stored** in any CXI file — computed as `(frame_rows/2) * pixel_size`
and `(frame_cols/2) * pixel_size` at `_rebuild` time.

Observed pixel sizes per detector:
| Detector | Pixel size |
|----------|-----------|
| ePix10k 2.2M | 100 µm |
| AGIPD 1M | 200 µm |
| EIGER 4M | 100 µm |
| Jungfrau 4M | 75 µm |

---

## Architecture & Data Flow

```
AsymmetricCXIDataset.__getitem__
  │
  ├─ open CXI (h5py, lazy)
  │   ├─ read dist, wavelength, x_pixel_size
  │   └─ hitfinder.set_geometry(dist, wavelength, pixel_size)  [if attr exists]
  │
  ├─ assemble frame via PADAssembler → float32 (H, W)
  │
  ├─ hitfinder.find_peaks(assembled)   ← BEFORE gcn() [order fix]
  │   │
  │   │  GPUHitfinder (gpu.py)
  │   │    set_geometry(**kwargs) → self._mod.set_geometry(**kwargs)
  │   │    find_peaks(frame)      → self._mod.find_peaks(frame)
  │   │
  │   │  gpu_pf8.py
  │   │    if _geometry_changed or shape changed → _rebuild()
  │   │      Detector(pixel1, pixel2)
  │   │      AzimuthalIntegrator(dist, poni1, poni2, wavelength)
  │   │      setup_sparse_integrator() → CSR LUT
  │   │      OCL_PeakFinder(lut, ...)      ← built once per geometry
  │   │    mask = frame < 0
  │   │    frame_clean[mask] = 0
  │   │    res = pf.peakfinder8(data, error_model, polarization, ...)
  │   │    return column_stack([res["pos1"], res["pos0"]])  → (N,2) [x,y]
  │   │
  │   └─ centroids (N, 2) float32
  │
  ├─ gcn(assembled)                    ← AFTER find_peaks() [order fix]
  └─ crop → augment → lcn → tensor
```

---

## Component Interfaces

### `src/hitfinders/gpu_pf8.py` (new file)

Module-level functions (no class — consumed by `GPUHitfinder` via dynamic import):

```python
set_geometry(dist, wavelength, pixel_size, poni1, poni2)
    # None values keep the current setting; sets _geometry_changed=True

find_peaks(frame: np.ndarray) -> np.ndarray
    # frame: 2D float32 (H,W), raw assembled pre-GCN
    # returns: float32 (N_peaks, 2), columns [x, y]; shape (0,2) if no peaks
```

Internal:
```python
_rebuild(frame_shape: tuple[int, int]) -> None
    # Reconstructs AzimuthalIntegrator + OCL_PeakFinder
    # Triggered when: _geometry_changed=True OR frame.shape != _last_shape
```

**Mask strategy:** `frame < 0` only. Sigma-clipping inside OCL_PeakFinder handles
outlier/saturation cases. No hard saturation threshold needed for pre-GCN float32 data.

**poni computation:** `poni1 = (nrows/2) * pixel_size`, `poni2 = (ncols/2) * pixel_size`
unless overridden via `set_geometry(poni1=..., poni2=...)`.

**OCL context:** `pyopencl.create_some_context(interactive=False)` — created once,
reused across `_rebuild` calls. Lets pyopencl pick the best GPU automatically.

### Env-var tuning knobs

| Env var | Default | Notes |
|---------|---------|-------|
| `HITFINDER_DIST` | `0.2` m | fallback if CXI key missing |
| `HITFINDER_WAVELENGTH` | `1.3e-10` m | fallback |
| `HITFINDER_PIXEL_SIZE` | `75e-6` m | fallback (JUNGFRAU value) |
| `HITFINDER_NPT` | `1000` | radial bins for background |
| `HITFINDER_POL_FACTOR` | `0.99` | polarization factor |
| `HITFINDER_CYCLE` | `5` | sigma-clip cycles |
| `HITFINDER_CUTOFF_PICK` | `3.0` | intense-pixel σ cutoff |
| `HITFINDER_CUTOFF_PEAK` | `3.0` | peak σ cutoff |
| `HITFINDER_NOISE` | `1.0` | noise floor |
| `HITFINDER_CONNECTED` | `3` | min connected pixels per peak |
| `HITFINDER_PATCH_SIZE` | `3` | centroiding neighbourhood |

### `src/hitfinders/gpu.py` (two additions)

1. `_load()` stores `self._mod = mod` alongside `self._fn = mod.find_peaks`
2. New method:
```python
def set_geometry(self, **kwargs) -> None:
    if self._fn is None:
        self._load()
    if hasattr(self._mod, "set_geometry"):
        self._mod.set_geometry(**kwargs)
```

### `src/data/dataset.py` (two targeted edits in `AsymmetricCXIDataset.__getitem__`)

1. After reading frame from CXI, call geometry update:
```python
# read geometry from CXI metadata, fall back silently on KeyError
try:
    dist = float(f["entry_1/instrument_1/detector_1/distance"][()])
    wl   = float(f["entry_1/instrument_1/source_1/wavelength"][()])
    px   = float(f["entry_1/instrument_1/detector_1/x_pixel_size"][()])  # x==y confirmed for all 4 detectors
    if hasattr(self._hitfinder, "set_geometry"):
        self._hitfinder.set_geometry(dist=dist, wavelength=wl, pixel_size=px)
except KeyError:
    pass  # hitfinder falls back to env-var defaults
```

2. Swap call order: `find_peaks(assembled)` before `gcn(assembled)`.

---

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| CXI geometry key missing | `KeyError` caught; env-var defaults used silently |
| `set_geometry` all-None | No-op; `_geometry_changed` stays False |
| `find_peaks` before any `set_geometry` | env-var defaults used on first `_rebuild` |
| 0 peaks found | `np.zeros((0, 2), dtype=np.float32)` returned |
| pyopencl unavailable | `ImportError` propagates (GPU hitfinder is opt-in) |
| Script lacks `set_geometry` | `hasattr` guard in `gpu.py`; no error |

---

## Tests (additions to `tests/test_hitfinders.py`)

| Test | What it verifies |
|------|-----------------|
| `test_gpu_hitfinder_set_geometry_passthrough` | `GPUHitfinder.set_geometry` calls `_mod.set_geometry` with correct kwargs |
| `test_gpu_hitfinder_set_geometry_no_attr` | No error when script lacks `set_geometry` (backward compat) |
| `test_gpu_pf8_set_geometry_updates_state` | `set_geometry` updates `_dist`, `_wavelength`, `_pixel_size`, sets `_geometry_changed=True` |
| `test_gpu_pf8_find_peaks_shape` | With mocked `OCL_PeakFinder`, output is `(N, 2)` float32 |
| `test_gpu_pf8_find_peaks_empty` | Returns `(0, 2)` float32 when no peaks |
| `test_dataset_hitfinder_before_gcn` | `find_peaks` called before `gcn` in `AsymmetricCXIDataset` (mock call order) |

---

## Files to Create / Modify

| File | Action |
|------|--------|
| `src/hitfinders/gpu_pf8.py` | **Create** |
| `src/hitfinders/gpu.py` | Modify — `self._mod` + `set_geometry` pass-through |
| `src/data/dataset.py` | Modify — order fix + `set_geometry` call on file open |
| `docs/data_spec.md` | Modify — add confirmed geometry HDF5 keys table |
| `tests/test_hitfinders.py` | Modify — add 6 new test cases |
