# GPU Hitfinder — Implementation Spec

> **Status:** In-progress design. Resume from here in the next session.
> All design decisions below were agreed in conversation on 2026-08-04.

---

## Goal

Build a concrete `find_peaks` script for `GPUHitfinder` (wrapper lives at
`src/hitfinders/gpu.py`) that uses **pyFAI's `OCL_PeakFinder`** (OpenCL-backed)
instead of the NumPy PF8 backend.

Output file: `src/hitfinders/gpu_pf8.py`

---

## Interface Contract

`GPUHitfinder` (`src/hitfinders/gpu.py`) dynamically imports a user-supplied script
and calls two functions on it:

```python
set_geometry(dist=None, wavelength=None, pixel_size=None, poni1=None, poni2=None)
find_peaks(frame: np.ndarray) -> np.ndarray   # shape (N_peaks, 2), columns [x, y]
```

### Design choice: Option B — `set_geometry` + module-level state

- `set_geometry()` is called **once per CXI file open** from `AsymmetricCXIDataset`
  (or wherever the file path changes) to update geometry from CXI metadata.
- `find_peaks()` is called **per frame** and rebuilds `ai` + `OCL_PeakFinder`
  lazily if geometry changed.
- `None` values in `set_geometry` keep the current default — callers only pass
  what they know.

---

## Geometry Source — CXI Metadata (unconfirmed keys)

Standard CXI schema paths to try (need probing against real files to confirm):

| Parameter | CXI key | Units |
|-----------|---------|-------|
| Distance | `entry_1/instrument_1/detector_1/distance` | metres |
| Wavelength | `entry_1/instrument_1/source_1/wavelength` | metres |
| Beam centre X | `entry_1/instrument_1/detector_1/beam_center_x` | pixels |
| Beam centre Y | `entry_1/instrument_1/detector_1/beam_center_y` | pixels |

**Action needed:** probe a real CXI file before implementing the dataset side:
```bash
python scripts/probe_hdf5.py /path/to/file.cxi 2>&1 | grep -iE "dist|wavelength|beam|energy|source"
```

---

## Sensible Fallback Defaults

All overridable via env vars (set by caller before import, or exported in shell):

| Env var | Default | Notes |
|---------|---------|-------|
| `HITFINDER_PIXEL_SIZE` | `75e-6` m | Eiger4M / JUNGFRAU pixel size |
| `HITFINDER_DIST` | `0.2` m | 200 mm — conservative XFEL distance |
| `HITFINDER_WAVELENGTH` | `1.3e-10` m | ~1.3 Å typical XFEL |
| `HITFINDER_PONI1` | frame centre × pixel_size | computed from frame shape at runtime |
| `HITFINDER_PONI2` | frame centre × pixel_size | computed from frame shape at runtime |
| `HITFINDER_DEVICE` | `"cuda"` | already set by `GPUHitfinder` wrapper |
| `HITFINDER_NPT` | `1000` | radial bins for background estimation |
| `HITFINDER_POL_FACTOR` | `0.99` | polarization factor |
| `HITFINDER_CYCLE` | `5` | sigma-clip cycles |
| `HITFINDER_CUTOFF_PICK` | `3.0` | intense-pixel cutoff (σ) |
| `HITFINDER_CUTOFF_PEAK` | `3.0` | peak cutoff (σ) |

---

## pyFAI Building Blocks (provided by user)

### Imports
```python
import os
import numpy
import pyFAI
from pyFAI import units
import pyopencl
from pyFAI.opencl.peak_finder import OCL_PeakFinder
from pyFAI.containers import ErrorModel
```

### Mask (per-frame)
```python
mask = numpy.logical_or(frame > 65000, frame < 0)
frame_clean = frame.copy()
frame_clean[mask] = 0
```

### Detector + AzimuthalIntegrator construction (no .poni file needed)
```python
det = pyFAI.detector.Detector(pixel1=PIXEL_SIZE, pixel2=PIXEL_SIZE)
det.mask = mask   # set per-frame

ai = pyFAI.AzimuthalIntegrator(
    dist=DIST,
    poni1=PONI1,   # beam centre row in metres = centre_row * pixel_size
    poni2=PONI2,   # beam centre col in metres = centre_col * pixel_size
    wavelength=WAVELENGTH,
)
ai.detector = det
```

### OpenCL context
```python
ctx = pyopencl.create_some_context(interactive=False)
```
Note: `HITFINDER_DEVICE` is a PyTorch-style string set by the wrapper. Need to
map it to an OpenCL platform/device index — or just use `create_some_context`
and let pyopencl pick the best GPU automatically (acceptable for v1).

### OCL_PeakFinder init
```python
unit = units.to_unit("r_mm")
image_size = det.shape[0] * det.shape[1]

integrator = ai.setup_sparse_integrator(
    ai.detector.shape, NPT, mask=mask,
    unit=unit, split="no", algo="CSR", scale=False
)
polarization = ai._cached_array["last_polarization"]

pf = OCL_PeakFinder(
    integrator.lut,
    image_size=image_size,
    bin_centers=integrator.bin_centers,
    radius=ai._cached_array[unit.name.split("_")[0] + "_center"],
    mask=mask,
    ctx=ctx,
    unit=unit,
)
```

### Peak-finding call
```python
kwargs = {
    "data": frame_clean,
    "error_model": ErrorModel.parse("azimuthal"),
    "polarization": polarization.array,
    "polarization_checksum": polarization.checksum,
}

n_intense = pf.count_intense(**kwargs, cycle=CYCLE, cutoff_pick=CUTOFF_PICK)
n_peaks   = pf._count_peak(**kwargs, cycle=CYCLE, cutoff_peak=CUTOFF_PEAK)
```

### ⚠️ MISSING PIECE — peak position extraction

The example only shows `count_intense` / `_count_peak` (return counts, not coords).
Need the call that returns actual `(row, col)` / `(x, y)` positions. Likely one of:

```python
peaks = pf.find_peaks(**kwargs, cycle=CYCLE, cutoff_pick=CUTOFF_PICK, cutoff_peak=CUTOFF_PEAK)
# or inspect pf after counting:
peaks = pf.peaks   # structured array?
```

**Action needed:** user to provide example of coordinate extraction before
`find_peaks()` in `gpu_pf8.py` can be completed.

---

## Changes Required Outside the Script

### 1. `src/hitfinders/gpu.py` — add `set_geometry` pass-through

```python
def _load(self) -> None:
    ...
    self._mod = mod          # store module reference (new)
    self._fn = mod.find_peaks

def set_geometry(self, **kwargs) -> None:
    if self._fn is None:
        self._load()
    if hasattr(self._mod, "set_geometry"):
        self._mod.set_geometry(**kwargs)
```

### 2. `src/data/dataset.py` — call `set_geometry` on file open

In `AsymmetricCXIDataset.__getitem__` (or wherever CXI files are opened),
after confirming CXI metadata keys:

```python
with h5py.File(path, 'r') as f:
    dist = float(f["entry_1/instrument_1/detector_1/distance"][()])
    wl   = float(f["entry_1/instrument_1/source_1/wavelength"][()])
self.hitfinder.set_geometry(dist=dist, wavelength=wl)
```

---

## Files to Create / Modify

| File | Action | Status |
|------|--------|--------|
| `src/hitfinders/gpu_pf8.py` | Create | **Blocked** on peak position extraction |
| `src/hitfinders/gpu.py` | Modify — add `set_geometry` pass-through + `self._mod` | Ready |
| `src/data/dataset.py` | Modify — call `set_geometry` on CXI open | Blocked on confirmed CXI keys |
| `docs/data_spec.md` | Update — add confirmed geometry HDF5 keys | Blocked on probing CXI files |
| `tests/test_hitfinders.py` | Add `GPUHitfinder` + `set_geometry` tests | After implementation |

---

## Next Session Checklist

1. [ ] User provides peak coordinate extraction example (unblock `gpu_pf8.py`)
2. [ ] Probe a real CXI file for geometry keys (`dist`, `wavelength`, `beam_center_x/y`)
3. [ ] Implement `gpu_pf8.py` using all building blocks above
4. [ ] Update `gpu.py` wrapper with `set_geometry` + `self._mod`
5. [ ] Update `dataset.py` to call `set_geometry` on file open
6. [ ] Update `data_spec.md` with confirmed geometry keys
7. [ ] Write tests
