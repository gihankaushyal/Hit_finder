# Geometry-Aware CXI Assembly — Design Spec

**Date:** 2026-06-26  
**Branch:** `phase-04-geometry-assembly` (from `main`)  
**Status:** Approved — ready for implementation

---

## Context and Motivation

The current LODO pipeline uses `preprocess_assembled()` for all detectors, which naively row-stacks multi-panel data via `_to_2d()`. A smoke test revealed that only JUNGFRAU 4M arrives pre-assembled; the other three detectors are stored as unassembled or stacked panels:

| Detector | Raw shape | Current handling | Problem |
|---|---|---|---|
| AGIPD | (16, 512, 128) | Row-stacked | Panels concatenated, no spatial layout |
| JUNGFRAU 4M | (2164, 2068) | preprocess_assembled | Already correct — no change needed |
| ePix10k | (5632, 384) | Row-stacked | 64 panels concatenated, no geometry |
| Eiger4M | (5632, 384) | Row-stacked | 64 panels concatenated, no geometry |

The network can learn panel-edge artifacts instead of Bragg spot features — a plausible explanation for fold 1 cross-detector AP of 0.56 (barely above chance).

CrystFEL geometry files for all detectors are available at `/data/bioxfel/user/gihan/Resonet/geoms/`. Reborn's `geometry_file_to_pad_geometry_list()` loads them correctly. Pixel counts verified to match raw data for all three detectors needing assembly.

---

## Panel Layout (Verified)

| Detector | Panels | Panel size | Total pixels | Data shape |
|---|---|---|---|---|
| AGIPD | 128 | 64 × 128 | 1,048,576 | (16, 512, 128) |
| ePix10k | 64 | 176 × 192 | 2,162,688 | (5632, 384) |
| Eiger4M | 64 | 176 × 192 | 2,162,688 | (5632, 384) |

**AGIPD splitting:** frame `(16, 512, 128)` → 16 modules × 8 ASICs each → 128 panels of `(64, 128)`. Split manually: `panel = frame[module, asic*64:(asic+1)*64, :]`.

**ePix10k / Eiger4M splitting:** `pads.split_data(frame)` using `parent_data_slice` from the CrystFEL geom file. Panels referenced by `min_ss:max_ss+1, min_fs:max_fs+1` within the `(5632, 384)` array.

---

## Detector Description Field

All 4 CXI files have `entry_1/instrument_1/detector_1/description`:

| Description string | Detector |
|---|---|
| `'AGIPD 1M'` | AGIPD |
| `'Jungfrau 4M'` | JUNGFRAU 4M |
| `'ePix10k 2.2M'` | ePix10k |
| `'EIGER 4M'` | Eiger4M |

Unknown descriptions raise `ValueError` — no silent fallback.

---

## Files to Create or Modify

### 1. Geometry files — copy into repo
Copy from `/data/bioxfel/user/gihan/Resonet/geoms/` to `src/preprocessing/data/`:
- `AGIPD.geom` → `agipd.geom`
- `Epix10k.geom` → `epix10k.geom`
- `Eigar.geom` → `eiger4m.geom`

### 2. `src/preprocessing/io.py` — add `read_detector_description()`
```python
def read_detector_description(path: str | Path) -> str:
    """Return the detector description string from CXI metadata.
    Reads entry_1/instrument_1/detector_1/description.
    Raises ValueError if key is absent.
    """
```

### 3. `src/preprocessing/geometry.py` — add `get_geometry()`
```python
_GEOM_FILES = {
    "AGIPD 1M":     Path(__file__).parent / "data" / "agipd.geom",
    "ePix10k 2.2M": Path(__file__).parent / "data" / "epix10k.geom",
    "EIGER 4M":     Path(__file__).parent / "data" / "eiger4m.geom",
}
_GEOM_CACHE: dict[str, PADGeometryList] = {}

def get_geometry(detector_desc: str) -> PADGeometryList:
    """Load and cache PADGeometryList for the given detector description.
    Raises ValueError for unknown descriptors or JUNGFRAU (already assembled).
    """
```

Uses `geometry_file_to_pad_geometry_list()` from `reborn.external.crystfel`. Module-level `_GEOM_CACHE` so geom files are loaded once per process.

### 4. `src/preprocessing/pipeline.py` — add `preprocess_with_geometry()`
```python
def preprocess_with_geometry(
    frame: np.ndarray,
    pads: PADGeometryList,
    detector_desc: str,
    lcn_window: int = LCN_WINDOW_DEFAULT,
) -> np.ndarray:
    """Geometry-aware preprocessing: split panels → assemble → GCN → LCN → resize."""
```

Panel splitting logic:
- `"AGIPD 1M"`: manual loop `frame[m, a*64:(a+1)*64, :]` for m in 0..15, a in 0..7
- `"ePix10k 2.2M"` / `"EIGER 4M"`: `pads.split_data(frame)`

Then: `assemble_image(pads, panels)` → `gcn()` → `lcn()` → `sk_resize(TARGET_SIZE)`.

Reuses existing `assemble_image()` from `src/preprocessing/geometry.py`, `gcn()` and `lcn()` from `src/preprocessing/normalize.py`.

### 5. `src/data/dataset.py` — update `MultiFrameCXIDataset`

In `__init__`:
- Read detector description for each unique CXI path via `read_detector_description()`
- Store `_path_to_desc: dict[Path, str]`
- Pre-load geometry: `_desc_to_pads: dict[str, PADGeometryList]` (call `get_geometry()` for non-JUNGFRAU descriptors)
- Both done before DataLoader workers fork — safe for multiprocessing

In `__getitem__`:
```python
desc = self._path_to_desc[path]
if desc in self._desc_to_pads:
    frame = preprocess_with_geometry(frame, self._desc_to_pads[desc], desc)
else:
    frame = preprocess_assembled(frame)  # JUNGFRAU 4M path
```

No new constructor parameters — geometry is used automatically based on metadata.

---

## Data Flow

```
CXI file
  → read_detector_description()  →  "AGIPD 1M"
  → get_geometry("AGIPD 1M")    →  PADGeometryList (128 panels, cached)
  → read_frame(path, idx)        →  np.ndarray (16, 512, 128)
  → split into 128 × (64, 128)
  → assemble_image(pads, panels) →  2D spatially correct image
  → gcn() → lcn() → resize()    →  (224, 224) float32
```

JUNGFRAU bypasses geometry and calls `preprocess_assembled()` directly.

---

## Error Handling

- Missing description key → `ValueError` (raised by `read_detector_description`)
- Unknown description string → `ValueError` (raised by `get_geometry`)
- Panel count mismatch → caught naturally by `pads.split_data()` or manual split length check
- No silent fallback anywhere

---

## Testing

### New test file: `tests/test_geometry_assembly.py`
- One parametrized test per detector: load frame 0 from production CXI, run through `preprocess_with_geometry`, assert output shape `(224, 224)` and dtype `float32`
- JUNGFRAU: assert `preprocess_assembled` still produces `(224, 224)`
- Test `read_detector_description()` returns correct string for each file
- Test `get_geometry()` raises `ValueError` for `"Jungfrau 4M"` and unknown strings

### Update `scripts/smoke_test_detector_shapes.py`
- Add "assembled shape" column showing the intermediate assembled 2D shape before resize
- Confirm all 4 detectors go through the right path

### LODO re-run
After merging, re-run all 4 LODO folds with geometry-corrected inputs and compare cross-detector AP to the pre-fix baseline (fold 1: 0.5649).

---

## Branch and PR Plan

- Branch: `phase-04-geometry-assembly` from `main`
- PR: `phase-04-geometry-assembly` → `main`
- Run `/code-review high` before merge
