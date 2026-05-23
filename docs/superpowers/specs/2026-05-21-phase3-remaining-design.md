# Phase 3 Remaining Items — Design Spec

**Date:** 2026-05-21
**Phase:** 3 (Preprocessing)
**Branch:** `phase-03`

---

## Overview

Two tasks remain before Phase 3 can close:

1. **LCN Window Ablation** — compare preprocessing output across window sizes 3, 9, 15, 31 on real detector data via visual inspection; record chosen window in PLANNING.md.
2. **HDF5 Key Confirmation** — probe real HDF5/CXI files from all four detectors to verify (or correct) the `entry/data/data` key used in `src/preprocessing/io.py`.

---

## Task 1: LCN Window Ablation

### Goal

Determine the best LCN window size from {3, 9, 15, 31} by visually comparing preprocessed images across all four detector types. The default in `normalize.py` is `LCN_WINDOW_DEFAULT = 9`; this ablation either confirms it or changes it.

### Implementation

**File:** `notebooks/lcn_ablation.ipynb`

**Structure:**

| Cell | Content |
|------|---------|
| Config | File paths for one representative frame per detector (user-supplied variables) |
| Load | `load_pad_geometry()` + read panel data from HDF5 for each detector |
| Preprocess | `preprocess(panel_data, pads, lcn_window=W)` for W ∈ {3, 9, 15, 31} |
| Plot | 4×4 matplotlib grid — rows = detectors, columns = window sizes; `viridis` colourmap; shared colour scale per row |
| Export | Save each detector row as PNG to `docs/figures/lcn_ablation/<detector>_lcn_comparison.png` |
| Decision | Markdown cell recording chosen window size and rationale |

**Reuse:** Uses `src.preprocessing.pipeline.preprocess()` and `src.preprocessing.geometry.load_pad_geometry()` unchanged — no new source code.

### Outputs

- `notebooks/lcn_ablation.ipynb` (committed, outputs cleared before commit)
- `docs/figures/lcn_ablation/AGIPD_lcn_comparison.png`
- `docs/figures/lcn_ablation/JUNGFRAU_4M_lcn_comparison.png`
- `docs/figures/lcn_ablation/ePix10k_lcn_comparison.png`
- `docs/figures/lcn_ablation/Eiger4M_lcn_comparison.png`

### PLANNING.md update

Check off the ablation item; record chosen window size.

---

## Task 2: HDF5 Key Confirmation

### Goal

Verify that the key `entry/data/data` used in `src/preprocessing/io.py` is correct for all four detectors. Update code and docs if any detector uses a different key.

### Detector Coverage

All four detectors have real files available: AGIPD, JUNGFRAU 4M, ePix10k, Eiger4M.

### Implementation

**File:** `scripts/probe_hdf5.py`

Accepts one file path per detector (CLI args or top-of-file config dict). For each file:
1. Opens with `h5py` and walks the full key tree via `visititems()`
2. Prints the hierarchy with shape and dtype for every dataset
3. Flags whether `entry/data/data` exists and whether its shape matches the expected detector dimensions from CLAUDE.md

**Outcomes:**

| Scenario | Action |
|----------|--------|
| `entry/data/data` correct for all four | No code change — mark PLANNING.md item complete |
| One or more detectors use a different key | Update `io.py` with a per-detector key map; update `docs/data_spec.md` |

### Test

Add parametrized test in `tests/test_io.py`:

```python
@pytest.mark.skipif(not Path(FILE).exists(), reason="real data not available")
def test_hdf5_key_<detector>():
    ...
```

Guarded on file existence so CI passes without real data. Reads confirmed key, asserts shape matches detector spec.

### PLANNING.md update

Check off the HDF5 key confirmation item; record confirmed keys per detector.

---

## Feature Branches

| Task | Branch |
|------|--------|
| LCN ablation | `phase-03-lcn-ablation` |
| HDF5 key confirmation | `phase-03-hdf5-key` |

Each branch gets a PR targeting `phase-03` (not `main`). Run `/code-review:code-review` and `/requesting-code-review` before merging.

---

## Definition of Done

- [ ] `notebooks/lcn_ablation.ipynb` committed with cleared outputs
- [ ] 4 PNG figures committed to `docs/figures/lcn_ablation/`
- [ ] Chosen LCN window recorded in PLANNING.md
- [ ] `scripts/probe_hdf5.py` committed and run against all four detector files
- [ ] `io.py` and `docs/data_spec.md` updated if key differs
- [ ] Parametrized HDF5 key test added to `tests/test_io.py`
- [ ] Both PLANNING.md items checked off
- [ ] Phase-03 → main PR opened once both feature PRs are merged
