# Phase 3 Remaining Items Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the two remaining Phase 3 checklist items — LCN window ablation (notebook + PNGs) and HDF5 key confirmation (probe script + parametrized tests) — so Phase 3 can close with a phase-03 → main PR.

**Architecture:** Two independent feature branches (`phase-03-lcn-ablation`, `phase-03-hdf5-key`), each with a PR targeting `phase-03`. The ablation is purely exploratory (no new source code); the HDF5 task may update `src/preprocessing/io.py` and `docs/data_spec.md` depending on probe findings.

**Tech Stack:** Python 3.11, h5py, numpy, matplotlib, Jupyter (nbformat), Reborn (`src.preprocessing.geometry`, `src.preprocessing.pipeline`), pytest, gh CLI.

---

## Part A — LCN Window Ablation

---

### Task A1: Create feature branch

**Files:** none

- [ ] **Step 1: Create and push branch**

```bash
git checkout phase-03
git pull
git checkout -b phase-03-lcn-ablation
git push -u origin phase-03-lcn-ablation
```

---

### Task A2: Write the ablation notebook

**Files:**
- Create: `notebooks/lcn_ablation.ipynb`

- [ ] **Step 1: Create the notebook file**

Write `notebooks/lcn_ablation.ipynb` with the following cell contents (use `jupyter notebook` or paste the JSON directly). Each block below is one cell.

**Cell 1 — Imports and config (code):**
```python
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing.geometry import load_pad_geometry
from src.preprocessing.pipeline import preprocess

# ── Set these to real detector files on your machine ──────────────────────────
DETECTOR_FILES: dict[str, str] = {
    "AGIPD":        "/path/to/agipd_frame.h5",
    "JUNGFRAU_4M":  "/path/to/jungfrau_frame.cxi",
    "ePix10k":      "/path/to/epix_frame.h5",
    "Eiger4M":      "/path/to/eiger_frame.h5",
}

WINDOWS = [3, 9, 15, 31]
OUT_DIR = Path("docs/figures/lcn_ablation")
OUT_DIR.mkdir(parents=True, exist_ok=True)
```

**Cell 2 — Helper: load one frame (code):**
```python
import h5py


def load_panel_data(detector: str, path: str) -> tuple:
    """Return (panel_data_list, pads) for one detector frame."""
    pads = load_pad_geometry(detector)
    with h5py.File(path, "r") as f:
        raw = f["entry/data/data"][()]  # shape depends on detector
    if raw.ndim == 2:
        panel_data = [raw[i * pads[0].n_ss:(i + 1) * pads[0].n_ss, :]
                      for i in range(len(pads))] if len(pads) > 1 else [raw]
    elif raw.ndim == 3:
        panel_data = [raw[i] for i in range(raw.shape[0])]
    else:
        raise ValueError(f"Unexpected raw ndim {raw.ndim} for {detector}")
    return panel_data, pads
```

**Cell 3 — Run ablation and plot (code):**
```python
fig, axes = plt.subplots(
    len(DETECTOR_FILES), len(WINDOWS),
    figsize=(4 * len(WINDOWS), 3 * len(DETECTOR_FILES)),
)

for row_idx, (detector, fpath) in enumerate(DETECTOR_FILES.items()):
    panel_data, pads = load_panel_data(detector, fpath)
    images = [preprocess(panel_data, pads, lcn_window=w) for w in WINDOWS]

    vmin = min(img.min() for img in images)
    vmax = max(img.max() for img in images)

    for col_idx, (w, img) in enumerate(zip(WINDOWS, images)):
        ax = axes[row_idx, col_idx]
        ax.imshow(img, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(f"window={w}", fontsize=9)
        if col_idx == 0:
            ax.set_ylabel(detector, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle("LCN Window Ablation — all four detectors", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / "all_detectors_lcn_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved all_detectors_lcn_comparison.png")
```

**Cell 4 — Export per-detector PNGs (code):**
```python
for detector, fpath in DETECTOR_FILES.items():
    panel_data, pads = load_panel_data(detector, fpath)
    fig, axes = plt.subplots(1, len(WINDOWS), figsize=(4 * len(WINDOWS), 3))
    images = [preprocess(panel_data, pads, lcn_window=w) for w in WINDOWS]
    vmin = min(img.min() for img in images)
    vmax = max(img.max() for img in images)
    for ax, w, img in zip(axes, WINDOWS, images):
        ax.imshow(img, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(f"window={w}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(detector, fontsize=11)
    plt.tight_layout()
    out = OUT_DIR / f"{detector}_lcn_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
```

**Cell 5 — Decision (markdown):**
```
## Decision

After visual inspection:

- **Chosen window size:** _fill in after review_
- **Rationale:** _fill in after review_
- **Action:** Update `LCN_WINDOW_DEFAULT` in `src/preprocessing/normalize.py` if not 9.
```

- [ ] **Step 2: Verify notebook runs end-to-end**

Fill in the four real file paths in Cell 1, then run all cells:
```bash
jupyter nbconvert --to notebook --execute notebooks/lcn_ablation.ipynb \
    --output notebooks/lcn_ablation_executed.ipynb
```
Expected: no errors, 5 PNG files in `docs/figures/lcn_ablation/`.

- [ ] **Step 3: Inspect plots and fill in the decision cell**

Open the 4×4 comparison grid. Choose the window where Bragg peaks are most visible against background. Fill in the markdown decision cell.

- [ ] **Step 4: Clear outputs and commit notebook**

```bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True \
    --to notebook notebooks/lcn_ablation.ipynb \
    --output notebooks/lcn_ablation.ipynb
git add notebooks/lcn_ablation.ipynb docs/figures/lcn_ablation/
git commit -m "feat(phase3): LCN window ablation notebook and comparison figures"
```

---

### Task A3: Update LCN_WINDOW_DEFAULT if needed

**Files:**
- Possibly modify: `src/preprocessing/normalize.py:10`

- [ ] **Step 1: Check current default**

```bash
grep "LCN_WINDOW_DEFAULT" src/preprocessing/normalize.py
```

- [ ] **Step 2: Update if chosen window ≠ 9**

If the ablation decision chose a different window, edit line 10 of `src/preprocessing/normalize.py`:

```python
# Change 9 to your chosen value, e.g. 15:
LCN_WINDOW_DEFAULT: int = 15
```

- [ ] **Step 3: Run full test suite to confirm nothing broke**

```bash
pytest tests/ -v
```
Expected: all tests pass.

- [ ] **Step 4: Commit if changed (skip if window stays 9)**

```bash
git add src/preprocessing/normalize.py
git commit -m "fix(normalize): update LCN_WINDOW_DEFAULT to <chosen> based on ablation"
```

---

### Task A4: Update PLANNING.md and open PR

**Files:**
- Modify: `PLANNING.md`

- [ ] **Step 1: Check off ablation item in PLANNING.md**

Find and update this line in `PLANNING.md`:
```
- [ ] Ablate LCN window size (window=9 default; compare 3, 9, 15, 31 on validation set)
```
Change to:
```
- [x] Ablate LCN window size — chosen window=<N> (see `docs/figures/lcn_ablation/`)
```

- [ ] **Step 2: Commit**

```bash
git add PLANNING.md
git commit -m "docs: mark LCN ablation complete in PLANNING.md (window=<N>)"
```

- [ ] **Step 3: Push and open PR targeting phase-03**

```bash
git push
gh pr create \
  --base phase-03 \
  --title "feat(phase3): LCN window ablation — notebook, figures, chosen window" \
  --body "$(cat <<'EOF'
## Summary
- Adds `notebooks/lcn_ablation.ipynb` comparing LCN windows 3, 9, 15, 31 across all four detectors
- Exports per-detector PNGs to `docs/figures/lcn_ablation/`
- Records chosen window in PLANNING.md decision cell
- Updates `LCN_WINDOW_DEFAULT` in normalize.py if chosen window ≠ 9

## Test plan
- [ ] Notebook executes without errors
- [ ] 5 PNG figures present in docs/figures/lcn_ablation/
- [ ] Decision cell filled in with chosen window and rationale
- [ ] pytest tests/ -v passes

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Run code review skills on the PR**

```bash
# Get the PR number from the output above, then:
# Invoke /code-review:code-review and /requesting-code-review in Claude Code
```

---

## Part B — HDF5 Key Confirmation

---

### Task B1: Create feature branch

**Files:** none

- [ ] **Step 1: Create and push branch from phase-03**

```bash
git checkout phase-03
git pull
git checkout -b phase-03-hdf5-key
git push -u origin phase-03-hdf5-key
```

---

### Task B2: Write failing parametrized HDF5 key tests

**Files:**
- Modify: `tests/test_io.py`

- [ ] **Step 1: Add parametrized real-data tests to `tests/test_io.py`**

Append this block at the end of `tests/test_io.py`:

```python
# ---------------------------------------------------------------------------
# Real-data HDF5 key confirmation (skipped when files absent — CI safe)
# ---------------------------------------------------------------------------

# Update paths to real detector files on your machine before running.
_REAL_DETECTOR_FILES: dict[str, str] = {
    "AGIPD":       "/path/to/agipd_frame.h5",
    "JUNGFRAU_4M": "/path/to/jungfrau_frame.cxi",
    "ePix10k":     "/path/to/epix_frame.h5",
    "Eiger4M":     "/path/to/eiger_frame.h5",
}

# Update these after running probe_hdf5.py if any key differs from entry/data/data.
_CONFIRMED_KEYS: dict[str, str] = {
    "AGIPD":       "entry/data/data",
    "JUNGFRAU_4M": "entry/data/data",
    "ePix10k":     "entry/data/data",
    "Eiger4M":     "entry/data/data",
}

# Expected ndim per detector (pre-assembly raw data).
_EXPECTED_NDIM: dict[str, int] = {
    "AGIPD":       3,   # (16, 512, 128)
    "JUNGFRAU_4M": 3,   # (8, 512, 1024)
    "ePix10k":     2,   # varies, at least 2D
    "Eiger4M":     2,   # (2162, 2068)
}


@pytest.mark.parametrize("detector", list(_REAL_DETECTOR_FILES.keys()))
def test_real_hdf5_key_exists(detector: str) -> None:
    path = Path(_REAL_DETECTOR_FILES[detector])
    if not path.exists():
        pytest.skip(f"Real data file not available: {path}")
    key = _CONFIRMED_KEYS[detector]
    with h5py.File(path, "r") as f:
        assert key in f, (
            f"Key '{key}' not found in {detector} file. "
            f"Run scripts/probe_hdf5.py to find the correct key."
        )


@pytest.mark.parametrize("detector", list(_REAL_DETECTOR_FILES.keys()))
def test_real_hdf5_data_shape(detector: str) -> None:
    path = Path(_REAL_DETECTOR_FILES[detector])
    if not path.exists():
        pytest.skip(f"Real data file not available: {path}")
    key = _CONFIRMED_KEYS[detector]
    with h5py.File(path, "r") as f:
        data = f[key][()]
    expected_ndim = _EXPECTED_NDIM[detector]
    assert data.ndim == expected_ndim, (
        f"{detector}: expected {expected_ndim}D array, got {data.ndim}D "
        f"with shape {data.shape}"
    )
    assert np.issubdtype(data.dtype, np.number), (
        f"{detector}: data dtype {data.dtype} is not numeric"
    )
```

- [ ] **Step 2: Run tests to confirm they skip gracefully (files not yet set)**

```bash
pytest tests/test_io.py -v -k "real_hdf5"
```
Expected: 8 tests, all `SKIPPED` with "Real data file not available".

- [ ] **Step 3: Commit failing/skipping tests**

```bash
git add tests/test_io.py
git commit -m "test(io): add parametrized real-data HDF5 key confirmation tests (skip when absent)"
```

---

### Task B3: Write the probe script

**Files:**
- Create: `scripts/probe_hdf5.py`

- [ ] **Step 1: Write `scripts/probe_hdf5.py`**

```python
"""Walk HDF5/CXI file structure — report keys, shapes, dtypes, and flag entry/data/data."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py

EXPECTED_KEY = "entry/data/data"

# Update these paths before running.
DETECTOR_FILES: dict[str, str] = {
    "AGIPD":       "/path/to/agipd_frame.h5",
    "JUNGFRAU_4M": "/path/to/jungfrau_frame.cxi",
    "ePix10k":     "/path/to/epix_frame.h5",
    "Eiger4M":     "/path/to/eiger_frame.h5",
}


def probe_file(detector: str, path: str) -> bool:
    """Print HDF5 tree for one file. Returns True if EXPECTED_KEY found."""
    print(f"\n{'=' * 64}")
    print(f"Detector : {detector}")
    print(f"File     : {path}")
    print(f"{'=' * 64}")

    if not Path(path).exists():
        print("  [SKIP] File not found.")
        return False

    found = False

    def visitor(name: str, obj: object) -> None:
        nonlocal found
        depth = name.count("/")
        indent = "  " + "  " * depth
        if isinstance(obj, h5py.Dataset):
            marker = "  <-- entry/data/data ✓" if name == EXPECTED_KEY else ""
            if name == EXPECTED_KEY:
                found = True
            print(f"{indent}{name}  shape={obj.shape}  dtype={obj.dtype}{marker}")
        else:
            print(f"{indent}{name}/")

    with h5py.File(path, "r") as f:
        f.visititems(visitor)

    status = "FOUND ✓" if found else "NOT FOUND ✗ — update io.py and _CONFIRMED_KEYS in test_io.py"
    print(f"\n  entry/data/data: {status}")
    return found


def main() -> None:
    files = DETECTOR_FILES

    # Override via CLI: python probe_hdf5.py AGIPD=/path/to/file.h5 ...
    if len(sys.argv) > 1:
        files = {}
        for arg in sys.argv[1:]:
            det, _, path = arg.partition("=")
            files[det.strip()] = path.strip()

    results = {det: probe_file(det, path) for det, path in files.items()}

    print(f"\n{'=' * 64}")
    print("Summary")
    print(f"{'=' * 64}")
    for det, ok in results.items():
        status = "OK" if ok else "ACTION REQUIRED"
        print(f"  {det:15s}: {status}")

    all_ok = all(results.values())
    if all_ok:
        print("\nAll detectors confirmed — no changes needed to io.py.")
    else:
        print("\nSome detectors need key updates. See ACTION REQUIRED rows above.")
        print("1. Find the correct key from the tree printout above.")
        print("2. Update io.py (see Task B4).")
        print("3. Update _CONFIRMED_KEYS in tests/test_io.py.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit the probe script**

```bash
git add scripts/probe_hdf5.py
git commit -m "feat(phase3): add probe_hdf5.py to walk HDF5 key hierarchy"
```

---

### Task B4: Run the probe and update io.py if needed

**Files:**
- Possibly modify: `src/preprocessing/io.py:43`
- Possibly modify: `tests/test_io.py` (`_CONFIRMED_KEYS` dict)

- [ ] **Step 1: Fill in real file paths in `scripts/probe_hdf5.py` and run**

```bash
python scripts/probe_hdf5.py
```

Read the output. For each detector, note whether `entry/data/data` is found.

- [ ] **Step 2: If all four show FOUND — skip to Step 5**

No code changes needed. Proceed to Task B5.

- [ ] **Step 3: If any detector uses a different key — update io.py**

Replace the hardcoded key lookup in `src/preprocessing/io.py` (line ~43) with a per-detector key map. The current code is:

```python
    with h5py.File(path, "r") as f:
        return f["entry/data/data"][()].astype(np.float32)
```

Replace with (fill in the correct keys from probe output):

```python
_HDF5_KEY_MAP: dict[str, str] = {
    ".h5":  "entry/data/data",   # update per probe findings
    ".cxi": "entry/data/data",   # update per probe findings
}


def _hdf5_key(path: Path) -> str:
    return _HDF5_KEY_MAP.get(path.suffix.lower(), "entry/data/data")
```

And update the reader call:

```python
    with h5py.File(path, "r") as f:
        return f[_hdf5_key(path)][()].astype(np.float32)
```

- [ ] **Step 4: Update `_CONFIRMED_KEYS` in `tests/test_io.py`**

Edit the `_CONFIRMED_KEYS` dict to match the keys confirmed by the probe:

```python
_CONFIRMED_KEYS: dict[str, str] = {
    "AGIPD":       "<confirmed key>",
    "JUNGFRAU_4M": "<confirmed key>",
    "ePix10k":     "<confirmed key>",
    "Eiger4M":     "<confirmed key>",
}
```

- [ ] **Step 5: Fill in real paths in `tests/test_io.py` and run real-data tests**

Edit `_REAL_DETECTOR_FILES` in `tests/test_io.py` to use the actual file paths, then:

```bash
pytest tests/test_io.py -v -k "real_hdf5"
```
Expected: 8 tests PASSED (not skipped).

- [ ] **Step 6: Run full suite to confirm no regressions**

```bash
pytest tests/ -v
```
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/preprocessing/io.py tests/test_io.py
git commit -m "fix(io): confirm HDF5 key per detector; update io.py and tests"
```

---

### Task B5: Update data_spec.md and PLANNING.md

**Files:**
- Modify: `docs/data_spec.md`
- Modify: `PLANNING.md`

- [ ] **Step 1: Add confirmed keys table to `docs/data_spec.md`**

Append this section to `docs/data_spec.md`:

```markdown
## Confirmed HDF5 Keys (verified 2026-05-21)

| Detector     | File format | Confirmed key         |
|--------------|-------------|----------------------|
| AGIPD        | .h5         | entry/data/data      |
| JUNGFRAU 4M  | .cxi        | entry/data/data      |
| ePix10k      | .h5         | entry/data/data      |
| Eiger4M      | .h5         | entry/data/data      |

_Update this table if a new beamtime reveals a different key convention._
```

(Fill in actual confirmed keys from probe output — replace `entry/data/data` for any that differ.)

- [ ] **Step 2: Check off PLANNING.md item**

Find and update:
```
- [ ] Confirm HDF5 key `entry/data/data` against real detector files
```
Change to:
```
- [x] Confirm HDF5 key `entry/data/data` against real detector files — see `docs/data_spec.md`
```

- [ ] **Step 3: Commit**

```bash
git add docs/data_spec.md PLANNING.md
git commit -m "docs: record confirmed HDF5 keys per detector; mark Phase 3 item complete"
```

---

### Task B6: Open PR for phase-03-hdf5-key

- [ ] **Step 1: Push and open PR targeting phase-03**

```bash
git push
gh pr create \
  --base phase-03 \
  --title "feat(phase3): HDF5 key confirmation — probe script, tests, data_spec update" \
  --body "$(cat <<'EOF'
## Summary
- Adds `scripts/probe_hdf5.py` to walk HDF5/CXI key hierarchy for all four detectors
- Adds parametrized real-data tests in `tests/test_io.py` (CI-safe, skipped when files absent)
- Updates `src/preprocessing/io.py` if any key differs from entry/data/data
- Records confirmed keys in `docs/data_spec.md`
- Checks off PLANNING.md item

## Test plan
- [ ] `python scripts/probe_hdf5.py` runs and prints summary for all four detectors
- [ ] `pytest tests/test_io.py -v -k real_hdf5` — 8 tests PASSED with real files
- [ ] `pytest tests/ -v` — full suite passes
- [ ] docs/data_spec.md has confirmed keys table

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 2: Run code review skills on the PR**

```bash
# Invoke /code-review:code-review and /requesting-code-review in Claude Code
```

---

## Part C — Close Phase 3

*(After both feature PRs are merged into phase-03)*

- [ ] Run `pytest tests/ -v` on `phase-03` — confirm full suite green
- [ ] Open phase-03 → main PR:

```bash
gh pr create \
  --base main \
  --title "Phase 3: Preprocessing complete" \
  --body "$(cat <<'EOF'
## Summary
Closes out Phase 3 (Preprocessing). All six checklist items complete:
- normalize.py (GCN + LCN)
- pipeline.py (full pipeline)
- tests/test_normalize.py
- tests/test_pipeline.py
- LCN window ablation (notebook + figures)
- HDF5 key confirmation (probe script + tests)

## Test plan
- [ ] pytest tests/ -v passes
- [ ] CI green on this PR
- [ ] PLANNING.md Phase 3 checklist fully checked off

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] Run `/code-review:code-review` and `/requesting-code-review` on the phase PR
- [ ] Update README.md: mark Phase 3 ✅ Complete, Phase 4 🔄 IN PROGRESS
