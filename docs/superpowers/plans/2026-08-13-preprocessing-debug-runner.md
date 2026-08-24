# Preprocessing Pipeline Debug Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 10-image preprocessing pipeline debug runner that samples frames across all four detectors, executes every pipeline step with per-step logging, and renders the resulting crops in a notebook.

**Architecture:** A standalone script (`scripts/run_pipeline_debug.py`) exposes a `run()` function that samples ~10 frames from the four production detector directories, executes the full pipeline step-by-step with structured console logging, and returns a list of result dicts. A companion notebook (`notebooks/pipeline_debug.ipynb`) calls `run()` and renders a 2×5 crop grid plus a summary log table. The augment functions are applied inline (not via `augment.py`) so we can log exact k, flip bits, and hole coordinates before applying each operation.

**Tech Stack:** Python 3.11, NumPy, PyTorch, h5py, matplotlib, pandas, pyFAI/pyopencl (GPU hitfinder), existing `src/` modules (io, geometry, pipeline, normalize, augment, hitfinders/gpu, data/dataset, utils/config).

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `scripts/run_pipeline_debug.py` | **Create** | Frame sampling, per-step pipeline execution, structured logging, `run()` entry point |
| `notebooks/pipeline_debug.ipynb` | **Create** | Import `run()`, 2×5 crop grid, pandas log table |

No existing files are modified.

---

## Augmentation Order (deviates from training pipeline — intentional)

The training pipeline in `AsymmetricCXIDataset` applies augmentation as: rot90 → flip → cutout → LCN. This debug runner intentionally applies: rot90 → flip → **LCN** → **cutout**. This order was requested explicitly; document it in the script docstring.

---

## Task 1: Frame-List Builder

**Files:**
- Create: `scripts/run_pipeline_debug.py` (initial skeleton + frame sampler)

- [ ] **Step 1: Create the script file with imports and frame sampler**

```python
# scripts/run_pipeline_debug.py
"""Preprocessing pipeline debug runner.

Samples ~10 frames across all four detectors and executes the full
SFX preprocessing pipeline with per-step console logging.

Augmentation order differs from training: rot90 → flip → LCN → cutout.
Cutout is applied after LCN (intentional — requested for this runner).

Usage:
    python scripts/run_pipeline_debug.py [--config <path>] [--seed <int>] [--device <str>]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from src.preprocessing.io import (
    count_frames,
    read_embedded_labels,
    read_detector_description,
    read_frame,
)
from src.preprocessing.geometry import get_assembler, get_geometry
from src.preprocessing.normalize import gcn, lcn
from src.preprocessing.pipeline import assemble_only
from src.hitfinders.gpu import GPUHitfinder
from src.data.dataset import _crop_within_margin
from src.utils.config import load_config

# Path to the pyFAI GPU hitfinder script consumed by GPUHitfinder
_GPU_PF8_SCRIPT = Path(__file__).parent.parent / "src" / "hitfinders" / "gpu_pf8.py"

# Default number of frames to sample per detector (total = 10)
_DEFAULT_N = {"AGIPD": 3, "JUNGFRAU_4M": 3, "ePix10k": 2, "Eiger4M": 2}


def _build_frame_list(
    lodo_cfg: dict,
    n_per_detector: dict[str, int],
    rng: np.random.Generator,
) -> list[tuple[str, Path, int]]:
    """Sample CXI files and frame indices for each detector.

    Returns a list of (detector_name, cxi_path, frame_idx) tuples.
    Picks one random CXI file per detector, then samples frame indices
    without replacement (seeded).
    """
    pattern = lodo_cfg.get("cxi_pattern", "compressed*.cxi")
    frame_list: list[tuple[str, Path, int]] = []

    for det_name, n in n_per_detector.items():
        det_dir = Path(lodo_cfg["detector_dirs"][det_name])
        files = sorted(det_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No files matching '{pattern}' in {det_dir}"
            )
        cxi_path = files[int(rng.integers(0, len(files)))]
        n_frames = count_frames(cxi_path)
        indices = rng.choice(n_frames, size=min(n, n_frames), replace=False)
        for idx in indices.tolist():
            frame_list.append((det_name, cxi_path, int(idx)))

    return frame_list
```

- [ ] **Step 2: Verify the file is importable**

```bash
cd /data/bioxfel/user/gihan/Hit_finder
python -c "from scripts.run_pipeline_debug import _build_frame_list; print('OK')"
```

Expected: `OK` (or an import error — fix before proceeding).

- [ ] **Step 3: Commit skeleton**

```bash
git add scripts/run_pipeline_debug.py
git commit -m "feat: add run_pipeline_debug.py skeleton with frame-list builder"
```

---

## Task 2: Step Logger

**Files:**
- Modify: `scripts/run_pipeline_debug.py`

- [ ] **Step 1: Add the `_log` helper below the imports block**

```python
def _log(step: int | None, label: str, msg: str) -> None:
    """Print a single step log line.

    Format:  [Step  N] Label          message
    """
    if step is not None:
        prefix = f"  [Step {step:2d}]"
    else:
        prefix = "          →"
    print(f"{prefix} {label:<14s} {msg}")
```

- [ ] **Step 2: Add the frame header printer**

```python
def _frame_header(frame_no: int, total: int, det: str, filename: str, frame_idx: int) -> None:
    bar = "━" * 62
    print(f"\n{bar}")
    print(f"FRAME {frame_no}/{total}  {det}  {filename}  frame {frame_idx}")
    print(bar)
```

- [ ] **Step 3: Verify both functions print correctly**

```bash
python - <<'EOF'
import sys; sys.path.insert(0, '.')
from scripts.run_pipeline_debug import _log, _frame_header
_frame_header(1, 10, "AGIPD", "compressed_0001.cxi", 42)
_log(1, "Read", "0.023s  shape=(16,512,128)  true_label=1")
_log(None, "Tensor", "(1,224,224) float32  derived_label=1")
EOF
```

Expected output:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FRAME 1/10  AGIPD  compressed_0001.cxi  frame 42
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [Step  1] Read           0.023s  shape=(16,512,128)  true_label=1
           → Tensor        (1,224,224) float32  derived_label=1
```

---

## Task 3: Single-Frame Pipeline Function

**Files:**
- Modify: `scripts/run_pipeline_debug.py`

Add `_process_frame()` below `_frame_header`. This is the core of the runner. All augmentation operations are applied inline (not via `augment.py` wrappers) so that intermediate values can be logged before each operation is applied.

- [ ] **Step 1: Add Steps 1–4 (read, geometry, assembly, hitfinder)**

```python
def _process_frame(
    frame_no: int,
    total: int,
    det_name: str,
    cxi_path: Path,
    frame_idx: int,
    hitfinder: GPUHitfinder,
    rng: np.random.Generator,
    label_key: str = "entry_1/labels/hit",
) -> dict[str, Any]:
    """Run the full preprocessing pipeline on one frame and return a result dict."""
    _frame_header(frame_no, total, det_name, cxi_path.name, frame_idx)
    result: dict[str, Any] = {
        "frame_no": frame_no,
        "detector": det_name,
        "file": cxi_path.name,
        "file_path": cxi_path,
        "frame_idx": frame_idx,
    }

    # ── Step 1: Read ────────────────────────────────────────────
    t0 = time.perf_counter()
    frame = read_frame(cxi_path, frame_idx)
    labels = read_embedded_labels(cxi_path, label_key)
    true_label = int(labels[frame_idx]) if frame_idx < len(labels) else -1
    elapsed = time.perf_counter() - t0
    _log(1, "Read", f"{elapsed:.3f}s  shape={frame.shape}  dtype={frame.dtype}  true_label={true_label}")
    result.update({"true_label": true_label, "read_time_s": elapsed})

    # ── Step 2: Geometry ─────────────────────────────────────────
    desc = read_detector_description(cxi_path)
    pads = get_geometry(desc)
    assembler = get_assembler(desc)
    _log(2, "Geometry", f"{desc}  ({len(pads)} panels)")
    result["detector_desc"] = desc

    # ── Step 3: Assembly ─────────────────────────────────────────
    t0 = time.perf_counter()
    assembled = assemble_only(frame, pads, desc, assembler)
    elapsed = time.perf_counter() - t0
    _log(3, "Assembly", f"{elapsed:.3f}s  shape={assembled.shape}  min={assembled.min():.0f}  max={assembled.max():.0f}")
    result.update({
        "assembled_shape": assembled.shape,
        "assembled_range": (float(assembled.min()), float(assembled.max())),
        "assembly_time_s": elapsed,
    })

    # ── Step 4: Hitfinder ────────────────────────────────────────
    with h5py.File(cxi_path, "r") as f:
        dist       = float(f["entry_1/instrument_1/detector_1/distance"][()])
        wavelength = float(f["entry_1/instrument_1/source_1/wavelength"][()])
        pixel_size = float(f["entry_1/instrument_1/detector_1/x_pixel_size"][()])
    hitfinder.set_geometry(dist=dist, wavelength=wavelength, pixel_size=pixel_size)

    peaks = hitfinder.find_peaks(np.asarray(assembled, dtype=np.float32))
    n_peaks = len(peaks)
    has_peaks = n_peaks > 0

    mismatch = (true_label == 1 and not has_peaks) or (true_label == 0 and has_peaks)
    match_tag = "✓ label match" if not mismatch else "⚠ MISMATCH"
    _log(4, "Hitfinder", f"GPUHitfinder (gpu_pf8)  →  {n_peaks} peaks  {match_tag}")
    if mismatch:
        if true_label == 1:
            print(f"             ⚠ metadata=hit  hitfinder=no peaks")
        else:
            print(f"             ⚠ metadata=non-hit  hitfinder={n_peaks} peaks")
    result.update({"n_peaks": n_peaks, "hitfinder_name": "GPUHitfinder (gpu_pf8)", "label_mismatch": mismatch})
```

- [ ] **Step 2: Add Steps 5–7 (GCN, pad+shift, crop decision)**

Append inside `_process_frame`, after Step 4 block:

```python
    # ── Step 5: GCN ──────────────────────────────────────────────
    assembled_gcn = gcn(assembled)
    gs = {
        "mean": float(assembled_gcn.mean()),
        "std":  float(assembled_gcn.std()),
        "min":  float(assembled_gcn.min()),
        "max":  float(assembled_gcn.max()),
    }
    _log(5, "GCN", f"μ={gs['mean']:.3f}  σ={gs['std']:.3f}  min={gs['min']:.2f}  max={gs['max']:.2f}")
    result["gcn_stats"] = gs

    # ── Step 6: Pad + shift centroids ────────────────────────────
    PAD = 112
    padded = np.pad(assembled_gcn, PAD, mode="constant", constant_values=0.0)
    shifted = peaks + PAD if has_peaks else peaks  # (N,2) [x,y] in padded coords
    ph, pw = padded.shape
    _log(6, "Pad+shift", f"padded={padded.shape}  centroids shifted +{PAD}px")

    # ── Step 7: Crop decision ─────────────────────────────────────
    crop: np.ndarray
    derived_label: int
    decision: str

    if has_peaks:
        coin = int(rng.integers(0, 2))          # 0 = non-hit path, 1 = hit path
        if coin == 1:
            # Path A — hit crop centred on a random Bragg peak
            peak_i = int(rng.integers(0, len(shifted)))
            cx = int(round(float(shifted[peak_i, 0])))
            cy = int(round(float(shifted[peak_i, 1])))
            left = int(np.clip(cx - 112, 0, pw - 224))
            top  = int(np.clip(cy - 112, 0, ph - 224))
            crop = padded[top:top + 224, left:left + 224].copy()
            derived_label = 1
            decision = f"coin=HIT  peak=({cx},{cy})→crop_tl=({top},{left})"
        else:
            # Path B — non-hit hard-negative with 50 px clearance
            crop_found = False
            top = left = 0
            for _ in range(50):
                top  = int(rng.integers(0, max(1, ph - 224 + 1)))
                left = int(rng.integers(0, max(1, pw - 224 + 1)))
                if not _crop_within_margin(top, left, 224, shifted, margin=50):
                    crop = padded[top:top + 224, left:left + 224].copy()
                    crop_found = True
                    break
            if not crop_found:
                top = left = 0
                crop = padded[:224, :224].copy()
                decision = "coin=NON-HIT  fallback (50 attempts exhausted)"
            else:
                decision = f"coin=NON-HIT  random crop_tl=({top},{left})"
            derived_label = 0
    else:
        # No peaks — forced random non-hit crop (no clearance needed)
        top  = int(rng.integers(0, max(1, ph - 224 + 1)))
        left = int(rng.integers(0, max(1, pw - 224 + 1)))
        crop = padded[top:top + 224, left:left + 224].copy()
        derived_label = 0
        decision = f"forced-non-hit  random crop_tl=({top},{left})"

    _log(7, "Crop", f"{decision}  label={derived_label}")
    result.update({"crop_decision": decision, "crop_top_left": (top, left), "derived_label": derived_label})
```

- [ ] **Step 3: Add Steps 8–10 (augment geometric, LCN, cutout) and return**

Append inside `_process_frame`, after Step 7 block:

```python
    # ── Step 8: Augment — geometric (rot90 + flip, before LCN) ──
    # Draw values explicitly so we can log them before applying.
    k = int(rng.integers(0, 4))
    crop = np.rot90(crop, k=k).copy()

    bits = rng.integers(0, 2, size=2)
    h_flip, v_flip = bool(bits[0]), bool(bits[1])
    if h_flip:
        crop = np.fliplr(crop).copy()
    if v_flip:
        crop = np.flipud(crop).copy()

    flip_str = ("H" if h_flip else "-") + ("V" if v_flip else "-")
    _log(8, "Augment", f"rot90=k{k}  flip={flip_str}")
    result["augment"] = {"rot90_k": k, "flip_h": h_flip, "flip_v": v_flip}

    # ── Step 9: LCN ──────────────────────────────────────────────
    crop = lcn(crop)
    ls = {
        "mean": float(crop.mean()),
        "std":  float(crop.std()),
        "min":  float(crop.min()),
        "max":  float(crop.max()),
    }
    _log(9, "LCN", f"μ={ls['mean']:.3f}  σ={ls['std']:.3f}  min={ls['min']:.2f}  max={ls['max']:.2f}")
    result["lcn_stats"] = ls

    # ── Step 10: Cutout — applied AFTER LCN (debug runner only) ──
    # Draw hole coordinates explicitly before zeroing so we can log them.
    N_HOLES, HOLE_SIZE = 3, 32
    h_img, w_img = crop.shape
    holes: list[tuple[int, int]] = []
    for _ in range(N_HOLES):
        r = int(rng.integers(0, max(1, h_img - HOLE_SIZE + 1)))
        c = int(rng.integers(0, max(1, w_img - HOLE_SIZE + 1)))
        crop[r : r + HOLE_SIZE, c : c + HOLE_SIZE] = 0.0
        holes.append((r, c))
    holes_str = "  ".join(f"({r},{c})" for r, c in holes)
    _log(10, "Cutout", f"3×32² holes at {holes_str}")
    result["cutout_holes"] = holes

    # ── Final tensor ──────────────────────────────────────────────
    tensor = torch.from_numpy(np.ascontiguousarray(crop)).unsqueeze(0).float()
    _log(None, "Tensor", f"shape={tuple(tensor.shape)}  dtype={tensor.dtype}  derived_label={derived_label}")
    result["tensor"] = tensor

    return result
```

- [ ] **Step 4: Commit the per-frame pipeline**

```bash
git add scripts/run_pipeline_debug.py
git commit -m "feat: add _process_frame() with 10-step per-frame logging"
```

---

## Task 4: `run()` Entry Point and `__main__` Block

**Files:**
- Modify: `scripts/run_pipeline_debug.py`

- [ ] **Step 1: Add `run()` public function**

Append below `_process_frame`:

```python
def run(
    config_path: str = "configs/supervised/resnet18_asymmetric.yaml",
    n_per_detector: dict[str, int] | None = None,
    seed: int = 42,
    device: str = "cuda",
) -> list[dict[str, Any]]:
    """Run the preprocessing debug pipeline on ~10 frames across all detectors.

    Args:
        config_path: Path to YAML config (must have lodo.detector_dirs and
            lodo.cxi_pattern keys — resnet18_asymmetric.yaml satisfies this).
        n_per_detector: Frames per detector. Defaults to {AGIPD:3, JF:3, ePix:2, Eiger:2}.
        seed: Random seed for frame sampling and per-step decisions.
        device: PyTorch/OpenCL device for the GPU hitfinder ("cuda" or "cpu").

    Returns:
        List of result dicts, one per frame processed. Each dict contains:
        frame_no, detector, file, file_path, frame_idx, true_label, detector_desc,
        assembled_shape, assembled_range, assembly_time_s, n_peaks, hitfinder_name,
        label_mismatch, gcn_stats, crop_decision, crop_top_left, derived_label,
        augment, lcn_stats, cutout_holes, tensor.
    """
    if n_per_detector is None:
        n_per_detector = _DEFAULT_N.copy()

    cfg = load_config(config_path)
    lodo_cfg = cfg["lodo"]
    label_key = lodo_cfg.get("label_key", "entry_1/labels/hit")

    rng_sample = np.random.default_rng(seed)
    frame_list = _build_frame_list(lodo_cfg, n_per_detector, rng_sample)

    bar = "═" * 62
    print(f"\n{bar}")
    print(f"SFX PREPROCESSING DEBUG RUN — {len(frame_list)} frames  seed={seed}  device={device}")
    print(bar)
    for i, (det, path, idx) in enumerate(frame_list, 1):
        print(f"  [{i:2d}] {det:<14s} {path.name}  frame {idx}")
    print()

    hitfinder = GPUHitfinder(script_path=_GPU_PF8_SCRIPT, device=device)

    results: list[dict[str, Any]] = []
    for i, (det_name, cxi_path, frame_idx) in enumerate(frame_list, 1):
        result = _process_frame(
            frame_no=i,
            total=len(frame_list),
            det_name=det_name,
            cxi_path=cxi_path,
            frame_idx=frame_idx,
            hitfinder=hitfinder,
            rng=np.random.default_rng(seed + i),
            label_key=label_key,
        )
        results.append(result)

    mismatches = sum(r["label_mismatch"] for r in results)
    print(f"\n{bar}")
    print(f"DONE  {len(results)} frames  {mismatches} label mismatches")
    print(bar)

    return results
```

- [ ] **Step 2: Add `__main__` block**

Append at the very end of the file:

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFX preprocessing pipeline debug runner")
    parser.add_argument("--config", default="configs/supervised/resnet18_asymmetric.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(config_path=args.config, seed=args.seed, device=args.device)
```

- [ ] **Step 3: Smoke-test the script (dry import)**

```bash
cd /data/bioxfel/user/gihan/Hit_finder
python -c "from scripts.run_pipeline_debug import run; print('run() importable')"
```

Expected: `run() importable`

- [ ] **Step 4: Commit**

```bash
git add scripts/run_pipeline_debug.py
git commit -m "feat: add run() entry point and __main__ CLI to preprocessing debug runner"
```

---

## Task 5: Notebook

**Files:**
- Create: `notebooks/pipeline_debug.ipynb`

- [ ] **Step 1: Create the notebook**

Create `notebooks/pipeline_debug.ipynb` with the following cells. Use the NotebookEdit tool or write the raw JSON directly.

**Cell 1 — Configuration**
```python
import sys
sys.path.insert(0, "..")

CONFIG   = "../configs/supervised/resnet18_asymmetric.yaml"
SEED     = 42
DEVICE   = "cuda"   # change to "cpu" if no GPU available
```

**Cell 2 — Run pipeline**
```python
from scripts.run_pipeline_debug import run

results = run(config_path=CONFIG, seed=SEED, device=DEVICE)
print(f"\nReturned {len(results)} result dicts.")
```

**Cell 3 — Crop grid (2 rows × 5 cols)**
```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.flatten()

for ax, r in zip(axes, results):
    crop_np = r["tensor"].squeeze(0).numpy()
    vmin, vmax = np.percentile(crop_np, [2, 98])
    ax.imshow(crop_np, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")

    label_color = "limegreen" if r["derived_label"] == 1 else "tomato"
    mismatch_tag = " ⚠" if r["label_mismatch"] else ""
    title = (
        f"F{r['frame_no']}  {r['detector']}\n"
        f"peaks={r['n_peaks']}  label={r['derived_label']}{mismatch_tag}"
    )
    ax.set_title(title, fontsize=9, color=label_color)
    ax.axis("off")

plt.suptitle("Preprocessing Debug Run — 10 Frames (all detectors)", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()
```

**Cell 4 — Log table**
```python
import pandas as pd

rows = []
for r in results:
    rows.append({
        "frame": r["frame_no"],
        "detector": r["detector"],
        "file": r["file"],
        "frame_idx": r["frame_idx"],
        "true_label": r["true_label"],
        "n_peaks": r["n_peaks"],
        "mismatch": "⚠" if r["label_mismatch"] else "✓",
        "crop": r["crop_decision"].split("  ")[0],
        "derived_label": r["derived_label"],
        "rot90_k": r["augment"]["rot90_k"],
        "flip": ("H" if r["augment"]["flip_h"] else "-") + ("V" if r["augment"]["flip_v"] else "-"),
        "gcn_max": round(r["gcn_stats"]["max"], 2),
        "lcn_std": round(r["lcn_stats"]["std"], 3),
        "assemble_ms": round(r["assembly_time_s"] * 1000, 1),
    })

df = pd.DataFrame(rows).set_index("frame")
pd.set_option("display.max_colwidth", 30)
display(df)
```

- [ ] **Step 2: Verify notebook executes without error**

In JupyterLab or via:
```bash
cd /data/bioxfel/user/gihan/Hit_finder/notebooks
jupyter nbconvert --to notebook --execute pipeline_debug.ipynb --output pipeline_debug_executed.ipynb 2>&1 | tail -20
```

Expected: no `Error` or `Traceback` lines in output.

- [ ] **Step 3: Commit**

```bash
git add notebooks/pipeline_debug.ipynb
git commit -m "feat: add pipeline_debug notebook with crop grid and log table"
```

---

## Spec Self-Review

**Coverage check:**
- ✅ 10 frames across 4 detectors (3+3+2+2)
- ✅ Step 1: Read + true_label from metadata
- ✅ Step 2: Geometry selection logged
- ✅ Step 3: Assembly + timing
- ✅ Step 4: GPU hitfinder (GPUHitfinder → gpu_pf8.py), peak count, mismatch alert
- ✅ Step 5: GCN stats
- ✅ Step 6: Pad + centroid shift
- ✅ Step 7: Crop decision (coin flip when peaks found; forced non-hit when no peaks)
- ✅ Step 8: Geometric augment (rot90 k logged, flip H/V logged)
- ✅ Step 9: LCN stats
- ✅ Step 10: Cutout AFTER LCN, hole coordinates logged
- ✅ Notebook: 2×5 crop grid + pandas log table
- ✅ Both script CLI and notebook `run()` import path

**Placeholder scan:** None found.

**Type consistency:** `_build_frame_list` returns `list[tuple[str, Path, int]]`; `run()` iterates it as `(det_name, cxi_path, frame_idx)` — consistent. `_process_frame` receives those same types — consistent. Result dict keys are the same across Tasks 3 and 4 — consistent.
