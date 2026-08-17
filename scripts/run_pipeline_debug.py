# scripts/run_pipeline_debug.py
"""Preprocessing pipeline debug runner.

Samples ~10 frames across all four detectors and executes the full
SFX preprocessing pipeline with per-step console logging.

NOTE: Augmentation order intentionally differs from the training pipeline.
Training order: rot90 → flip → cutout → LCN.
This runner's order: rot90 → flip → LCN → cutout (cutout applied after LCN).
This is a deliberate deviation for debugging purposes — results from this
runner are NOT directly representative of the training data distribution.

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
from src.preprocessing.augment import pad_border
from src.preprocessing.pipeline import _to_2d, assemble_only
from src.hitfinders.gpu import GPUHitfinder
from src.data.dataset import _crop_within_margin
from src.utils.config import load_config

# Path to the pyFAI GPU hitfinder script consumed by GPUHitfinder
_GPU_PF8_SCRIPT = Path(__file__).parent.parent / "src" / "hitfinders" / "gpu_pf8.py"

# Default number of frames to sample per detector (total = 10)
_DEFAULT_N: dict[str, int] = {"AGIPD": 3, "JUNGFRAU_4M": 3, "ePix10k": 2, "Eiger4M": 2}


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
            raise FileNotFoundError(f"No files matching '{pattern}' in {det_dir}")
        cxi_path = files[int(rng.integers(0, len(files)))]
        n_frames = count_frames(cxi_path)
        indices = rng.choice(n_frames, size=min(n, n_frames), replace=False)
        for idx in indices.tolist():
            frame_list.append((det_name, cxi_path, int(idx)))

    return frame_list


# ── Task 2: logging helpers ────────────────────────────────────────────────────


def _log(step: int | None, label: str, msg: str) -> None:
    """Print a single step log line: [Step  N] Label    message"""
    prefix = f"  [Step {step:2d}]" if step is not None else "          →"
    print(f"{prefix} {label:<14s} {msg}")


def _frame_header(
    frame_no: int, total: int, det: str, filename: str, frame_idx: int
) -> None:
    bar = "━" * 62
    print(f"\n{bar}")
    print(f"FRAME {frame_no}/{total}  {det}  {filename}  frame {frame_idx}")
    print(bar)


# ── Task 3: per-frame pipeline ─────────────────────────────────────────────────


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

    # ── Step 1: Read ───────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    frame = read_frame(cxi_path, frame_idx)
    labels = read_embedded_labels(cxi_path, label_key)
    true_label = int(labels[frame_idx]) if frame_idx < len(labels) else -1
    elapsed = time.perf_counter() - t0
    _log(
        1,
        "Read",
        f"{elapsed:.3f}s  shape={frame.shape}  dtype={frame.dtype}  true_label={true_label}",
    )
    result.update({"true_label": true_label, "read_time_s": elapsed})

    # ── Step 2: Geometry ───────────────────────────────────────────────────────
    desc = read_detector_description(cxi_path)
    if desc == "Jungfrau 4M":
        _log(2, "Geometry", f"{desc}  (pre-assembled — skipping Reborn assembly)")
        pads = assembler = None
    else:
        pads = get_geometry(desc)
        assembler = get_assembler(desc)
        _log(2, "Geometry", f"{desc}  ({len(pads)} panels)")
    result["detector_desc"] = desc

    # ── Step 3: Assembly ───────────────────────────────────────────────────────
    t0 = time.perf_counter()
    if desc == "Jungfrau 4M":
        assembled = _to_2d(frame)
    else:
        assembled = assemble_only(frame, pads, desc, assembler)
    elapsed = time.perf_counter() - t0
    _log(
        3,
        "Assembly",
        f"{elapsed:.3f}s  shape={assembled.shape}  min={assembled.min():.0f}  max={assembled.max():.0f}",
    )
    result.update(
        {
            "assembled_shape": assembled.shape,
            "assembled_range": (float(assembled.min()), float(assembled.max())),
            "assembly_time_s": elapsed,
        }
    )

    # ── Step 4: Hitfinder ──────────────────────────────────────────────────────
    with h5py.File(cxi_path, "r") as f:
        dist = float(f["entry_1/instrument_1/detector_1/distance"][()])
        wavelength = float(f["entry_1/instrument_1/source_1/wavelength"][()])
        pixel_size = float(f["entry_1/instrument_1/detector_1/x_pixel_size"][()])
    hitfinder.set_geometry(dist=dist, wavelength=wavelength, pixel_size=pixel_size)

    peaks = hitfinder.find_peaks(np.ascontiguousarray(assembled, dtype=np.float32))
    n_peaks = len(peaks)
    has_peaks = n_peaks > 0

    mismatch = (true_label == 1 and not has_peaks) or (true_label == 0 and has_peaks)
    match_tag = "✓ label match" if not mismatch else "⚠ MISMATCH"
    _log(4, "Hitfinder", f"GPUHitfinder (gpu_pf8)  →  {n_peaks} peaks  {match_tag}")
    if mismatch:
        if true_label == 1:
            print("             ⚠ metadata=hit  hitfinder=no peaks")
        else:
            print(f"             ⚠ metadata=non-hit  hitfinder={n_peaks} peaks")
    result.update(
        {
            "n_peaks": n_peaks,
            "hitfinder_name": "GPUHitfinder (gpu_pf8)",
            "label_mismatch": mismatch,
        }
    )

    # ── Step 5: GCN ───────────────────────────────────────────────────────────
    assembled_gcn = gcn(assembled)
    gs = {
        "mean": float(assembled_gcn.mean()),
        "std": float(assembled_gcn.std()),
        "min": float(assembled_gcn.min()),
        "max": float(assembled_gcn.max()),
    }
    _log(
        5,
        "GCN",
        f"μ={gs['mean']:.3f}  σ={gs['std']:.3f}  min={gs['min']:.2f}  max={gs['max']:.2f}",
    )
    result["gcn_stats"] = gs

    # ── Step 6: Pad + shift centroids ─────────────────────────────────────────
    PAD = 112
    padded = pad_border(assembled_gcn, PAD)
    shifted = (peaks + PAD).astype(np.float32) if has_peaks else peaks
    ph, pw = padded.shape
    _log(6, "Pad+shift", f"padded={padded.shape}  centroids shifted +{PAD}px")

    # ── Step 7: Crop decision ──────────────────────────────────────────────────
    derived_label: int
    decision: str
    top = left = 0

    if has_peaks:
        coin = int(rng.integers(0, 2))  # 0 = non-hit path, 1 = hit path
        if coin == 1:
            # Path A — hit crop centred on a random Bragg peak.
            # peaks[:, 0] = x (col), peaks[:, 1] = y (row) — GPUHitfinder/gpu_pf8 convention.
            peak_i = int(rng.integers(0, len(shifted)))
            cx = int(round(float(shifted[peak_i, 0])))  # x = column
            cy = int(round(float(shifted[peak_i, 1])))  # y = row
            left = int(np.clip(cx - 112, 0, pw - 224))
            top = int(np.clip(cy - 112, 0, ph - 224))
            crop: np.ndarray = padded[top : top + 224, left : left + 224].copy()
            derived_label = 1
            decision = f"coin=HIT  peak=({cx},{cy})→crop_tl=({top},{left})"
        else:
            # Path B — non-hit hard-negative with 50 px clearance from all peaks
            crop_found = False
            for _ in range(50):
                top = int(rng.integers(0, max(1, ph - 224 + 1)))
                left = int(rng.integers(0, max(1, pw - 224 + 1)))
                if not _crop_within_margin(top, left, 224, shifted, margin=50):
                    crop = padded[top : top + 224, left : left + 224].copy()
                    crop_found = True
                    break
            if not crop_found:
                top = left = 0
                crop = padded[:224, :224].copy()
                decision = "coin=NON-HIT  fallback (50 attempts exhausted)"
            else:
                decision = f"coin=NON-HIT  crop_tl=({top},{left})"
            derived_label = 0
    else:
        # No peaks — forced random non-hit crop (no clearance needed)
        top = int(rng.integers(0, max(1, ph - 224 + 1)))
        left = int(rng.integers(0, max(1, pw - 224 + 1)))
        crop = padded[top : top + 224, left : left + 224].copy()
        derived_label = 0
        decision = f"forced-non-hit  crop_tl=({top},{left})"

    _log(7, "Crop", f"{decision}  label={derived_label}")
    result.update(
        {
            "crop_decision": decision,
            "crop_top_left": (top, left),
            "derived_label": derived_label,
        }
    )

    # ── Step 8: Augment — rot90 + flip (before LCN) ───────────────────────────
    # Draw values explicitly so they can be logged before application.
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

    # ── Step 9: LCN ───────────────────────────────────────────────────────────
    result["crop_pre_lcn"] = crop.copy()  # for the eps ablation in the notebook
    crop = lcn(crop)
    ls = {
        "mean": float(crop.mean()),
        "std": float(crop.std()),
        "min": float(crop.min()),
        "max": float(crop.max()),
    }
    _log(
        9,
        "LCN",
        f"μ={ls['mean']:.3f}  σ={ls['std']:.3f}  min={ls['min']:.2f}  max={ls['max']:.2f}",
    )
    result["lcn_stats"] = ls

    # ── Step 10: Cutout — applied AFTER LCN (debug runner only) ───────────────
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

    # ── Final tensor ───────────────────────────────────────────────────────────
    tensor = torch.from_numpy(np.ascontiguousarray(crop)).unsqueeze(0).float()
    _log(
        None,
        "Tensor",
        f"shape={tuple(tensor.shape)}  dtype={tensor.dtype}  derived_label={derived_label}",
    )
    result["tensor"] = tensor

    return result


# ── Task 4: public entry point + CLI ──────────────────────────────────────────


def run(
    config_path: str = "configs/supervised/resnet18_asymmetric.yaml",
    n_per_detector: dict[str, int] | None = None,
    seed: int = 42,
    device: str = "cuda",
) -> list[dict[str, Any]]:
    """Run the preprocessing debug pipeline on ~10 frames across all detectors.

    Args:
        config_path: Path to YAML config with lodo.detector_dirs and lodo.cxi_pattern.
        n_per_detector: Frames per detector. Defaults to {AGIPD:3, JF:3, ePix:2, Eiger:2}.
        seed: Random seed for frame sampling and per-step decisions.
        device: OpenCL device for GPU hitfinder ("cuda" or "cpu").

    Returns:
        List of result dicts, one per frame. Each dict contains: frame_no, detector,
        file, file_path, frame_idx, true_label, detector_desc, assembled_shape,
        assembled_range, assembly_time_s, n_peaks, hitfinder_name, label_mismatch,
        gcn_stats, crop_decision, crop_top_left, derived_label, augment, lcn_stats,
        cutout_holes, tensor.
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
    print(
        f"SFX PREPROCESSING DEBUG RUN — {len(frame_list)} frames  seed={seed}  device={device}"
    )
    print(bar)
    for i, (det, path, idx) in enumerate(frame_list, 1):
        print(f"  [{i:2d}] {det:<14s} {path.name}  frame {idx}")

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
    print(f"DONE  {len(results)} frames processed  {mismatches} label mismatches")
    print(bar)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SFX preprocessing pipeline debug runner"
    )
    parser.add_argument(
        "--config", default="configs/supervised/resnet18_asymmetric.yaml"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(config_path=args.config, seed=args.seed, device=args.device)
