"""Visualise geometry-assembled diffraction patterns before normalisation.

Reads one frame from a CXI file, assembles it into a spatially correct 2D
image, and saves a PNG.  No GCN, LCN, or resize is applied — the output shows
exactly what enters the normalisation pipeline.

Assembly strategy (confirmed by visual inspection 2026-06-26):
  AGIPD 1M     — Reborn standard pads + PADAssembler(frame.ravel())
  ePix10k 2.2M — Reborn standard pads + PADAssembler(frame.ravel())
  EIGER 4M     — CrystFEL geom pads  + PADAssembler(concat panel ravels)
  Jungfrau 4M  — pre-assembled canvas, passed through _to_2d directly

Usage:
    python scripts/visualize_assembled.py <cxi_path> [--frame N] [--out path.png] [--vmax V]

Examples:
    python scripts/visualize_assembled.py /data/.../agipd_20k/compressed0.cxi
    python scripts/visualize_assembled.py file.cxi --frame 5 --out agipd_frame5.png
    python scripts/visualize_assembled.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from reborn.detector import PADAssembler

from src.preprocessing.geometry import extract_panels_from_canvas, get_assembler, get_geometry
from src.preprocessing.io import read_detector_description, read_frame
from src.preprocessing.pipeline import _to_2d

DATA_ROOT = Path("/data/bioxfel/user/gihan/Resonet/production")

PRODUCTION_FILES: dict[str, Path] = {
    "AGIPD":       DATA_ROOT / "agipd_20k"    / "compressed0.cxi",
    "JUNGFRAU_4M": DATA_ROOT / "jungfrau_20k" / "compressed0.cxi",
    "ePix10k":     DATA_ROOT / "epix10k_20k"  / "compressed0.cxi",
    "Eiger4M":     DATA_ROOT / "eiger4m_20k"  / "compressed0.cxi",
}


def assemble_raw(cxi_path: Path, frame_idx: int) -> tuple[np.ndarray, str]:
    """Return (assembled_2d, detector_desc) without normalisation or resize."""
    raw = read_frame(cxi_path, frame_idx).astype(np.float32)
    desc = read_detector_description(cxi_path)

    if desc == "Jungfrau 4M":
        return _to_2d(raw), desc

    pads = get_geometry(desc)
    asm  = get_assembler(desc)

    if pads.defines_slicing():
        # Canvas-based detectors (e.g. EigerRESoNeT, JUNGFRAU_4M): extract panels
        # via parent_data_slice before passing to PADAssembler.
        panels = extract_panels_from_canvas(raw, pads)
        flat = np.concatenate([p.ravel() for p in panels])
    else:
        # Reborn standard pads (AGIPD, ePix10k, Eiger4M): raw ravel matches
        # PADAssembler's flat_indices order directly.
        flat = raw.ravel()

    assembled = asm.assemble_data(flat)
    return assembled.astype(np.float32), desc


def save_image(
    image: np.ndarray,
    desc: str,
    frame_idx: int,
    out_path: Path,
    vmax: float | None,
) -> None:
    pos = image[image > 0]
    p_low  = np.percentile(pos, 1)  if len(pos) else 0.0
    p_high = np.percentile(pos, 99) if len(pos) else 1.0
    vmin_auto = max(0.0, float(p_low))
    vmax_auto = float(p_high) if vmax is None else vmax

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    im = ax.imshow(
        image,
        cmap="inferno",
        vmin=vmin_auto,
        vmax=vmax_auto,
        origin="upper",
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Intensity (ADU)")
    ax.set_title(
        f"{desc} — frame {frame_idx}\n"
        f"assembled shape: {image.shape}  |  "
        f"vmin={vmin_auto:.0f}  vmax={vmax_auto:.0f}",
        fontsize=9,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}  (shape={image.shape})")


def process_one(cxi_path: Path, frame_idx: int, out_path: Path | None, vmax: float | None) -> None:
    print(f"\n{cxi_path.name}  frame={frame_idx}")
    image, desc = assemble_raw(cxi_path, frame_idx)
    print(f"  detector:        {desc}")
    print(f"  assembled shape: {image.shape}")
    print(f"  intensity range: [{image.min():.1f}, {image.max():.1f}]")

    if out_path is None:
        stem = f"{cxi_path.stem}_frame{frame_idx}_{desc.replace(' ', '_')}"
        out_path = Path(stem + ".png")

    save_image(image, desc, frame_idx, out_path, vmax)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("cxi_path", nargs="?", type=Path, help="Path to a CXI file")
    parser.add_argument("--frame", type=int, default=0, help="Frame index (default: 0)")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path")
    parser.add_argument("--vmax", type=float, default=None, help="Colour scale upper limit")
    parser.add_argument("--all", action="store_true", help="Process all four production files")
    args = parser.parse_args()

    if args.all:
        for label, path in PRODUCTION_FILES.items():
            if not path.exists():
                print(f"  SKIP {label}: {path} not found")
                continue
            out = Path(f"assembled_{label.lower()}_frame{args.frame}.png")
            process_one(path, args.frame, out, args.vmax)
    elif args.cxi_path is not None:
        process_one(args.cxi_path, args.frame, args.out, args.vmax)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
