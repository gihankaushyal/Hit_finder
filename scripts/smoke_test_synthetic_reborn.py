"""Smoke test: run one synthetic Eiger-like frame through the FULL pipeline.

Constructs a synthetic PADGeometry from the geom metadata stored in the HDF5
file so the complete Reborn assembly path is exercised:
    assemble_image() → _to_2d() → GCN → LCN → resize 224×224

Usage:
    conda run -n sfx-hitfinder python3 scripts/smoke_test_synthetic_reborn.py
"""

import sys
from pathlib import Path

import h5py
import numpy as np
from reborn.detector import PADGeometry, PADGeometryList

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.pipeline import preprocess

DATA_FILE = Path("/data/bioxfel/user/gihan/Resonet/hitfinder_10k/compressed0.h5")

# geom column indices (from commandline --randDist --randWave --randCent)
_GEOM_DIST_MM = 0   # detector distance in mm
_GEOM_WAVE_AA = 1   # wavelength in Ångströms (unused here)
_GEOM_PIX_MM  = 2   # pixel size in mm (always 0.075 for Eiger)


def build_pad(distance_m: float, pixel_size_m: float, n_ss: int, n_fs: int) -> PADGeometry:
    """Build a flat, on-axis PADGeometry for a monolithic single-panel detector."""
    return PADGeometry(
        distance=distance_m,
        pixel_size=pixel_size_m,
        shape=(n_ss, n_fs),
    )


def main() -> None:
    print(f"Source: {DATA_FILE}")

    with h5py.File(DATA_FILE, "r") as f:
        frame = f["images"][0].astype(np.float32)
        geom_row = f["geom"][0]

    n_ss, n_fs = frame.shape
    distance_m = float(geom_row[_GEOM_DIST_MM]) * 1e-3
    pixel_size_m = float(geom_row[_GEOM_PIX_MM]) * 1e-3

    print(f"Input  — shape: {frame.shape}  dtype: {frame.dtype}  min: {frame.min():.1f}  max: {frame.max():.1f}")
    print(f"Geometry — n_ss: {n_ss}  n_fs: {n_fs}  pixel_size: {pixel_size_m*1e6:.1f} μm  distance: {distance_m*1e3:.1f} mm")

    pad = build_pad(distance_m, pixel_size_m, n_ss, n_fs)
    pads = PADGeometryList([pad])

    output = preprocess(panel_data=[frame], pads=pads)

    finite = np.isfinite(output).all()
    print(f"Output — shape: {output.shape}  dtype: {output.dtype}  min: {output.min():.4f}  max: {output.max():.4f}  mean: {output.mean():.4f}  all_finite: {finite}")

    if not finite:
        print("FAIL: output contains NaN or Inf")
        sys.exit(1)
    if output.shape != (224, 224):
        print(f"FAIL: expected shape (224, 224), got {output.shape}")
        sys.exit(1)

    print("PASS")


if __name__ == "__main__":
    main()
