"""Smoke test: run one synthetic Eiger4M frame through GCN → LCN → resize.

Images in the hitfinder_10k files are already assembled (512×512, uint16),
so the Reborn geometry step is skipped.

Usage:
    python scripts/smoke_test_synthetic.py
"""

import sys
from pathlib import Path

import h5py
import numpy as np
from skimage.transform import resize as sk_resize

# Allow imports from src/ without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.normalize import LCN_WINDOW_DEFAULT, gcn, lcn

DATA_FILE = Path("/data/bioxfel/user/gihan/Resonet/hitfinder_10k/compressed0.h5")
TARGET_SIZE = (224, 224)


def main() -> None:
    print(f"Source: {DATA_FILE}")

    with h5py.File(DATA_FILE, "r") as f:
        raw = f["images"][0].astype(np.float32)

    print(
        f"Input  — shape: {raw.shape}  dtype: {raw.dtype}  min: {raw.min():.1f}  max: {raw.max():.1f}"
    )

    image_gcn = gcn(raw)
    image_lcn = lcn(image_gcn, window=LCN_WINDOW_DEFAULT)
    output = sk_resize(
        image_lcn, TARGET_SIZE, anti_aliasing=True, preserve_range=True
    ).astype(np.float32)

    finite = np.isfinite(output).all()
    print(
        f"Output — shape: {output.shape}  dtype: {output.dtype}  min: {output.min():.4f}  max: {output.max():.4f}  mean: {output.mean():.4f}  all_finite: {finite}"
    )

    if not finite:
        print("FAIL: output contains NaN or Inf")
        sys.exit(1)
    if output.shape != TARGET_SIZE:
        print(f"FAIL: expected shape {TARGET_SIZE}, got {output.shape}")
        sys.exit(1)

    print("PASS")


if __name__ == "__main__":
    main()
