"""Smoke test: raw shape, assembled shape, and preprocessed output for each detector.

Reads frame 0 from one CXI file per detector, prints:
  - raw shape (detector-native, from read_frame)
  - assembled shape (intermediate 2D image before GCN/LCN/resize)
  - output shape (should be 224×224 for all)
  - which preprocessing path was taken (geometry-aware vs assembled)

Usage:
    python scripts/smoke_test_detector_shapes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.preprocessing.geometry import (
    assemble_image,
    extract_panels_from_canvas,
    get_geometry,
)
from src.preprocessing.io import read_detector_description, read_frame
from src.preprocessing.augment import PAD_BORDER_DEFAULT, pad_border
from src.preprocessing.pipeline import _to_2d

DATA_ROOT = Path("/data/bioxfel/user/gihan/Resonet/production")

DETECTORS = {
    "AGIPD": DATA_ROOT / "agipd_20k" / "compressed0.cxi",
    "JUNGFRAU_4M": DATA_ROOT / "jungfrau_20k" / "compressed0.cxi",
    "ePix10k": DATA_ROOT / "epix10k_20k" / "compressed0.cxi",
    "Eiger4M": DATA_ROOT / "eiger4m_20k" / "compressed0.cxi",
}

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def _assemble_intermediate(raw: np.ndarray, desc: str) -> tuple[tuple[int, ...], str]:
    """Return (assembled_shape, path_label) without running GCN/LCN/resize."""
    if desc == "Jungfrau 4M":
        assembled = _to_2d(raw)
        return assembled.shape, "assembled"

    pads = get_geometry(desc)
    if desc == "AGIPD 1M":
        panels = [
            raw[m, a * 64 : (a + 1) * 64, :].astype(np.float32)
            for m in range(16)
            for a in range(8)
        ]
    else:
        panels = extract_panels_from_canvas(raw.astype(np.float32), pads)

    assembled = assemble_image(pads, panels)
    assembled_2d = _to_2d(assembled)
    return assembled_2d.shape, "geometry"


def run():
    all_passed = True

    header = f"\n{'Detector':<14}  {'Raw shape':<22}  {'Assembled shape':<18}  {'Output':<10}  {'Path':<10}  Result"
    print(header)
    print("-" * len(header))

    for detector, cxi_path in DETECTORS.items():
        try:
            raw = read_frame(cxi_path, frame_idx=0)
            raw_shape = raw.shape

            desc = read_detector_description(cxi_path)

            assembled_shape, path_label = _assemble_intermediate(raw, desc)

            # Assembly + symmetric border padding (new pipeline: no resize to 224)
            pads = get_geometry(desc) if desc != "Jungfrau 4M" else None
            if pads is not None:
                from src.preprocessing.geometry import get_assembler
                from src.preprocessing.pipeline import assemble_only
                assembler = get_assembler(desc)
                assembled_raw = assemble_only(raw, pads, desc, assembler=assembler)
            else:
                assembled_raw = _to_2d(raw)
            padded = pad_border(assembled_raw)
            out_shape = padded.shape
            expected = (
                assembled_shape[0] + 2 * PAD_BORDER_DEFAULT,
                assembled_shape[1] + 2 * PAD_BORDER_DEFAULT,
            )
            ok = out_shape == expected
            status = PASS if ok else FAIL
            if not ok:
                all_passed = False

            print(
                f"{detector:<14}  {str(raw_shape):<22}  {str(assembled_shape):<18}  "
                f"{str(out_shape):<10}  {path_label:<10}  {status}"
            )

        except Exception as e:
            all_passed = False
            print(
                f"{detector:<14}  {'ERROR':<22}  {'—':<18}  {'—':<10}  {'—':<10}  {FAIL}  ({e})"
            )

    print()
    if all_passed:
        print("All detectors passed.")
    else:
        print("One or more detectors failed.")
        sys.exit(1)


if __name__ == "__main__":
    run()
