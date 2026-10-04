"""Smoke test: raw shape, assembled shape, and preprocessed output for each detector.

Reads frame 0 from one CXI file per detector, prints:
  - raw shape (detector-native, from read_frame)
  - assembled shape (intermediate 2D image before GCN/LCN/resize)
  - output shape (should be assembled shape + 2*PAD_BORDER_DEFAULT)

Assembled shape and output shape are both derived from assemble_only(), the
same function the production pipeline calls (src/preprocessing/pipeline.py) —
this checks pad_border() adds the expected border, not a second,
independently-computed assembly path. An earlier version of this script
recomputed the assembled shape via a separate concat_data()+reshape() path
(src/preprocessing/geometry.py::assemble_image()); that path diverges from
assemble_only()'s PADAssembler-based assembly for AGIPD/ePix10k/Eiger4M
(no gap pixels — PADAssembler places panels at lab-frame float positions,
producing a different canvas than a plain reshape), which produced spurious
FAILs unrelated to the pipeline itself.

Usage:
    python scripts/smoke_test_detector_shapes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.geometry import get_assembler, get_geometry
from src.preprocessing.io import read_detector_description, read_frame
from src.preprocessing.augment import PAD_BORDER_DEFAULT, pad_border
from src.preprocessing.pipeline import assemble_only

DATA_ROOT = Path("/data/bioxfel/user/gihan/Resonet/production")

DETECTORS = {
    "AGIPD": DATA_ROOT / "agipd_20k" / "compressed0.cxi",
    "JUNGFRAU_4M": DATA_ROOT / "jungfrau_20k" / "compressed0.cxi",
    "ePix10k": DATA_ROOT / "epix10k_20k" / "compressed0.cxi",
    "Eiger4M": DATA_ROOT / "eiger4m_20k" / "compressed0.cxi",
}

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def run():
    all_passed = True

    header = f"\n{'Detector':<14}  {'Raw shape':<22}  {'Assembled shape':<18}  {'Output':<10}  Result"
    print(header)
    print("-" * len(header))

    for detector, cxi_path in DETECTORS.items():
        try:
            raw = read_frame(cxi_path, frame_idx=0)
            raw_shape = raw.shape

            desc = read_detector_description(cxi_path)

            pads = get_geometry(desc)
            assembler = get_assembler(desc)
            assembled_raw = assemble_only(raw, pads, desc, assembler=assembler)
            assembled_shape = assembled_raw.shape

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
                f"{str(out_shape):<10}  {status}"
            )

        except Exception as e:
            all_passed = False
            print(f"{detector:<14}  {'ERROR':<22}  {'—':<18}  {'—':<10}  {FAIL}  ({e})")

    print()
    if all_passed:
        print("All detectors passed.")
    else:
        print("One or more detectors failed.")
        sys.exit(1)


if __name__ == "__main__":
    run()
