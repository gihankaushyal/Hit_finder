"""Smoke test: raw frame shape and preprocessed output shape for each detector.

Reads frame 0 from one CXI file per detector, prints the raw shape coming out
of read_frame(), runs it through preprocess_assembled(), and asserts the output
is (224, 224).

Usage:
    python scripts/smoke_test_detector_shapes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.io import read_frame
from src.preprocessing.pipeline import preprocess_assembled

DATA_ROOT = Path("/data/bioxfel/user/gihan/Resonet/production")

DETECTORS = {
    "AGIPD":       DATA_ROOT / "agipd_20k"    / "compressed0.cxi",
    "JUNGFRAU_4M": DATA_ROOT / "jungfrau_20k" / "compressed0.cxi",
    "ePix10k":     DATA_ROOT / "epix10k_20k"  / "compressed0.cxi",
    "Eiger4M":     DATA_ROOT / "eiger4m_20k"  / "compressed0.cxi",
}

EXPECTED_OUTPUT_SHAPE = (224, 224)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def run():
    all_passed = True

    print(f"\n{'Detector':<14}  {'Raw shape':<30}  {'Output shape':<14}  Result")
    print("-" * 72)

    for detector, cxi_path in DETECTORS.items():
        try:
            raw = read_frame(cxi_path, frame_idx=0)
            raw_shape = raw.shape

            out = preprocess_assembled(raw)
            out_shape = out.shape

            ok = out_shape == EXPECTED_OUTPUT_SHAPE
            status = PASS if ok else FAIL
            if not ok:
                all_passed = False

            print(f"{detector:<14}  {str(raw_shape):<30}  {str(out_shape):<14}  {status}")

        except Exception as e:
            all_passed = False
            print(f"{detector:<14}  {'ERROR':<30}  {'—':<14}  {FAIL}  ({e})")

    print()
    if all_passed:
        print("All detectors passed.")
    else:
        print("One or more detectors failed.")
        sys.exit(1)


if __name__ == "__main__":
    run()
