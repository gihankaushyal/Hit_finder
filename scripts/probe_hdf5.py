"""Walk HDF5/CXI file structure — report keys, shapes, dtypes, and flag entry/data/data."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py

EXPECTED_KEY = "entry/data/data"

# Update these paths before running.
DETECTOR_FILES: dict[str, str] = {
    "AGIPD": "/path/to/agipd_frame.h5",
    "JUNGFRAU_4M": "/path/to/jungfrau_frame.cxi",
    "ePix10k": "/path/to/epix_frame.h5",
    "Eiger4M": "/path/to/eiger_frame.h5",
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

    status = (
        "FOUND ✓"
        if found
        else "NOT FOUND ✗ — update io.py and _CONFIRMED_KEYS in test_io.py"
    )
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
