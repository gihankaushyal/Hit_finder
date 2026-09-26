"""One-off verification that reborn's bundled eiger4m_64_pad_geometry_list()
matches our local eiger4m.geom-derived geometry, gating the migration in
src/preprocessing/geometry.py. Run manually; not part of the pytest suite.

Our eiger4m.geom sets a global clen=0.300m AND a per-panel coffset=-0.1860225
on every panel; CrystFEL geometry convention sums these, giving an effective
panel z of 0.300 - 0.1860225 = 0.1139775m. reborn's eiger4m_64_pad_geometry_list()
sets the *average* z directly to whatever detector_distance is passed (it has
no notion of a separate coffset), so 0.1139775 must be passed explicitly to
reproduce our current geometry. A first attempt at just clen=0.300 alone
failed with a uniform 0.1860225m offset across all 64 panels (X/Y and panel
ordering matched exactly) — the offset value is not a coincidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from reborn import detector
from reborn.external.crystfel import geometry_file_to_pad_geometry_list

_EIGER4M_GEOM = (
    Path(__file__).parent.parent / "src" / "preprocessing" / "data" / "eiger4m.geom"
)


EIGER4M_EFFECTIVE_DISTANCE_M = 0.1139775  # clen(0.300) + coffset(-0.1860225)


def main() -> None:
    ours = geometry_file_to_pad_geometry_list(str(_EIGER4M_GEOM))
    reborn_pads = detector.eiger4m_64_pad_geometry_list(
        detector_distance=EIGER4M_EFFECTIVE_DISTANCE_M
    )

    assert len(ours) == 64, f"our geometry panel count != 64: {len(ours)}"
    assert (
        len(reborn_pads) == 64
    ), f"reborn geometry panel count != 64: {len(reborn_pads)}"

    for i, pad in enumerate(ours):
        assert (
            pad.n_ss == 176 and pad.n_fs == 192
        ), f"our panel {i} shape mismatch: n_ss={pad.n_ss}, n_fs={pad.n_fs}"
    for i, pad in enumerate(reborn_pads):
        assert (
            pad.n_ss == 176 and pad.n_fs == 192
        ), f"reborn panel {i} shape mismatch: n_ss={pad.n_ss}, n_fs={pad.n_fs}"

    ours_positions = ours.position_vecs()
    reborn_positions = reborn_pads.position_vecs()

    max_diff = np.max(np.abs(ours_positions - reborn_positions))
    assert (
        max_diff < 1e-6
    ), f"max panel-position difference {max_diff:.3e} m exceeds 1e-6 m tolerance"

    print(f"Max panel-position difference: {max_diff:.3e} m")
    print("Verification PASSED")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"Verification FAILED: {exc}")
        sys.exit(1)
