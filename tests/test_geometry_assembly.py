"""Unit tests for geometry-aware CXI assembly functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.preprocessing.geometry import _EIGER4M_GEOM, get_geometry

# ---------------------------------------------------------------------------
# read_detector_description
# ---------------------------------------------------------------------------


class TestReadDetectorDescription:
    def test_returns_description_string(self, tmp_path):
        import h5py

        from src.preprocessing.io import read_detector_description

        cxi = tmp_path / "test.cxi"
        with h5py.File(cxi, "w") as f:
            f.create_dataset(
                "entry_1/instrument_1/detector_1/description",
                data=b"AGIPD 1M",
            )
        assert read_detector_description(cxi) == "AGIPD 1M"

    def test_raises_on_missing_key(self, tmp_path):
        import h5py

        from src.preprocessing.io import read_detector_description

        cxi = tmp_path / "test.cxi"
        with h5py.File(cxi, "w") as f:
            f.create_dataset("entry_1/data_1/data", data=np.zeros((10, 10)))
        with pytest.raises(ValueError, match="description key"):
            read_detector_description(cxi)

    def test_raises_on_missing_file(self, tmp_path):
        from src.preprocessing.io import read_detector_description

        with pytest.raises(FileNotFoundError):
            read_detector_description(tmp_path / "nonexistent.cxi")


# ---------------------------------------------------------------------------
# get_geometry
# ---------------------------------------------------------------------------


class TestGetGeometry:
    def test_raises_for_jungfrau(self):
        with pytest.raises(ValueError, match="pre-assembled"):
            get_geometry("Jungfrau 4M")

    def test_raises_for_unknown(self):
        with pytest.raises(ValueError, match="Unknown detector"):
            get_geometry("MYSTERY DETECTOR")

    def test_eiger4m_geom_file_exists(self):
        # AGIPD and ePix10k use Reborn standard loaders (no file needed).
        # Only Eiger4M requires a CrystFEL .geom file.
        assert _EIGER4M_GEOM.exists(), f"Missing Eiger4M geom file: {_EIGER4M_GEOM}"

    @pytest.mark.parametrize("desc", ["AGIPD 1M", "ePix10k 2.2M", "EIGER 4M"])
    def test_returns_pad_geometry_list(self, desc):
        from reborn.detector import PADGeometryList

        pads = get_geometry(desc)
        assert isinstance(pads, PADGeometryList)
        assert len(pads) > 0

    def test_caches_geometry(self):
        from src.preprocessing.geometry import _GEOM_CACHE

        # Pre-warm then verify same object returned
        pads1 = get_geometry("EIGER 4M")
        pads2 = get_geometry("EIGER 4M")
        assert pads1 is pads2


