"""Tests for preprocessing geometry: Reborn loading and image assembly."""

import numpy as np
import pytest

from src.preprocessing.geometry import DETECTOR_LOADERS, assemble_image, load_pad_geometry

DETECTORS = ["AGIPD", "JUNGFRAU_4M", "ePix10k", "Eiger4M"]


# ---------------------------------------------------------------------------
# load_pad_geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("detector_type", DETECTORS)
def test_load_pad_geometry_returns_nonempty_list(detector_type):
    pads = load_pad_geometry(detector_type)
    assert len(pads) >= 1


@pytest.mark.parametrize("detector_type", DETECTORS)
def test_load_pad_geometry_has_nonzero_pixels(detector_type):
    pads = load_pad_geometry(detector_type)
    assert pads.n_pixels > 0


@pytest.mark.parametrize("detector_type", DETECTORS)
def test_load_pad_geometry_panels_have_valid_dimensions(detector_type):
    pads = load_pad_geometry(detector_type)
    for pad in pads:
        assert pad.n_ss > 0
        assert pad.n_fs > 0


def test_load_pad_geometry_unknown_detector_raises():
    with pytest.raises(ValueError, match="Unknown detector type"):
        load_pad_geometry("CSPAD")


def test_all_four_detectors_covered():
    assert set(DETECTOR_LOADERS.keys()) == {"AGIPD", "JUNGFRAU_4M", "ePix10k", "Eiger4M"}


# ---------------------------------------------------------------------------
# assemble_image
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("detector_type", DETECTORS)
def test_assemble_image_returns_numpy_array(detector_type):
    pads = load_pad_geometry(detector_type)
    panel_data = [np.ones((p.n_ss, p.n_fs)) for p in pads]
    img = assemble_image(pads, panel_data)
    assert isinstance(img, np.ndarray)


@pytest.mark.parametrize("detector_type", DETECTORS)
def test_assemble_image_total_pixels_matches_pad_geometry(detector_type):
    """Assembled array must contain exactly n_pixels elements."""
    pads = load_pad_geometry(detector_type)
    panel_data = [np.ones((p.n_ss, p.n_fs)) for p in pads]
    img = assemble_image(pads, panel_data)
    assert img.size == pads.n_pixels


@pytest.mark.parametrize("detector_type", DETECTORS)
def test_assemble_image_preserves_pixel_values(detector_type):
    """Non-gap pixels in the assembled image must equal the input value."""
    pads = load_pad_geometry(detector_type)
    panel_data = [np.ones((p.n_ss, p.n_fs)) * 2.0 for p in pads]
    img = assemble_image(pads, panel_data)
    assert img.max() == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Assembly shape notes (documented, not enforced — shapes vary by detector)
# AGIPD:      assembled.ndim == 3  (16 modules × module_ss × module_fs)
# JUNGFRAU_4M: assembled.ndim == 2  (tiled 2D image)
# ePix10k:    assembled.ndim == 2  (tiled 2D image)
# Eiger4M:    assembled.ndim == 1  (monolithic — no tiling needed)
# Phase 3 preprocessing will normalize all detectors to 2D 224×224.
# ---------------------------------------------------------------------------


def test_agipd_assembly_shape_is_3d():
    pads = load_pad_geometry("AGIPD")
    panel_data = [np.ones((p.n_ss, p.n_fs)) for p in pads]
    img = assemble_image(pads, panel_data)
    assert img.ndim == 3, f"Expected AGIPD to assemble to 3D, got shape {img.shape}"


def test_jungfrau_and_epix_assembly_shape_is_2d():
    for det in ("JUNGFRAU_4M", "ePix10k"):
        pads = load_pad_geometry(det)
        panel_data = [np.ones((p.n_ss, p.n_fs)) for p in pads]
        img = assemble_image(pads, panel_data)
        assert img.ndim == 2, f"Expected {det} to assemble to 2D, got shape {img.shape}"


def test_eiger4m_assembly_is_flat_monolithic():
    pads = load_pad_geometry("Eiger4M")
    assert len(pads) == 1, "Eiger4M must have exactly 1 panel"
    panel_data = [np.ones((p.n_ss, p.n_fs)) for p in pads]
    img = assemble_image(pads, panel_data)
    assert img.ndim == 1, f"Expected Eiger4M flat output, got shape {img.shape}"
