"""Tests for preprocessing geometry: Reborn loading and image assembly."""

import numpy as np
import pytest

from src.preprocessing.geometry import (
    DETECTOR_LOADERS,
    assemble_image,
    eiger_resonet_pad_geometry_list,
    extract_panels_from_canvas,
    jungfrau4m_crystfel_pad_geometry_list,
    load_pad_geometry,
)

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


def test_all_detectors_covered():
    assert set(DETECTOR_LOADERS.keys()) == {
        "AGIPD",
        "JUNGFRAU_4M",
        "ePix10k",
        "Eiger4M",
        "EigerRESoNeT",
    }


# ---------------------------------------------------------------------------
# jungfrau4m_crystfel_pad_geometry_list
# ---------------------------------------------------------------------------


def test_jungfrau4m_crystfel_geometry_has_8_panels():
    pads = jungfrau4m_crystfel_pad_geometry_list()
    assert len(pads) == 8


def test_jungfrau4m_crystfel_geometry_pixel_count():
    pads = jungfrau4m_crystfel_pad_geometry_list()
    assert pads.n_pixels == 8 * 514 * 1030


def test_jungfrau4m_crystfel_geometry_default_distance_is_103mm():
    """Default distance must be ~103 mm — matches the source .geom file."""
    pads = jungfrau4m_crystfel_pad_geometry_list()
    dist = pads.average_detector_distance(beam_vec=[0, 0, 1])
    assert abs(dist - 0.103) < 1e-3


def test_jungfrau4m_load_pad_geometry_default_distance_is_103mm():
    """load_pad_geometry('JUNGFRAU_4M') with no explicit distance must use 103 mm."""
    pads = load_pad_geometry("JUNGFRAU_4M")
    dist = pads.average_detector_distance(beam_vec=[0, 0, 1])
    assert abs(dist - 0.103) < 1e-3


def test_jungfrau4m_crystfel_geometry_defines_slicing():
    pads = jungfrau4m_crystfel_pad_geometry_list()
    assert pads.defines_slicing()


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
def test_assemble_image_output_covers_all_pixels(detector_type):
    """Assembled output must contain at least n_pixels elements.

    PADAssembler-based geometries (e.g. JUNGFRAU_4M) produce a 2D lab-frame
    grid that is larger than n_pixels because gap pixels are zero-filled.
    """
    pads = load_pad_geometry(detector_type)
    panel_data = [np.ones((p.n_ss, p.n_fs)) for p in pads]
    img = assemble_image(pads, panel_data)
    assert img.size >= pads.n_pixels


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
# Eiger4M:    assembled.ndim == 2  (64-panel CrystFEL geom → PADAssembler → 2D)
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


def test_eiger4m_crystfel_geometry_properties():
    # CrystFEL-backed loader: 64 panels, defines parent_data_slice, assembles to 2D.
    pads = load_pad_geometry("Eiger4M")
    assert len(pads) == 64, f"Eiger4M CrystFEL geom must have 64 panels, got {len(pads)}"
    assert pads.defines_slicing(), "Eiger4M CrystFEL geom must define parent_data_slice"
    panel_data = [np.ones((p.n_ss, p.n_fs)) for p in pads]
    img = assemble_image(pads, panel_data)
    assert img.ndim == 2, f"Expected Eiger4M PADAssembler output to be 2D, got shape {img.shape}"


# ---------------------------------------------------------------------------
# EigerRESoNeT geometry and extract_panels_from_canvas
# ---------------------------------------------------------------------------


def test_eiger_resonet_geometry_loads():
    pads = eiger_resonet_pad_geometry_list()
    assert len(pads) == 64


def test_eiger_resonet_geometry_panel_dimensions():
    pads = eiger_resonet_pad_geometry_list()
    for pad in pads:
        assert pad.n_ss == 176
        assert pad.n_fs == 192


def test_eiger_resonet_geometry_total_pixels():
    pads = eiger_resonet_pad_geometry_list()
    assert pads.n_pixels == 5632 * 384


def test_eiger_resonet_geometry_defines_slicing():
    pads = eiger_resonet_pad_geometry_list()
    assert pads.defines_slicing()


def test_eiger_resonet_load_via_detector_loaders():
    pads = load_pad_geometry("EigerRESoNeT")
    assert len(pads) == 64


def test_extract_panels_from_canvas_count():
    pads = eiger_resonet_pad_geometry_list()
    canvas = np.zeros((5632, 384), dtype=np.float32)
    panels = extract_panels_from_canvas(canvas, pads)
    assert len(panels) == 64


def test_extract_panels_from_canvas_shape():
    pads = eiger_resonet_pad_geometry_list()
    canvas = np.zeros((5632, 384), dtype=np.float32)
    panels = extract_panels_from_canvas(canvas, pads)
    for panel in panels:
        assert panel.shape == (176, 192)


def test_extract_panels_from_canvas_preserves_values():
    pads = eiger_resonet_pad_geometry_list()
    canvas = np.ones((5632, 384), dtype=np.float32) * 42.0
    panels = extract_panels_from_canvas(canvas, pads)
    assert all(p.max() == pytest.approx(42.0) for p in panels)


def test_extract_panels_non_slicing_geometry_raises():
    from reborn import detector as reborn_detector

    pad = reborn_detector.PADGeometry()
    pad.n_ss = 10
    pad.n_fs = 10
    pad.parent_data_slice = None
    pads = reborn_detector.PADGeometryList([pad])
    assert not pads.defines_slicing()
    with pytest.raises(ValueError, match="does not define parent_data_slice"):
        extract_panels_from_canvas(np.zeros((100, 100)), pads)
