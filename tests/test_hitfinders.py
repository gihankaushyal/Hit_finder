# tests/test_hitfinders.py
from __future__ import annotations

import numpy as np
import pytest

from src.hitfinders.base import MockHitfinder
from src.hitfinders import get_hitfinder


def test_mock_hitfinder_returns_correct_shape():
    hf = MockHitfinder(peaks=np.array([[100.0, 200.0], [300.0, 400.0]]))
    frame = np.zeros((512, 512), dtype=np.float32)
    result = hf.find_peaks(frame)
    assert result.shape == (2, 2)
    assert result.dtype == np.float32


def test_mock_hitfinder_empty_peaks():
    hf = MockHitfinder(peaks=np.zeros((0, 2), dtype=np.float32))
    result = hf.find_peaks(np.zeros((512, 512), dtype=np.float32))
    assert result.shape == (0, 2)


def test_get_hitfinder_mock():
    cfg = {"hitfinder": {"backend": "mock"}}
    hf = get_hitfinder(cfg)
    assert isinstance(hf, MockHitfinder)


def test_get_hitfinder_unknown_backend_raises():
    cfg = {"hitfinder": {"backend": "unknown"}}
    with pytest.raises(ValueError, match="Unknown hitfinder backend"):
        get_hitfinder(cfg)


def test_pf8_hitfinder_find_peaks_raises_not_implemented():
    from src.hitfinders.pf8 import PF8Hitfinder
    hf = PF8Hitfinder(threshold_snr=5.0, min_peaks=1)
    with pytest.raises(NotImplementedError):
        hf.find_peaks(np.zeros((512, 512), dtype=np.float32))


def test_gpu_hitfinder_find_peaks_raises_not_implemented():
    from src.hitfinders.gpu import GPUHitfinder
    hf = GPUHitfinder()
    with pytest.raises(NotImplementedError):
        hf.find_peaks(np.zeros((512, 512), dtype=np.float32))


def test_hitfinder_protocol_isinstance():
    from src.hitfinders.base import Hitfinder
    hf = MockHitfinder()
    assert isinstance(hf, Hitfinder)
