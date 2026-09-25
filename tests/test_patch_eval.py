"""Tests for patch-grid tiling and patch-aggregation evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.augment import patch_grid


class TestPatchGrid:
    def test_non_overlapping_count_exact_fit(self):
        # 448 = 2 × 224, so 2×2 = 4 patches exactly
        img = np.zeros((448, 448), dtype=np.float32)
        patches = patch_grid(img, patch_size=224, stride=224)
        assert len(patches) == 4

    def test_non_overlapping_count_partial_edge(self):
        # floor(1273 / 224) = 5 → 5×5 = 25 patches; 153 px edge discarded
        img = np.zeros((1273, 1273), dtype=np.float32)
        patches = patch_grid(img, patch_size=224, stride=224)
        assert len(patches) == 25

    def test_default_stride_equals_patch_size(self):
        img = np.zeros((1273, 1273), dtype=np.float32)
        assert len(patch_grid(img, 224)) == len(patch_grid(img, 224, 224))

    def test_each_patch_is_correct_size(self):
        img = np.zeros((1273, 1273), dtype=np.float32)
        for p in patch_grid(img, 224, 224):
            assert p.shape == (224, 224)

    def test_patches_contain_correct_pixel_values(self):
        img = np.arange(448 * 448, dtype=np.float32).reshape(448, 448)
        patches = patch_grid(img, 224, 224)
        np.testing.assert_array_equal(patches[0], img[0:224, 0:224])
        np.testing.assert_array_equal(patches[1], img[0:224, 224:448])

    def test_overlapping_stride_increases_count(self):
        img = np.zeros((500, 500), dtype=np.float32)
        assert len(patch_grid(img, 224, 100)) > len(patch_grid(img, 224, 224))

    def test_image_smaller_than_patch_returns_empty(self):
        img = np.zeros((100, 100), dtype=np.float32)
        assert patch_grid(img, 224, 224) == []

    def test_row_major_order(self):
        img = np.zeros((448, 448), dtype=np.float32)
        img[0, 0] = 1.0  # top-left → patch index 0
        img[224, 0] = 2.0  # bottom-left → patch index 2
        patches = patch_grid(img, 224, 224)
        assert patches[0][0, 0] == 1.0
        assert patches[2][0, 0] == 2.0


from src.preprocessing.pipeline import preprocess_eval_patches


class TestPreprocessEvalPatches:
    def test_output_shape_agipd_size(self):
        # floor(1273/224)=5 → 5×5=25 patches
        img = np.random.default_rng(42).random((1273, 1273)).astype(np.float32)
        out = preprocess_eval_patches(img)
        assert out.shape == (25, 224, 224)

    def test_output_dtype_float32(self):
        img = np.zeros((500, 500), dtype=np.float32)
        out = preprocess_eval_patches(img, patch_size=224, stride=224)
        assert out.dtype == np.float32

    def test_raises_if_no_complete_patch(self):
        img = np.zeros((100, 100), dtype=np.float32)
        with pytest.raises(ValueError, match="no complete"):
            preprocess_eval_patches(img)

    def test_custom_stride_changes_count(self):
        img = np.random.default_rng(0).random((500, 500)).astype(np.float32)
        default_out = preprocess_eval_patches(img, stride=224)
        overlap_out = preprocess_eval_patches(img, stride=112)
        assert overlap_out.shape[0] > default_out.shape[0]

    def test_each_patch_has_zero_mean_after_gcn(self):
        img = np.random.default_rng(7).random((500, 500)).astype(np.float32) * 1000
        out = preprocess_eval_patches(img)
        patch_means = out.reshape(out.shape[0], -1).mean(axis=1)
        np.testing.assert_allclose(patch_means, 0.0, atol=2e-4)

    def test_output_is_finite(self):
        img = np.random.default_rng(99).random((1273, 1273)).astype(np.float32)
        out = preprocess_eval_patches(img)
        assert np.isfinite(out).all()


import torch
import torch.nn as nn
from pathlib import Path
from src.evaluation.benchmark import run_patch_agg


class _ConstantModel(nn.Module):
    """Always predicts the same 2-class logits for all inputs."""

    def __init__(self, hit_logit: float = 10.0):
        super().__init__()
        self.hit_logit = hit_logit

    def forward(self, x):
        batch = x.shape[0]
        return torch.tensor(
            [[-self.hit_logit, self.hit_logit]] * batch, dtype=torch.float32
        )


def _make_cxi(tmp_path, n_frames=4, n_hits=2, shape=(500, 500), desc="Jungfrau 4M"):
    """Write a minimal CXI-like HDF5 file with data and labels.

    desc="Jungfrau 4M" by default: an arbitrary-shape synthetic frame is not a
    real detector canvas for any of the 4 detector types, so Jungfrau's
    _fake_jungfrau_assembly fixture (below) is required to keep this
    shape-agnostic — real PADAssembler needs a full (2164, 2068) canvas.
    """
    import h5py

    path = tmp_path / "test.cxi"
    with h5py.File(path, "w") as f:
        data = np.random.default_rng(0).random((n_frames, *shape)).astype(np.float32)
        f.create_dataset("entry_1/data_1/data", data=data)
        labels = np.array(
            [1.0] * n_hits + [0.0] * (n_frames - n_hits), dtype=np.float32
        )
        f.create_dataset("entry_1/labels/hit", data=labels)
        if desc is not None:
            f.create_dataset(
                "entry_1/instrument_1/detector_1/description",
                data=desc.encode(),
            )
    return path


@pytest.fixture(autouse=True)
def _fake_jungfrau_assembly(monkeypatch: pytest.MonkeyPatch) -> None:
    """See identical fixture in tests/test_asymmetric_dataset.py — same
    rationale: these synthetic frames are undersized for real PADAssembler,
    so fall back to _to_2d() passthrough for Jungfrau 4M only.
    """
    from src.preprocessing.pipeline import _to_2d
    from src.preprocessing.pipeline import assemble_only as _real_assemble_only

    def _fake(frame, pads, detector_desc, assembler=None):
        if detector_desc == "Jungfrau 4M" and frame.shape != (2164, 2068):
            return _to_2d(frame)
        return _real_assemble_only(frame, pads, detector_desc, assembler=assembler)

    monkeypatch.setattr("src.preprocessing.pipeline.assemble_only", _fake)


class TestRunPatchAgg:
    def test_returns_required_keys(self, tmp_path):
        path = _make_cxi(tmp_path)
        model = _ConstantModel()
        result = run_patch_agg(
            model,
            session_map={"s0": path},
            session_ids=["s0"],
            label_key="entry_1/labels/hit",
            patch_stride=224,
            min_hit_patches=3,
            device="cpu",
        )
        for key in ("ap", "auc_roc", "f1", "threshold"):
            assert key in result, f"Missing key: {key}"

    def test_all_metrics_are_finite(self, tmp_path):
        path = _make_cxi(tmp_path, n_frames=8, n_hits=4)
        model = _ConstantModel(hit_logit=2.0)
        result = run_patch_agg(
            model,
            session_map={"s0": path},
            session_ids=["s0"],
            label_key="entry_1/labels/hit",
            patch_stride=224,
            min_hit_patches=3,
            device="cpu",
        )
        for key in ("ap", "auc_roc", "f1", "threshold"):
            assert np.isfinite(result[key]), f"{key} is not finite: {result[key]}"

    def test_empty_session_ids_returns_nan(self, tmp_path):
        model = _ConstantModel()
        result = run_patch_agg(
            model,
            session_map={},
            session_ids=[],
            label_key="entry_1/labels/hit",
            patch_stride=224,
            min_hit_patches=3,
            device="cpu",
        )
        assert np.isnan(result["ap"])

    def test_threshold_in_unit_interval(self, tmp_path):
        path = _make_cxi(tmp_path, n_frames=6, n_hits=3)
        model = _ConstantModel(hit_logit=2.0)
        result = run_patch_agg(
            model,
            session_map={"s0": path},
            session_ids=["s0"],
            label_key="entry_1/labels/hit",
            patch_stride=224,
            min_hit_patches=3,
            device="cpu",
        )
        assert 0.0 <= result["threshold"] <= 1.0


class TestBenchmarkAssemblyErrorPropagation:
    def test_geometry_error_propagates_not_swallowed(self, tmp_path, monkeypatch):
        path = _make_cxi(tmp_path, n_frames=1, n_hits=0, desc="AGIPD 1M")

        def _raise(*_args, **_kwargs):
            raise ValueError("boom")

        monkeypatch.setattr("src.preprocessing.geometry.get_geometry", _raise)

        model = _ConstantModel()
        with pytest.raises(ValueError, match="boom"):
            run_patch_agg(
                model,
                session_map={"s0": path},
                session_ids=["s0"],
                label_key="entry_1/labels/hit",
                patch_stride=224,
                min_hit_patches=3,
                device="cpu",
            )


def test_preprocess_eval_patches_from_gcn_matches_full_path() -> None:
    """Splitting GCN out of preprocess_eval_patches changes nothing.

    preprocess_eval_patches(x) must equal
    preprocess_eval_patches_from_gcn(fill_gaps(gcn(x))).
    """
    import numpy as np

    from src.preprocessing.normalize import gcn
    from src.preprocessing.pipeline import (
        fill_gaps_after_gcn,
        preprocess_eval_patches,
        preprocess_eval_patches_from_gcn,
    )

    rng = np.random.default_rng(0)
    frame = rng.random((448, 448)).astype(np.float32)

    want = preprocess_eval_patches(frame, patch_size=224, stride=224)

    gcn_frame = gcn(frame.astype(np.float32))
    gcn_frame = fill_gaps_after_gcn(gcn_frame, None, mask=None)
    got = preprocess_eval_patches_from_gcn(
        gcn_frame, mask=None, patch_size=224, stride=224
    )

    assert got.shape == want.shape == (4, 224, 224)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


def test_preprocess_eval_patches_from_gcn_honours_mask() -> None:
    """Invalid pixels are zeroed in the output, as in the masked NumPy LCN."""
    import numpy as np

    from src.preprocessing.pipeline import preprocess_eval_patches_from_gcn

    rng = np.random.default_rng(1)
    gcn_frame = rng.standard_normal((224, 224)).astype(np.float32)
    mask = np.ones((224, 224), dtype=bool)
    mask[:10, :] = False

    out = preprocess_eval_patches_from_gcn(gcn_frame, mask=mask)
    assert out.shape == (1, 224, 224)
    assert np.all(out[0][:10, :] == 0.0)


def test_preprocess_eval_patches_from_gcn_raises_when_no_patch_fits() -> None:
    import numpy as np
    import pytest

    from src.preprocessing.pipeline import preprocess_eval_patches_from_gcn

    with pytest.raises(ValueError, match="no complete"):
        preprocess_eval_patches_from_gcn(np.zeros((100, 100), np.float32))
