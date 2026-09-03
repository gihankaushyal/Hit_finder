"""Tests for SSLPretrainCXIDataset — random valid-region crops for MAE pretraining."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from src.data.dataset import SSLPretrainCXIDataset
from src.data.dataloader import ssl_crop_loader
from src.hitfinders import MockHitfinder

H, W = 512, 512
N_FRAMES = 8
LABEL_KEY = "entry_1/labels/hit"
DATA_KEY = "entry_1/data_1/data"


@pytest.fixture(scope="module")
def synthetic_cxi(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("ssl_data") / "synthetic.cxi"
    rng = np.random.default_rng(42)
    frames = rng.random((N_FRAMES, H, W)).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset(DATA_KEY, data=frames)
        f.create_dataset(
            LABEL_KEY, data=np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.float32)
        )
        det = f.create_group("entry_1/instrument_1/detector_1")
        det.create_dataset("description", data=b"Jungfrau 4M")
        det.create_dataset("distance", data=0.1)
        det.create_dataset("x_pixel_size", data=1e-4)
        f.create_dataset("entry_1/instrument_1/source_1/wavelength", data=1.3e-10)
    return path


def _dataset(synthetic_cxi, **kwargs) -> SSLPretrainCXIDataset:
    return SSLPretrainCXIDataset(
        session_ids=["s0"],
        session_map={"s0": synthetic_cxi},
        seed=42,
        **kwargs,
    )


class TestSSLPretrainDataset:
    def test_len_is_frames_not_crops(self, synthetic_cxi):
        """__len__ returns frame count regardless of crops_per_frame."""
        ds1 = _dataset(synthetic_cxi, crops_per_frame=1)
        ds2 = _dataset(synthetic_cxi, crops_per_frame=2)
        assert len(ds1) == N_FRAMES
        assert len(ds2) == N_FRAMES  # same: one item per frame

    def test_item_shape_and_dtype_single_crop(self, synthetic_cxi):
        """crops_per_frame=1 returns (1,1,224,224), (1,196), (1,1,224,224)."""
        crops, peak_patches, valid_mask = _dataset(synthetic_cxi, crops_per_frame=1)[0]
        assert crops.shape == (1, 1, 224, 224)
        assert crops.dtype == torch.float32
        assert peak_patches.shape == (1, 196)
        assert peak_patches.dtype == torch.bool
        assert valid_mask.shape == (1, 1, 224, 224)
        assert valid_mask.dtype == torch.float32

    def test_item_shape_and_dtype_multi_crop(self, synthetic_cxi):
        """crops_per_frame=2 returns (2,1,224,224), (2,196), (2,1,224,224)."""
        crops, peak_patches, valid_mask = _dataset(synthetic_cxi, crops_per_frame=2)[0]
        assert crops.shape == (2, 1, 224, 224)
        assert peak_patches.shape == (2, 196)
        assert valid_mask.shape == (2, 1, 224, 224)

    def test_no_hitfinder_means_no_peak_patches(self, synthetic_cxi):
        _, peak_patches, _ = _dataset(synthetic_cxi, crops_per_frame=1)[0]
        assert not peak_patches.any()

    def test_hitfinder_centroids_map_to_patch_ids(self, synthetic_cxi):
        ds = _dataset(
            synthetic_cxi,
            crops_per_frame=16,
            hitfinder=MockHitfinder(np.array([[256.0, 256.0]], dtype=np.float32)),
        )
        found = False
        for i in range(len(ds)):
            _, peak_patches, _ = ds[i]
            if peak_patches.any():
                found = True
                break
        assert found

    def test_patch_id_mapping_helpers(self):
        from src.data.dataset import _centroid_map, _patch_ids_from_map

        m = _centroid_map(np.array([[20.0, 35.0]]))
        assert m[35, 20] == 1.0
        ids = _patch_ids_from_map(m > 0.5)
        assert ids[2 * 14 + 1]
        assert ids.sum() == 1

    def test_deterministic_given_seed(self, synthetic_cxi):
        a, _, _ = _dataset(synthetic_cxi, crops_per_frame=1)[3]
        b, _, _ = _dataset(synthetic_cxi, crops_per_frame=1)[3]
        assert torch.equal(a, b)

    def test_multi_crops_differ_within_same_frame(self, synthetic_cxi):
        """The two crops from the same frame are different random windows."""
        crops, _, _ = _dataset(synthetic_cxi, crops_per_frame=2)[0]
        assert not torch.equal(crops[0], crops[1])

    def test_loader_batches_flatten_collate(self, synthetic_cxi):
        """After flatten collate, batch shape is (batch_size, 1, 224, 224)."""
        dl = ssl_crop_loader(
            session_map={"s0": synthetic_cxi},
            session_ids=["s0"],
            batch_size=4,
            num_workers=0,
            shuffle=False,
            crops_per_frame=1,
        )
        crops, peaks, vmasks = next(iter(dl))
        assert crops.shape == (4, 1, 224, 224)
        assert peaks.shape == (4, 196)
        assert vmasks.shape == (4, 1, 224, 224)

    def test_loader_multicrop_flatten(self, synthetic_cxi):
        """crops_per_frame=2, batch_size=4: loader_batch=2 frames, flatten → 4 crops."""
        dl = ssl_crop_loader(
            session_map={"s0": synthetic_cxi},
            session_ids=["s0"],
            batch_size=4,
            num_workers=0,
            shuffle=False,
            crops_per_frame=2,
        )
        crops, peaks, vmasks = next(iter(dl))
        assert crops.shape == (4, 1, 224, 224)
        assert peaks.shape == (4, 196)
        assert vmasks.shape == (4, 1, 224, 224)
