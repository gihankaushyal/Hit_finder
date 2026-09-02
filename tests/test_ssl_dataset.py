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
    def test_len_is_frames_times_crops(self, synthetic_cxi):
        ds = _dataset(synthetic_cxi, crops_per_frame=2)
        assert len(ds) == N_FRAMES * 2

    def test_item_shape_and_dtype(self, synthetic_cxi):
        crop, peak_patches, valid_mask = _dataset(synthetic_cxi)[0]
        assert crop.shape == (1, 224, 224)
        assert crop.dtype == torch.float32
        assert peak_patches.shape == (196,) and peak_patches.dtype == torch.bool
        assert valid_mask.shape == (1, 224, 224)
        assert valid_mask.dtype == torch.float32

    def test_no_hitfinder_means_no_peak_patches(self, synthetic_cxi):
        _, peak_patches, _ = _dataset(synthetic_cxi)[0]
        assert not peak_patches.any()

    def test_hitfinder_centroids_map_to_patch_ids(self, synthetic_cxi):
        # MockHitfinder returns a fixed centroid near the frame centre so a
        # large fraction of random 224-crops contain it; with 16 crops/frame
        # and a fixed seed the sweep below is deterministic.
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
        assert found  # some crop contains the centroid → mapped into a patch id

    def test_patch_id_mapping_helpers(self):
        from src.data.dataset import _centroid_map, _patch_ids_from_map

        m = _centroid_map(np.array([[20.0, 35.0]]))  # x=20 → col 20, y=35 → row 35
        assert m[35, 20] == 1.0
        ids = _patch_ids_from_map(m > 0.5)
        # 224/16 = 14-wide grid; row 35 → patch row 2, col 20 → patch col 1
        assert ids[2 * 14 + 1]
        assert ids.sum() == 1

    def test_deterministic_given_seed(self, synthetic_cxi):
        a, _, _ = _dataset(synthetic_cxi)[3]
        b, _, _ = _dataset(synthetic_cxi)[3]
        assert torch.equal(a, b)

    def test_loader_batches(self, synthetic_cxi):
        dl = ssl_crop_loader(
            session_map={"s0": synthetic_cxi},
            session_ids=["s0"],
            batch_size=4,
            num_workers=0,
            shuffle=False,
        )
        crops, peaks, vmasks = next(iter(dl))
        assert crops.shape == (4, 1, 224, 224)
        assert peaks.shape == (4, 196)
        assert vmasks.shape == (4, 1, 224, 224)
