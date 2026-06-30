"""Tests for src/preprocessing/augment.py and augmented pipeline functions."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.augment import (
    center_crop,
    random_crop,
    random_cutout,
    random_flip,
    random_rot90,
)
from src.preprocessing.pipeline import preprocess_eval, preprocess_train

RNG_SEED = 42


# ---------------------------------------------------------------------------
# random_crop
# ---------------------------------------------------------------------------


class TestRandomCrop:
    def test_output_shape(self):
        rng = np.random.default_rng(RNG_SEED)
        img = np.ones((500, 600), dtype=np.float32)
        crop = random_crop(img, 224, rng)
        assert crop.shape == (224, 224)

    def test_exact_size_image(self):
        rng = np.random.default_rng(RNG_SEED)
        img = np.ones((224, 224), dtype=np.float32)
        crop = random_crop(img, 224, rng)
        assert crop.shape == (224, 224)

    def test_raises_if_too_small(self):
        rng = np.random.default_rng(RNG_SEED)
        img = np.ones((100, 300), dtype=np.float32)
        with pytest.raises(ValueError, match="smaller than requested crop"):
            random_crop(img, 224, rng)

    def test_different_seeds_give_different_crops(self):
        img = np.arange(500 * 500, dtype=np.float32).reshape(500, 500)
        crop_a = random_crop(img, 224, np.random.default_rng(1))
        crop_b = random_crop(img, 224, np.random.default_rng(2))
        # Very unlikely to be identical with a large image
        assert not np.array_equal(crop_a, crop_b)

    def test_crop_is_contiguous_subarray(self):
        img = np.arange(400 * 400, dtype=np.float32).reshape(400, 400)
        rng = np.random.default_rng(RNG_SEED)
        crop = random_crop(img, 100, rng)
        # Every value in the crop must appear somewhere in img
        assert np.isin(crop, img).all()


# ---------------------------------------------------------------------------
# center_crop
# ---------------------------------------------------------------------------


class TestCenterCrop:
    def test_output_shape(self):
        img = np.ones((1273, 1273), dtype=np.float32)
        crop = center_crop(img, 224)
        assert crop.shape == (224, 224)

    def test_is_centred(self):
        h, w, size = 600, 800, 224
        img = np.zeros((h, w), dtype=np.float32)
        expected_top = (h - size) // 2
        expected_left = (w - size) // 2
        img[expected_top, expected_left] = 1.0
        crop = center_crop(img, size)
        assert crop[0, 0] == 1.0

    def test_deterministic(self):
        img = np.random.default_rng(RNG_SEED).random((900, 900)).astype(np.float32)
        assert np.array_equal(center_crop(img, 224), center_crop(img, 224))

    def test_raises_if_too_small(self):
        img = np.ones((100, 100), dtype=np.float32)
        with pytest.raises(ValueError, match="smaller than requested crop"):
            center_crop(img, 224)


# ---------------------------------------------------------------------------
# random_rot90
# ---------------------------------------------------------------------------


class TestRandomRot90:
    def test_output_shape_square(self):
        rng = np.random.default_rng(RNG_SEED)
        img = np.ones((224, 224), dtype=np.float32)
        assert random_rot90(img, rng).shape == (224, 224)

    def test_k1_matches_numpy(self):
        img = np.arange(16, dtype=np.float32).reshape(4, 4)

        # Force k=1 by using a deterministic rng that returns 1
        class FakeRng:
            def integers(self, low, high):
                return 1

        result = random_rot90(img, FakeRng())
        expected = np.rot90(img, k=1)
        np.testing.assert_array_equal(result, expected)

    def test_k0_identity(self):
        img = np.arange(16, dtype=np.float32).reshape(4, 4)

        class FakeRng:
            def integers(self, low, high):
                return 0

        result = random_rot90(img, FakeRng())
        np.testing.assert_array_equal(result, img)

    def test_rotations_cover_range(self):
        img = np.arange(16, dtype=np.float32).reshape(4, 4)
        seen_ks = set()
        for seed in range(200):
            rng = np.random.default_rng(seed)
            rot = random_rot90(img, rng)
            for k in range(4):
                if np.array_equal(rot, np.rot90(img, k=k)):
                    seen_ks.add(k)
        assert seen_ks == {0, 1, 2, 3}, f"Not all rotations seen: {seen_ks}"


# ---------------------------------------------------------------------------
# random_flip
# ---------------------------------------------------------------------------


class TestRandomFlip:
    def test_output_shape(self):
        rng = np.random.default_rng(RNG_SEED)
        img = np.ones((224, 224), dtype=np.float32)
        assert random_flip(img, rng).shape == (224, 224)

    def test_flips_occur_over_many_seeds(self):
        img = np.arange(16, dtype=np.float32).reshape(4, 4)
        flipped_h = set()
        flipped_v = set()
        for seed in range(200):
            result = random_flip(img, np.random.default_rng(seed))
            flipped_h.add(np.array_equal(result, np.fliplr(img)))
            flipped_v.add(np.array_equal(result, np.flipud(img)))
        # Both True and False should appear across seeds
        assert True in flipped_h and False in flipped_h
        assert True in flipped_v and False in flipped_v

    def test_no_flip_case(self):
        img = np.arange(16, dtype=np.float32).reshape(4, 4)

        class FakeRng:
            def integers(self, low, high, size=None):
                return np.zeros(size if size else 1, dtype=int)

        result = random_flip(img, FakeRng())
        np.testing.assert_array_equal(result, img)


# ---------------------------------------------------------------------------
# random_cutout
# ---------------------------------------------------------------------------


class TestRandomCutout:
    def test_output_shape(self):
        rng = np.random.default_rng(RNG_SEED)
        img = np.ones((224, 224), dtype=np.float32)
        assert random_cutout(img, rng).shape == (224, 224)

    def test_does_not_modify_input(self):
        img = np.ones((224, 224), dtype=np.float32)
        original = img.copy()
        random_cutout(img, np.random.default_rng(RNG_SEED))
        np.testing.assert_array_equal(img, original)

    def test_patches_are_zeroed(self):
        img = np.ones((224, 224), dtype=np.float32)
        result = random_cutout(
            img, np.random.default_rng(RNG_SEED), n_holes=3, hole_size=32
        )
        assert result.min() == 0.0
        assert result.max() == 1.0  # non-patch pixels unchanged

    def test_untouched_pixels_unchanged(self):
        img = np.full((224, 224), 5.0, dtype=np.float32)
        result = random_cutout(
            img, np.random.default_rng(RNG_SEED), n_holes=1, hole_size=10
        )
        # Non-zero pixels must equal 5.0
        assert (result[result != 0.0] == 5.0).all()

    def test_zero_holes_is_identity(self):
        img = np.random.default_rng(RNG_SEED).random((224, 224)).astype(np.float32)
        result = random_cutout(img, np.random.default_rng(RNG_SEED), n_holes=0)
        np.testing.assert_array_equal(result, img)


# ---------------------------------------------------------------------------
# preprocess_train  (end-to-end shape + dtype)
# ---------------------------------------------------------------------------


class TestPreprocessTrain:
    def test_output_shape_and_dtype(self):
        img = np.random.default_rng(RNG_SEED).random((1273, 1273)).astype(np.float32)
        rng = np.random.default_rng(RNG_SEED)
        result = preprocess_train(img, rng)
        assert result.shape == (224, 224)
        assert result.dtype == np.float32

    def test_stochastic_across_calls(self):
        img = np.random.default_rng(RNG_SEED).random((1273, 1273)).astype(np.float32)
        a = preprocess_train(img, np.random.default_rng(1))
        b = preprocess_train(img, np.random.default_rng(2))
        assert not np.array_equal(a, b)

    def test_cutout_params_forwarded(self):
        img = np.ones((500, 500), dtype=np.float32)
        rng = np.random.default_rng(RNG_SEED)
        result = preprocess_train(img, rng, n_cutout_holes=5, cutout_hole_size=20)
        assert result.shape == (224, 224)


# ---------------------------------------------------------------------------
# preprocess_eval  (determinism + shape + dtype)
# ---------------------------------------------------------------------------


class TestPreprocessEval:
    def test_output_shape_and_dtype(self):
        img = np.random.default_rng(RNG_SEED).random((1273, 1273)).astype(np.float32)
        result = preprocess_eval(img)
        assert result.shape == (224, 224)
        assert result.dtype == np.float32

    def test_deterministic(self):
        img = np.random.default_rng(RNG_SEED).random((1500, 1500)).astype(np.float32)
        a = preprocess_eval(img)
        b = preprocess_eval(img)
        np.testing.assert_array_equal(a, b)

    def test_different_from_train(self):
        img = np.random.default_rng(RNG_SEED).random((1273, 1273)).astype(np.float32)
        eval_out = preprocess_eval(img)
        # Train uses random crop; with high probability it differs from centre crop
        train_out = preprocess_train(img, np.random.default_rng(99))
        # They may occasionally coincide if random crop == centre crop, but for a
        # 1273×1273 image the probability is (1/(1050²))² ≈ 0, so this is safe.
        assert not np.array_equal(eval_out, train_out)
