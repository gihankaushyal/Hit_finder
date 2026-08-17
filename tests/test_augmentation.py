"""Tests for src/preprocessing/augment.py and augmented pipeline functions."""

from __future__ import annotations

import numpy as np

from src.preprocessing.augment import (
    PAD_BORDER_DEFAULT,
    pad_border,
    random_cutout,
    random_flip,
    random_rot90,
)

RNG_SEED = 42


# ---------------------------------------------------------------------------
# pad_border
# ---------------------------------------------------------------------------


class TestPadBorder:
    def test_output_shape(self) -> None:
        img = np.zeros((300, 400), dtype=np.float32)
        out = pad_border(img)
        assert out.shape == (300 + 2 * PAD_BORDER_DEFAULT, 400 + 2 * PAD_BORDER_DEFAULT)

    def test_border_is_zero(self) -> None:
        img = np.ones((300, 300), dtype=np.float32)
        out = pad_border(img)
        assert out[0, :].sum() == 0
        assert out[-1, :].sum() == 0
        assert out[:, 0].sum() == 0
        assert out[:, -1].sum() == 0

    def test_interior_unchanged(self) -> None:
        img = np.random.default_rng(0).random((300, 300)).astype(np.float32)
        out = pad_border(img)
        p = PAD_BORDER_DEFAULT
        np.testing.assert_array_equal(out[p:-p, p:-p], img)

    def test_zero_pad_size_is_identity(self) -> None:
        img = np.ones((224, 224), dtype=np.float32)
        out = pad_border(img, pad_size=0)
        np.testing.assert_array_equal(out, img)

    def test_custom_pad_size(self) -> None:
        img = np.zeros((100, 100), dtype=np.float32)
        out = pad_border(img, pad_size=50)
        assert out.shape == (200, 200)


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

    def test_avoid_protects_peak_pixels(self):
        # With protected pixels scattered across the image, no hole (expanded
        # by the margin) may ever touch them — over many seeded draws.
        img = np.ones((224, 224), dtype=np.float32)
        avoid = np.zeros((224, 224), dtype=bool)
        peaks = [(40, 60), (112, 112), (180, 30), (70, 200)]
        for r, c in peaks:
            avoid[r, c] = True
        for seed in range(50):
            result = random_cutout(
                img, np.random.default_rng(seed), n_holes=3, avoid=avoid
            )
            for r, c in peaks:
                assert result[r, c] == 1.0, f"peak ({r},{c}) occluded at seed {seed}"

    def test_avoid_margin_respected(self):
        # Every zeroed pixel must lie at least avoid_margin away from any
        # protected pixel.
        img = np.ones((224, 224), dtype=np.float32)
        avoid = np.zeros((224, 224), dtype=bool)
        avoid[112, 112] = True
        margin = 8
        for seed in range(50):
            result = random_cutout(
                img,
                np.random.default_rng(seed),
                n_holes=3,
                avoid=avoid,
                avoid_margin=margin,
            )
            zr, zc = np.nonzero(result == 0.0)
            if len(zr):
                dist = np.maximum(np.abs(zr - 112), np.abs(zc - 112)).min()
                assert dist > margin

    def test_fully_protected_image_skips_all_holes(self):
        img = np.ones((224, 224), dtype=np.float32)
        avoid = np.ones((224, 224), dtype=bool)
        result = random_cutout(img, np.random.default_rng(RNG_SEED), avoid=avoid)
        np.testing.assert_array_equal(result, img)
