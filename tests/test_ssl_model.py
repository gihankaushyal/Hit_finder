"""Tests for src/models/ssl.py — MAE ViT-S/16, masking pipelines, loss."""

import numpy as np
import pytest
import torch

from src.models.ssl import (
    MAE_MASK_RATIO_DEFAULT,
    peak_aware_masking,
    random_masking,
)


class TestRandomMasking:
    def test_default_ratio_is_lower_than_mae_standard(self):
        # Design decision 2026-09-01: default 0.6 (risk B3), not 0.75.
        assert MAE_MASK_RATIO_DEFAULT == 0.6

    def test_shapes_and_counts(self):
        B, L = 4, 196
        g = torch.Generator().manual_seed(0)
        ids_keep, mask, ids_restore = random_masking(B, L, mask_ratio=0.6, generator=g)
        len_keep = int(L * (1 - 0.6))
        assert ids_keep.shape == (B, len_keep)
        assert mask.shape == (B, L)
        assert ids_restore.shape == (B, L)
        # mask: 1 = masked, 0 = kept
        assert torch.all(mask.sum(dim=1) == L - len_keep)

    def test_mask_consistent_with_ids_keep(self):
        B, L = 2, 196
        g = torch.Generator().manual_seed(1)
        ids_keep, mask, _ = random_masking(B, L, mask_ratio=0.5, generator=g)
        for b in range(B):
            kept = set(ids_keep[b].tolist())
            for i in range(L):
                assert (i in kept) == (mask[b, i].item() == 0)

    def test_deterministic_with_generator(self):
        a = random_masking(2, 196, 0.6, generator=torch.Generator().manual_seed(7))
        b = random_masking(2, 196, 0.6, generator=torch.Generator().manual_seed(7))
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


class TestPeakAwareMasking:
    def test_peak_patches_are_masked(self):
        B, L = 2, 196
        peak_patches = torch.zeros(B, L, dtype=torch.bool)
        peak_patches[0, [3, 50, 120]] = True
        peak_patches[1, [7]] = True
        g = torch.Generator().manual_seed(0)
        ids_keep, mask, ids_restore = peak_aware_masking(
            peak_patches, mask_ratio=0.6, peak_mask_frac=1.0, generator=g
        )
        # every peak patch must be masked
        assert torch.all(mask[peak_patches] == 1)
        # exact masked count preserved
        len_keep = int(L * (1 - 0.6))
        assert torch.all(mask.sum(dim=1) == L - len_keep)

    def test_no_peaks_falls_back_to_random(self):
        B, L = 3, 196
        empty = torch.zeros(B, L, dtype=torch.bool)
        ids_keep, mask, _ = peak_aware_masking(
            empty,
            mask_ratio=0.6,
            peak_mask_frac=1.0,
            generator=torch.Generator().manual_seed(0),
        )
        len_keep = int(L * (1 - 0.6))
        assert torch.all(mask.sum(dim=1) == L - len_keep)

    def test_peak_mask_frac_half(self):
        B, L = 1, 196
        peak_patches = torch.zeros(B, L, dtype=torch.bool)
        peak_patches[0, :10] = True  # 10 peak patches
        _, mask, _ = peak_aware_masking(
            peak_patches,
            mask_ratio=0.6,
            peak_mask_frac=0.5,
            generator=torch.Generator().manual_seed(0),
        )
        # at least ceil(0.5 * 10) = 5 peak patches masked
        assert int(mask[0, :10].sum().item()) >= 5

    def test_more_peaks_than_budget_caps_at_budget(self):
        B, L = 1, 196
        peak_patches = torch.ones(B, L, dtype=torch.bool)  # everything is a peak
        _, mask, _ = peak_aware_masking(
            peak_patches,
            mask_ratio=0.25,
            peak_mask_frac=1.0,
            generator=torch.Generator().manual_seed(0),
        )
        len_keep = int(L * (1 - 0.25))
        assert torch.all(mask.sum(dim=1) == L - len_keep)  # never exceeds budget


from src.models.ssl import MAEViT, build_mae_model  # noqa: E402


def _tiny_mae() -> MAEViT:
    # Tiny dims for CPU tests; production dims come from configs/ssl/mae_pretrain.yaml.
    return MAEViT(
        img_size=224,
        patch_size=16,
        in_chans=1,
        embed_dim=64,
        depth=2,
        num_heads=2,
        decoder_embed_dim=32,
        decoder_depth=1,
        decoder_num_heads=2,
    )


class TestMAEViT:
    def test_forward_shapes(self):
        model = _tiny_mae()
        x = torch.randn(2, 1, 224, 224)
        loss, pred, mask = model(x, mask_ratio=0.6)
        L = (224 // 16) ** 2
        assert pred.shape == (2, L, 16 * 16 * 1)
        assert mask.shape == (2, L)
        assert loss.ndim == 0 and torch.isfinite(loss)

    def test_patchify_unpatchify_roundtrip(self):
        model = _tiny_mae()
        x = torch.randn(2, 1, 224, 224)
        assert torch.allclose(model.unpatchify(model.patchify(x)), x, atol=1e-6)

    def test_loss_only_on_masked_patches(self):
        model = _tiny_mae()
        x = torch.randn(1, 1, 224, 224)
        g = torch.Generator().manual_seed(0)
        loss1, pred, mask = model(x, mask_ratio=0.6, generator=g)
        # Reconstruction loss computed manually over masked patches must match.
        target = model.patchify(x)
        per_patch = ((pred - target) ** 2).mean(dim=-1)
        expected = (per_patch * mask).sum() / mask.sum()
        assert torch.allclose(loss1, expected, atol=1e-5)

    def test_invalid_pixels_excluded_from_loss(self):
        model = _tiny_mae()
        model.eval()  # deterministic forward
        x = torch.randn(1, 1, 224, 224)
        valid = torch.ones(1, 1, 224, 224)
        valid[:, :, :112, :] = 0  # top half invalid (gap pixels)
        g1 = torch.Generator().manual_seed(3)
        g2 = torch.Generator().manual_seed(3)
        with torch.no_grad():
            loss_masked, _, _ = model(x, mask_ratio=0.6, generator=g1, valid_mask=valid)
            # corrupting invalid pixels changes the input to the encoder, so
            # compare against a loss recomputed with the SAME pred: use the
            # reconstruction_loss method directly.
            _, pred, mask = model(x, mask_ratio=0.6, generator=g2, valid_mask=valid)
        x2 = x.clone()
        x2[:, :, :112, :] += 100.0  # corrupt only invalid pixels in the TARGET
        loss_corrupted_target = model.reconstruction_loss(
            x2, pred, mask, valid_mask=valid
        )
        loss_clean_target = model.reconstruction_loss(x, pred, mask, valid_mask=valid)
        assert torch.allclose(loss_clean_target, loss_corrupted_target, atol=1e-5)

    def test_peak_aware_path(self):
        model = _tiny_mae()
        x = torch.randn(2, 1, 224, 224)
        L = (224 // 16) ** 2
        peaks = torch.zeros(2, L, dtype=torch.bool)
        peaks[0, 5] = True
        loss, _, mask = model(
            x,
            mask_ratio=0.6,
            masking="peak_aware",
            peak_patches=peaks,
            generator=torch.Generator().manual_seed(0),
        )
        assert mask[0, 5] == 1

    def test_build_mae_model_from_config(self):
        cfg = {
            "ssl": {
                "embed_dim": 64,
                "depth": 2,
                "num_heads": 2,
                "decoder_embed_dim": 32,
                "decoder_depth": 1,
                "decoder_num_heads": 2,
                "norm_pix_loss": False,
            }
        }
        model = build_mae_model(cfg)
        assert isinstance(model, MAEViT)

    def test_unknown_masking_raises(self):
        model = _tiny_mae()
        with pytest.raises(ValueError):
            model(torch.randn(1, 1, 224, 224), masking="bogus")


from src.models.ssl import ViTClassifier, build_ssl_classifier  # noqa: E402


class TestSSLClassifier:
    def _cfg(self):
        return {
            "model": {"num_classes": 2},
            "ssl": {
                "embed_dim": 64,
                "depth": 2,
                "num_heads": 2,
                "decoder_embed_dim": 32,
                "decoder_depth": 1,
                "decoder_num_heads": 2,
            },
        }

    def test_forward_shape(self):
        clf = build_ssl_classifier(self._cfg())
        out = clf(torch.randn(2, 1, 224, 224))
        assert out.shape == (2, 2)

    def test_encoder_weights_transfer_from_mae_checkpoint(self, tmp_path):
        cfg = self._cfg()
        mae = build_mae_model(cfg)
        ckpt = tmp_path / "mae.pt"
        torch.save({"model_state_dict": mae.state_dict()}, ckpt)
        clf = build_ssl_classifier(cfg, mae_checkpoint=ckpt)
        # patch_embed weights must match the pretrained MAE exactly
        assert torch.equal(clf.patch_embed.proj.weight, mae.patch_embed.proj.weight)
        assert torch.equal(clf.blocks[0].mlp.fc1.weight, mae.blocks[0].mlp.fc1.weight)

    def test_linear_probe_freezes_encoder(self, tmp_path):
        cfg = self._cfg()
        mae = build_mae_model(cfg)
        ckpt = tmp_path / "mae.pt"
        torch.save({"model_state_dict": mae.state_dict()}, ckpt)
        clf = build_ssl_classifier(cfg, mae_checkpoint=ckpt, freeze_encoder=True)
        assert not clf.patch_embed.proj.weight.requires_grad
        assert clf.head.weight.requires_grad

    def test_missing_checkpoint_raises(self):
        from pathlib import Path

        with pytest.raises(FileNotFoundError):
            build_ssl_classifier(self._cfg(), mae_checkpoint=Path("/nonexistent.pt"))
