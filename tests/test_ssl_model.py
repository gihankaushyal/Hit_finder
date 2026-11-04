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


class TestSSLConfigs:
    def test_pretrain_config_loads_and_merges_base(self):
        from src.utils.config import load_config

        cfg = load_config("configs/ssl/mae_pretrain.yaml")
        assert cfg["ssl"]["masking"] == "random"
        assert cfg["ssl"]["mask_ratio"] == 0.6
        assert cfg["seed"] == 42  # from base.yaml deep-merge

    def test_finetune_config_loads(self):
        from src.utils.config import load_config

        cfg = load_config("configs/ssl/mae_finetune.yaml")
        assert cfg["model"]["num_classes"] == 2
        assert cfg["benchmark"]["aggregation"] == "vote"

    def test_pretrain_config_builds_mae_model(self):
        from src.utils.config import load_config

        cfg = load_config("configs/ssl/mae_pretrain.yaml")
        model = build_mae_model(cfg)
        assert isinstance(model, MAEViT)


class TestFineTuneRecipe:
    """Opt-in recipe knobs: drop_path, head init, LP-FT head copy, layer decay, schedule."""

    @staticmethod
    def _cfg(**ssl):
        return {"ssl": {"embed_dim": 64, "depth": 4, "num_heads": 4, **ssl}}

    def test_defaults_leave_state_dict_and_optimizer_unchanged(self):
        from src.models.ssl import build_ssl_classifier
        from src.training.recipe import build_optimizer

        m = build_ssl_classifier(self._cfg())
        assert not any("drop_path" in k for k in m.state_dict())
        opt = build_optimizer(m, {"learning_rate": 1e-4, "weight_decay": 1e-4})
        assert len(opt.param_groups) == 1
        assert opt.param_groups[0]["weight_decay"] == 1e-4

    def test_drop_path_grows_with_depth_and_is_off_in_eval(self):
        from timm.layers import DropPath

        from src.models.ssl import build_ssl_classifier

        m = build_ssl_classifier(self._cfg(drop_path_rate=0.3))
        # rate 0.0 collapses to Identity in timm, so block 0 has no DropPath at all
        assert not isinstance(m.blocks[0].drop_path1, DropPath)
        rates = [b.drop_path1.drop_prob for b in m.blocks[1:]]
        assert rates == sorted(rates) and rates[-1] == pytest.approx(0.3)
        m.eval()
        x = torch.randn(2, 1, 224, 224)
        assert torch.equal(m(x), m(x))

    def test_trunc_normal_head_init_is_near_zero(self):
        from src.models.ssl import build_ssl_classifier

        m = build_ssl_classifier(self._cfg(head_init="trunc_normal"))
        assert m.head.weight.abs().max() < 1e-3
        assert torch.all(m.head.bias == 0)

    def test_head_checkpoint_copies_only_head(self, tmp_path):
        from src.models.ssl import build_ssl_classifier

        src = build_ssl_classifier(self._cfg())
        torch.save({"model_state_dict": src.state_dict()}, tmp_path / "probe.pt")
        dst = build_ssl_classifier(self._cfg(), head_checkpoint=tmp_path / "probe.pt")
        assert torch.equal(dst.head.weight, src.head.weight)
        assert not torch.equal(
            dst.blocks[0].attn.qkv.weight, src.blocks[0].attn.qkv.weight
        )

    def test_layer_decay_groups_cover_each_param_once_and_scale_by_depth(self):
        from src.models.ssl import build_ssl_classifier
        from src.training.recipe import build_optimizer

        m = build_ssl_classifier(self._cfg())
        cfg = {"learning_rate": 1.0, "weight_decay": 0.05, "layer_decay": 0.5}
        opt = build_optimizer(m, cfg)
        grouped = [p for g in opt.param_groups for p in g["params"]]
        trainable = [p for p in m.parameters() if p.requires_grad]
        assert len(grouped) == len(trainable) == len({id(p) for p in grouped})
        lrs = {g["lr"] for g in opt.param_groups}
        assert max(lrs) == 1.0 and min(lrs) == pytest.approx(
            0.5**5
        )  # depth 4 -> 5 steps
        for g in opt.param_groups:
            if g["weight_decay"] == 0.0:
                assert all(p.ndim == 1 or p is m.cls_token for p in g["params"])

    def test_frozen_params_are_excluded_from_groups(self):
        from src.models.ssl import build_ssl_classifier
        from src.training.recipe import build_optimizer

        m = build_ssl_classifier(self._cfg(), freeze_encoder=True)
        opt = build_optimizer(
            m, {"learning_rate": 1.0, "weight_decay": 0.05, "layer_decay": 0.75}
        )
        assert (
            sum(len(g["params"]) for g in opt.param_groups) == 2
        )  # head weight + bias

    def test_warmup_then_cosine_shape(self):
        from src.training.recipe import lr_factor

        cfg = {"warmup_epochs": 5, "lr_schedule": "cosine"}
        f = [lr_factor(e, 50, cfg) for e in range(1, 51)]
        assert f[0] == pytest.approx(0.2) and f[4] == pytest.approx(1.0)
        assert f[5] == pytest.approx(1.0)
        assert all(a >= b for a, b in zip(f[5:], f[6:]))
        assert 0.0 < f[-1] < 0.01

    def test_default_schedule_is_noop(self):
        from src.models.ssl import build_ssl_classifier
        from src.training.recipe import apply_lr_schedule, build_optimizer

        m = build_ssl_classifier(self._cfg())
        cfg = {"learning_rate": 1e-4, "weight_decay": 1e-4}
        opt = build_optimizer(m, cfg)
        apply_lr_schedule(opt, 7, 100, cfg)
        assert opt.param_groups[0]["lr"] == 1e-4


class TestFineTuneCliGuards:
    """--head-checkpoint / --random-init must not be able to collide with a real v2 run."""

    @staticmethod
    def _run(monkeypatch, *extra):
        import sys

        from src.training import train_ssl_finetune

        argv = [
            "x", "--config", "configs/ssl/mae_finetune.yaml", "--fold", "4",
            "--run-name-prefix", "vits16-mae-v2", *extra,
        ]  # fmt: skip
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as e:
            train_ssl_finetune.main()
        return e.value.code

    def test_head_checkpoint_requires_run_suffix(self, monkeypatch, tmp_path):
        assert self._run(monkeypatch, "--head-checkpoint", str(tmp_path / "p.pt"),
                         "--pretrain-checkpoint", "x.pt") == 2  # fmt: skip

    def test_empty_or_bad_run_suffix_rejected(self, monkeypatch):
        for bad in ("-", "a/b", "has space"):
            assert self._run(monkeypatch, "--pretrain-checkpoint", "x.pt",
                             "--run-suffix", bad) == 2  # fmt: skip

    def test_recipe_config_requires_run_suffix(self, monkeypatch):
        import sys

        from src.training import train_ssl_finetune

        argv = [
            "x", "--config", "configs/ssl/mae_finetune_careful.yaml", "--fold", "4",
            "--run-name-prefix", "vits16-mae-finetune-v2",
            "--pretrain-checkpoint", "x.pt",
        ]  # fmt: skip
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as e:
            train_ssl_finetune.main()
        assert e.value.code == 2

    def test_default_v2_config_has_no_recipe_keys(self):
        from src.training.train_ssl_finetune import non_default_recipe_keys
        from src.utils.config import load_config

        assert (
            non_default_recipe_keys(load_config("configs/ssl/mae_finetune.yaml")) == []
        )
        careful = non_default_recipe_keys(
            load_config("configs/ssl/mae_finetune_careful.yaml")
        )
        assert {"warmup_epochs", "lr_schedule", "layer_decay", "drop_path_rate"} <= set(
            careful
        )

    def test_head_checkpoint_from_another_fold_is_rejected(self, tmp_path):
        from src.models.ssl import build_ssl_classifier
        from src.training.train_ssl_finetune import check_head_checkpoint

        cfg = {"ssl": {"embed_dim": 64, "depth": 2, "num_heads": 4}}
        full = build_ssl_classifier(cfg).state_dict()
        run = tmp_path / "vits16-mae-probe-v2-fold4-seed42"
        run.mkdir()
        torch.save(
            {"model_state_dict": {k: full[k] for k in ("head.weight", "head.bias")}},
            run / "best.pt",
        )
        check_head_checkpoint(run / "best.pt", 4)  # same fold: fine
        with pytest.raises(SystemExit, match="cross-detector leak"):
            check_head_checkpoint(run / "best.pt", 1)

    def test_head_checkpoint_with_unknown_fold_is_rejected(self, tmp_path):
        from src.models.ssl import build_ssl_classifier
        from src.training.train_ssl_finetune import check_head_checkpoint

        cfg = {"ssl": {"embed_dim": 64, "depth": 2, "num_heads": 4}}
        full = build_ssl_classifier(cfg).state_dict()
        torch.save(
            {"model_state_dict": {k: full[k] for k in ("head.weight", "head.bias")}},
            tmp_path / "copied_best.pt",
        )
        with pytest.raises(SystemExit, match="cannot tell which fold"):
            check_head_checkpoint(tmp_path / "copied_best.pt", 4)

    def test_hparam_only_change_requires_run_suffix(self):
        from src.training.train_ssl_finetune import non_default_recipe_keys
        from src.utils.config import load_config

        ref = load_config("configs/ssl/mae_finetune.yaml")
        assert non_default_recipe_keys(ref, ref) == []
        lr_sweep = load_config("configs/ssl/mae_finetune.yaml")
        lr_sweep["training"]["learning_rate"] = ref["training"]["learning_rate"] * 2
        assert non_default_recipe_keys(lr_sweep, ref) == ["learning_rate"]

    def test_probe_ignores_drop_path(self):
        from src.models.ssl import build_ssl_classifier

        cfg = {
            "ssl": {"embed_dim": 64, "depth": 2, "num_heads": 4, "drop_path_rate": 0.1}
        }
        probe = build_ssl_classifier(cfg, freeze_encoder=True)
        probe.train()
        assert not any(
            type(m).__name__ == "DropPath" for m in probe.modules()
        ), "frozen encoder must not carry stochastic depth"
        assert any(
            type(m).__name__ == "DropPath" for m in build_ssl_classifier(cfg).modules()
        )

    def test_partial_head_checkpoint_is_rejected(self, tmp_path):
        from src.models.ssl import build_ssl_classifier

        cfg = {"ssl": {"embed_dim": 64, "depth": 2, "num_heads": 4}}
        full = build_ssl_classifier(cfg).state_dict()
        torch.save(
            {"model_state_dict": {"head.weight": full["head.weight"]}},
            tmp_path / "h.pt",
        )
        with pytest.raises(RuntimeError, match="head keys"):
            build_ssl_classifier(cfg, head_checkpoint=tmp_path / "h.pt")


class TestCarefulConfigs:
    def test_diag_config_differs_only_in_diagnostic_keys(self):
        import yaml

        def load(name):
            with open(f"configs/ssl/{name}") as f:
                return yaml.safe_load(f)

        base, diag = load("mae_finetune_careful.yaml"), load(
            "mae_finetune_careful_diag.yaml"
        )
        extra = {"log_val_breakdown", "save_epochs"}
        assert set(diag["training"]) - set(base["training"]) == extra
        diag["training"] = {k: v for k, v in diag["training"].items() if k not in extra}
        assert diag == base
