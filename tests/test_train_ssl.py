"""Smoke tests for SSL pretraining and fine-tuning loops (CPU, MockHitfinder)."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from src.training.train_ssl_pretrain import run_pretrain

H, W = 512, 512
N_FRAMES = 8
LABEL_KEY = "entry_1/labels/hit"
DATA_KEY = "entry_1/data_1/data"


@pytest.fixture(scope="module")
def synthetic_cxi(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("ssl_train") / "synthetic.cxi"
    rng = np.random.default_rng(42)
    with h5py.File(path, "w") as f:
        f.create_dataset(DATA_KEY, data=rng.random((N_FRAMES, H, W)).astype(np.float32))
        f.create_dataset(
            LABEL_KEY, data=np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.float32)
        )
        det = f.create_group("entry_1/instrument_1/detector_1")
        det.create_dataset("description", data=b"Jungfrau 4M")
        det.create_dataset("distance", data=0.1)
        det.create_dataset("x_pixel_size", data=1e-4)
        f.create_dataset("entry_1/instrument_1/source_1/wavelength", data=1.3e-10)
    return path


def _tiny_cfg(ckpt_dir: Path) -> dict:
    return {
        "seed": 42,
        "ssl": {
            "masking": "random",
            "mask_ratio": 0.6,
            "peak_mask_frac": 1.0,
            "norm_pix_loss": False,
            "embed_dim": 64,
            "depth": 2,
            "num_heads": 2,
            "decoder_embed_dim": 32,
            "decoder_depth": 1,
            "decoder_num_heads": 2,
            "crops_per_frame": 1,
            "min_valid_frac": 0.5,
        },
        "training": {
            "epochs": 2,
            "warmup_epochs": 1,
            "learning_rate": 1e-4,
            "weight_decay": 0.05,
            "batch_size": 4,
            "num_workers": 0,
            "grad_clip": 1.0,
            "checkpoint_every": 1,
        },
        "hitfinder": {"backend": "mock"},
        "wandb": {"project": "sfx-hitfinder-test", "tags": ["ssl-pretrain"]},
        "checkpoint_dir": str(ckpt_dir),
    }


class TestPretrainSmoke:
    def test_two_epochs_writes_resumable_checkpoint(
        self, synthetic_cxi, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("WANDB_MODE", "disabled")
        cfg = _tiny_cfg(tmp_path / "ckpt")
        summary = run_pretrain(
            cfg,
            session_map={"s0": synthetic_cxi},
            session_ids=["s0"],
            run_name="mae-test-fold0",
            device="cpu",
        )
        ckpt = Path(cfg["checkpoint_dir"]) / "mae-test-fold0" / "last.pt"
        assert ckpt.exists()
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        for key in ("epoch", "model_state_dict", "optimizer_state_dict", "loss"):
            assert key in state
        assert summary["epochs_run"] == 2
        assert np.isfinite(summary["final_loss"])

    def test_resume_continues_from_last(self, synthetic_cxi, tmp_path, monkeypatch):
        monkeypatch.setenv("WANDB_MODE", "disabled")
        cfg = _tiny_cfg(tmp_path / "ckpt2")
        run_pretrain(cfg, {"s0": synthetic_cxi}, ["s0"], "mae-test-fold1", "cpu")
        cfg["training"]["epochs"] = 3
        summary = run_pretrain(
            cfg, {"s0": synthetic_cxi}, ["s0"], "mae-test-fold1", "cpu", resume=True
        )
        assert summary["epochs_run"] == 1  # only epoch 3 ran

    def test_peak_aware_smoke(self, synthetic_cxi, tmp_path, monkeypatch):
        monkeypatch.setenv("WANDB_MODE", "disabled")
        cfg = _tiny_cfg(tmp_path / "ckpt3")
        cfg["ssl"]["masking"] = "peak_aware"
        summary = run_pretrain(
            cfg, {"s0": synthetic_cxi}, ["s0"], "mae-test-fold2", "cpu"
        )
        assert np.isfinite(summary["final_loss"])


class TestFinetuneBuilder:
    def test_builder_produces_classifier(self, tmp_path):
        from src.models.ssl import build_mae_model
        from src.training.train_ssl_finetune import build_finetune_model_builder

        cfg = {
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
        mae = build_mae_model(cfg)
        ckpt = tmp_path / "mae.pt"
        torch.save({"model_state_dict": mae.state_dict()}, ckpt)
        builder = build_finetune_model_builder(cfg, ckpt, linear_probe=False)
        model = builder()
        assert model(torch.randn(1, 1, 224, 224)).shape == (1, 2)

    def test_linear_probe_builder_freezes(self, tmp_path):
        from src.models.ssl import build_mae_model
        from src.training.train_ssl_finetune import build_finetune_model_builder

        cfg = {
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
        mae = build_mae_model(cfg)
        ckpt = tmp_path / "mae.pt"
        torch.save({"model_state_dict": mae.state_dict()}, ckpt)
        model = build_finetune_model_builder(cfg, ckpt, linear_probe=True)()
        frozen = [p for n, p in model.named_parameters() if not n.startswith("head.")]
        assert all(not p.requires_grad for p in frozen)
