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


class TestPretrainRunNaming:
    def test_run_name_from_prefix(self, tmp_path):
        from src.training.train_ssl_pretrain import prepare_pretrain_run

        cfg = _tiny_cfg(tmp_path / "ckpt")
        assert (
            prepare_pretrain_run("mae-vits16-v2", 3, cfg)
            == "mae-vits16-v2-fold3-seed42"
        )

    @pytest.mark.parametrize(
        "bad", ["mae-vits16", "vits16-mae-v2", "mae-vits16-fold1-seed42"]
    )
    def test_invalid_prefix_exits(self, tmp_path, bad):
        from src.training.train_ssl_pretrain import prepare_pretrain_run

        with pytest.raises(SystemExit):
            prepare_pretrain_run(bad, 1, _tiny_cfg(tmp_path / "ckpt"))

    def _existing_run(self, cfg: dict) -> Path:
        run_dir = Path(cfg["checkpoint_dir"]) / "mae-vits16-v2-fold1-seed42"
        run_dir.mkdir(parents=True)
        for name in ("last.pt", "epoch20.pt", "epoch40.pt"):
            (run_dir / name).write_bytes(b"x")
        return run_dir

    def test_existing_checkpoint_without_flag_exits(self, tmp_path):
        from src.training.train_ssl_pretrain import prepare_pretrain_run

        cfg = _tiny_cfg(tmp_path / "ckpt")
        self._existing_run(cfg)
        with pytest.raises(SystemExit):
            prepare_pretrain_run("mae-vits16-v2", 1, cfg)

    def test_resume_keeps_checkpoints(self, tmp_path):
        from src.training.train_ssl_pretrain import prepare_pretrain_run

        cfg = _tiny_cfg(tmp_path / "ckpt")
        run_dir = self._existing_run(cfg)
        prepare_pretrain_run("mae-vits16-v2", 1, cfg, resume_training=True)
        assert (run_dir / "last.pt").exists()
        assert len(list(run_dir.glob("epoch*.pt"))) == 2

    def test_override_removes_last_and_epoch_snapshots(self, tmp_path):
        from src.training.train_ssl_pretrain import prepare_pretrain_run

        cfg = _tiny_cfg(tmp_path / "ckpt")
        run_dir = self._existing_run(cfg)
        prepare_pretrain_run("mae-vits16-v2", 1, cfg, override_training=True)
        assert not (run_dir / "last.pt").exists()
        assert not list(run_dir.glob("epoch*.pt"))

    def test_gate_checks_the_directory_run_pretrain_writes(
        self, synthetic_cxi, tmp_path, monkeypatch
    ):
        """The gate and run_pretrain must agree on where last.pt lives."""
        from src.training.train_ssl_pretrain import prepare_pretrain_run

        monkeypatch.setenv("WANDB_MODE", "disabled")
        cfg = _tiny_cfg(tmp_path / "ckpt")
        run_name = prepare_pretrain_run("mae-vits16-v2", 1, cfg)
        run_pretrain(cfg, {"s0": synthetic_cxi}, ["s0"], run_name, "cpu")
        with pytest.raises(SystemExit):
            prepare_pretrain_run("mae-vits16-v2", 1, cfg)
        prepare_pretrain_run("mae-vits16-v2", 1, cfg, resume_training=True)
        prepare_pretrain_run("mae-vits16-v2", 1, cfg, override_training=True)
        assert not (Path(cfg["checkpoint_dir"]) / run_name / "last.pt").exists()


class TestFinetuneRunNaming:
    def test_finetune_prefix_expansion(self, tmp_path, monkeypatch):
        from src.training.train_ssl_finetune import prepare_finetune_run

        monkeypatch.chdir(tmp_path)
        assert (
            prepare_finetune_run("vits16-mae-v2", 1, {"seed": 42})
            == "vits16-mae-finetune-v2"
        )

    def test_probe_prefix_expansion(self, tmp_path, monkeypatch):
        from src.training.train_ssl_finetune import prepare_finetune_run

        monkeypatch.chdir(tmp_path)
        assert (
            prepare_finetune_run("vits16-mae-v2", 1, {"seed": 42}, linear_probe=True)
            == "vits16-mae-probe-v2"
        )

    @pytest.mark.parametrize(
        "bad", ["vits16-mae-finetune", "vits16-mae-finetune-v2", "mae-vits16-v2"]
    )
    def test_invalid_prefix_exits(self, tmp_path, monkeypatch, bad):
        from src.training.train_ssl_finetune import prepare_finetune_run

        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit):
            prepare_finetune_run(bad, 1, {"seed": 42})

    def _existing_run(self, root: Path, run_name: str) -> Path:
        run_dir = root / "checkpoints" / run_name
        run_dir.mkdir(parents=True)
        (run_dir / "best.pt").write_bytes(b"x")
        (run_dir / "results.json").write_text("{}")
        return run_dir

    def test_existing_checkpoint_without_flag_exits(self, tmp_path, monkeypatch):
        from src.training.train_ssl_finetune import prepare_finetune_run

        monkeypatch.chdir(tmp_path)
        self._existing_run(tmp_path, "vits16-mae-finetune-v2-fold1-seed42")
        with pytest.raises(SystemExit):
            prepare_finetune_run("vits16-mae-v2", 1, {"seed": 42})

    def test_probe_gate_is_independent_of_finetune(self, tmp_path, monkeypatch):
        from src.training.train_ssl_finetune import prepare_finetune_run

        monkeypatch.chdir(tmp_path)
        self._existing_run(tmp_path, "vits16-mae-finetune-v2-fold1-seed42")
        # a finished fine-tune must not block the probe of the same fold
        prepare_finetune_run("vits16-mae-v2", 1, {"seed": 42}, linear_probe=True)

    def test_override_removes_best_and_results(self, tmp_path, monkeypatch):
        from src.training.train_ssl_finetune import prepare_finetune_run

        monkeypatch.chdir(tmp_path)
        run_dir = self._existing_run(tmp_path, "vits16-mae-probe-v2-fold4-seed42")
        prepare_finetune_run(
            "vits16-mae-v2", 4, {"seed": 42}, linear_probe=True, override_training=True
        )
        assert not (run_dir / "best.pt").exists()
        assert not (run_dir / "results.json").exists()

    def test_read_pretrain_epoch(self, tmp_path):
        from src.training.train_ssl_finetune import read_pretrain_epoch

        path = tmp_path / "last.pt"
        torch.save({"epoch": 250, "model_state_dict": {}}, path)
        assert read_pretrain_epoch(path) == 250

    def test_read_pretrain_epoch_on_a_real_pretrain_checkpoint(
        self, synthetic_cxi, tmp_path, monkeypatch
    ):
        from src.training.train_ssl_finetune import read_pretrain_epoch

        monkeypatch.setenv("WANDB_MODE", "disabled")
        cfg = _tiny_cfg(tmp_path / "ckpt")
        run_pretrain(cfg, {"s0": synthetic_cxi}, ["s0"], "mae-test-fold0", "cpu")
        ckpt = Path(cfg["checkpoint_dir"]) / "mae-test-fold0" / "last.pt"
        assert read_pretrain_epoch(ckpt) == 2

    def test_finetune_config_has_no_run_suffix(self):
        """The merged config (base.yaml + model values) must not add a name suffix."""
        from src.utils.config import load_config

        cfg_path = (
            Path(__file__).parent.parent / "configs" / "ssl" / "mae_finetune.yaml"
        )
        assert "run_suffix" not in load_config(str(cfg_path))["wandb"]

    def test_override_also_removes_a_previous_inference_result(
        self, tmp_path, monkeypatch
    ):
        from src.training.train_ssl_finetune import prepare_finetune_run

        monkeypatch.chdir(tmp_path)
        run_dir = self._existing_run(tmp_path, "vits16-mae-finetune-v2-fold1-seed42")
        (run_dir / "results.inference.json").write_text("{}")
        prepare_finetune_run("vits16-mae-v2", 1, {"seed": 42}, override_training=True)
        assert not (run_dir / "best.pt").exists()
        assert not (run_dir / "results.json").exists()
        assert not (run_dir / "results.inference.json").exists()

    def test_inference_only_needs_an_existing_checkpoint_and_deletes_nothing(
        self, tmp_path, monkeypatch
    ):
        from src.training.train_ssl_finetune import prepare_finetune_run

        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit):
            prepare_finetune_run("vits16-mae-v2", 1, {"seed": 42}, inference_only=True)
        run_dir = self._existing_run(tmp_path, "vits16-mae-finetune-v2-fold1-seed42")
        prefix = prepare_finetune_run(
            "vits16-mae-v2", 1, {"seed": 42}, inference_only=True
        )
        assert prefix == "vits16-mae-finetune-v2"
        assert (run_dir / "best.pt").exists()
        assert (run_dir / "results.json").exists()


class TestOverrideIsDeferred:
    """--override-training must not delete anything until training is about to start."""

    def test_pretrain_dry_run_keeps_checkpoints(self, tmp_path):
        from src.training.train_ssl_pretrain import prepare_pretrain_run

        cfg = _tiny_cfg(tmp_path / "ckpt")
        run_dir = Path(cfg["checkpoint_dir"]) / "mae-vits16-v2-fold1-seed42"
        run_dir.mkdir(parents=True)
        (run_dir / "last.pt").write_bytes(b"x")
        prepare_pretrain_run(
            "mae-vits16-v2", 1, cfg, override_training=True, dry_run=True
        )
        assert (run_dir / "last.pt").exists()

    def test_finetune_dry_run_keeps_checkpoints(self, tmp_path, monkeypatch):
        from src.training.train_ssl_finetune import prepare_finetune_run

        monkeypatch.chdir(tmp_path)
        run_dir = tmp_path / "checkpoints" / "vits16-mae-finetune-v2-fold1-seed42"
        run_dir.mkdir(parents=True)
        (run_dir / "best.pt").write_bytes(b"x")
        prepare_finetune_run(
            "vits16-mae-v2", 1, {"seed": 42}, override_training=True, dry_run=True
        )
        assert (run_dir / "best.pt").exists()

    def test_read_pretrain_epoch_names_the_file_when_epoch_is_missing(self, tmp_path):
        from src.training.train_ssl_finetune import read_pretrain_epoch

        path = tmp_path / "weights_only.pt"
        torch.save({"model_state_dict": {}}, path)
        with pytest.raises(SystemExit) as exc:
            read_pretrain_epoch(path)
        assert str(path) in str(exc.value)


class TestOverrideRotatesWandbId:
    @staticmethod
    def _patch_tag(monkeypatch):
        tagged = []
        monkeypatch.setattr(
            "src.training.wandb_identity.tag_overridden",
            lambda project, entity, run_id: tagged.append(run_id) or True,
        )
        return tagged

    def test_pretrain_override_rotates_wandb_id(self, tmp_path, monkeypatch):
        from src.training.train_ssl_pretrain import prepare_pretrain_run

        tagged = self._patch_tag(monkeypatch)
        cfg = _tiny_cfg(tmp_path / "ckpt")
        cfg["wandb"] = {"project": "p"}
        run_dir = Path(cfg["checkpoint_dir"]) / "mae-vits16-v2-fold1-seed42"
        run_dir.mkdir(parents=True)
        (run_dir / "last.pt").write_bytes(b"x")
        prepare_pretrain_run("mae-vits16-v2", 1, cfg, override_training=True)
        assert (run_dir / "wandb_id.txt").read_text().strip() == (
            "mae-vits16-v2-fold1-seed42-o2"
        )
        assert tagged == ["mae-vits16-v2-fold1-seed42"]

    def test_pretrain_dry_run_writes_no_id(self, tmp_path, monkeypatch):
        from src.training.train_ssl_pretrain import prepare_pretrain_run

        tagged = self._patch_tag(monkeypatch)
        cfg = _tiny_cfg(tmp_path / "ckpt")
        cfg["wandb"] = {"project": "p"}
        run_dir = Path(cfg["checkpoint_dir"]) / "mae-vits16-v2-fold1-seed42"
        run_dir.mkdir(parents=True)
        (run_dir / "last.pt").write_bytes(b"x")
        prepare_pretrain_run(
            "mae-vits16-v2", 1, cfg, override_training=True, dry_run=True
        )
        assert not (run_dir / "wandb_id.txt").exists()
        assert (run_dir / "last.pt").exists()
        assert tagged == []

    def test_finetune_override_rotates_wandb_id(self, tmp_path, monkeypatch):
        from src.training.train_ssl_finetune import prepare_finetune_run

        monkeypatch.chdir(tmp_path)
        tagged = self._patch_tag(monkeypatch)
        cfg = {"seed": 42, "wandb": {"project": "p"}}
        name = "vits16-mae-finetune-v2-fold1-seed42"
        run_dir = tmp_path / "checkpoints" / name
        run_dir.mkdir(parents=True)
        (run_dir / "best.pt").write_bytes(b"x")
        prepare_finetune_run("vits16-mae-v2", 1, cfg, override_training=True)
        assert (run_dir / "wandb_id.txt").read_text().strip() == f"{name}-o2"
        assert tagged == [name]

    def test_finetune_dry_run_writes_no_id(self, tmp_path, monkeypatch):
        from src.training.train_ssl_finetune import prepare_finetune_run

        monkeypatch.chdir(tmp_path)
        tagged = self._patch_tag(monkeypatch)
        cfg = {"seed": 42, "wandb": {"project": "p"}}
        run_dir = tmp_path / "checkpoints" / "vits16-mae-finetune-v2-fold1-seed42"
        run_dir.mkdir(parents=True)
        (run_dir / "best.pt").write_bytes(b"x")
        prepare_finetune_run(
            "vits16-mae-v2", 1, cfg, override_training=True, dry_run=True
        )
        assert not (run_dir / "wandb_id.txt").exists()
        assert tagged == []


class _InitCalled(Exception):
    def __init__(self, kwargs: dict) -> None:
        super().__init__("wandb.init called")
        self.kwargs = kwargs


class TestPretrainWandbId:
    def test_wandb_init_uses_the_recorded_id(
        self, synthetic_cxi, tmp_path, monkeypatch
    ):
        import wandb

        cfg = _tiny_cfg(tmp_path / "ckpt")
        run_dir = Path(cfg["checkpoint_dir"]) / "mae-test-fold0"
        run_dir.mkdir(parents=True)
        (run_dir / "wandb_id.txt").write_text("mae-test-fold0-o2\n")

        def _init(**kwargs):
            raise _InitCalled(kwargs)

        monkeypatch.setattr(wandb, "init", _init)
        with pytest.raises(_InitCalled) as exc:
            run_pretrain(cfg, {"s0": synthetic_cxi}, ["s0"], "mae-test-fold0", "cpu")
        assert exc.value.kwargs["id"] == "mae-test-fold0-o2"
        assert exc.value.kwargs["name"] == "mae-test-fold0"


class _ExistingRunApi:
    def __init__(self, existing, **kwargs):
        self.existing = existing

    def run(self, path):
        if path in self.existing:
            return type("R", (), {"tags": [], "update": lambda self: None})()
        raise RuntimeError("not found")


class TestPretrainFreshStartWandbId:
    def _setup(self, tmp_path, monkeypatch, existing):
        import wandb
        from src.training import wandb_identity as wi

        cfg = _tiny_cfg(tmp_path / "ckpt")
        run_dir = Path(cfg["checkpoint_dir"]) / "mae-test-fold0"
        run_dir.mkdir(parents=True)
        monkeypatch.setattr(wi, "wandb_enabled", lambda: True)
        monkeypatch.setattr(
            wandb, "Api", lambda **kw: _ExistingRunApi(existing, **kw), raising=False
        )

        def _init(**kwargs):
            raise _InitCalled(kwargs)

        monkeypatch.setattr(wandb, "init", _init)
        return cfg, run_dir

    def test_fresh_start_with_existing_run_rotates(
        self, synthetic_cxi, tmp_path, monkeypatch
    ):
        cfg, _ = self._setup(
            tmp_path, monkeypatch, {"sfx-hitfinder-test/mae-test-fold0"}
        )
        with pytest.raises(_InitCalled) as exc:
            run_pretrain(cfg, {"s0": synthetic_cxi}, ["s0"], "mae-test-fold0", "cpu")
        assert exc.value.kwargs["id"] == "mae-test-fold0-o2"

    def test_genuine_resume_keeps_current_id(
        self, synthetic_cxi, tmp_path, monkeypatch
    ):
        cfg, run_dir = self._setup(
            tmp_path, monkeypatch, {"sfx-hitfinder-test/mae-test-fold0"}
        )
        import torch

        # a real last.pt makes resume genuine; produce one via a tiny model
        from src.models.ssl import build_mae_model

        model = build_mae_model(cfg)
        opt = torch.optim.AdamW(model.parameters())
        torch.save(
            {
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
            },
            run_dir / "last.pt",
        )
        with pytest.raises(_InitCalled) as exc:
            run_pretrain(
                cfg,
                {"s0": synthetic_cxi},
                ["s0"],
                "mae-test-fold0",
                "cpu",
                resume=True,
            )
        assert exc.value.kwargs["id"] == "mae-test-fold0"
