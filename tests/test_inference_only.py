"""--inference-only: result helpers, the _train_fold inference path, CLI wiring.

No real training, SLURM or W&B: wandb is faked, the training loader raises if built.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.evaluation.benchmark import (
    SPLIT_CROSS_DETECTOR,
    SPLIT_IN_DOMAIN_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
)
from src.training import lodo
from src.training.inference_results import (
    INFERENCE_RESULTS_NAME,
    RESULTS_NAME,
    inference_block,
    inference_result_path,
    summary_updates,
    wandb_enabled,
)

REPO = Path(__file__).resolve().parent.parent
METRICS = {"ap": 0.9, "auc_roc": 0.95, "f1": 0.8, "threshold": 0.4}


class TestInferenceResultPath:
    def test_results_json_when_the_run_has_none(self, tmp_path):
        assert inference_result_path(tmp_path) == tmp_path / RESULTS_NAME

    def test_side_file_when_results_json_exists(self, tmp_path):
        (tmp_path / RESULTS_NAME).write_text("{}")
        assert inference_result_path(tmp_path) == tmp_path / INFERENCE_RESULTS_NAME

    def test_side_file_is_reused_on_a_second_inference(self, tmp_path):
        (tmp_path / RESULTS_NAME).write_text("{}")
        (tmp_path / INFERENCE_RESULTS_NAME).write_text("{}")
        assert inference_result_path(tmp_path) == tmp_path / INFERENCE_RESULTS_NAME


class TestInferenceBlock:
    def test_records_settings_and_checkpoint_provenance(self):
        block = inference_block(
            aggregation="vote",
            patch_stride=224,
            min_hit_patches=3,
            checkpoint={"epoch": 12, "val_f1": 0.77},
        )
        assert block["aggregation"] == "vote"
        assert block["patch_stride"] == 224
        assert block["min_hit_patches"] == 3
        assert block["checkpoint_epoch"] == 12
        assert block["checkpoint_val_f1"] == 0.77
        assert block["evaluated_at"].endswith("+00:00")

    def test_tolerates_a_checkpoint_without_epoch_or_f1(self):
        block = inference_block(
            aggregation="max", patch_stride=112, min_hit_patches=3, checkpoint={}
        )
        assert block["checkpoint_epoch"] is None
        assert block["checkpoint_val_f1"] is None


class TestSummaryUpdates:
    def test_keys_are_prefixed_and_cover_both_sets(self):
        updates = summary_updates(METRICS, {**METRICS, "ap": 0.5}, threshold=0.4)
        assert updates["inference/threshold"] == 0.4
        assert updates["inference/in_domain/ap"] == 0.9
        assert updates["inference/in_domain/auc"] == 0.95
        assert updates["inference/cross/ap"] == 0.5
        assert updates["inference/cross/f1"] == 0.8
        assert updates["inference/cross/threshold"] == 0.4
        assert all(k.startswith("inference/") for k in updates)


class TestWandbEnabled:
    @pytest.mark.parametrize("mode", ["offline", "disabled", "OFFLINE", "Disabled"])
    def test_off_for_offline_modes(self, monkeypatch, mode):
        monkeypatch.setenv("WANDB_MODE", mode)
        assert wandb_enabled() is False

    @pytest.mark.parametrize("mode", [None, "", "online"])
    def test_on_otherwise(self, monkeypatch, mode):
        if mode is None:
            monkeypatch.delenv("WANDB_MODE", raising=False)
        else:
            monkeypatch.setenv("WANDB_MODE", mode)
        assert wandb_enabled() is True


RUN_PREFIX = "resnet18-asymmetric-v2"
RUN_NAME = f"{RUN_PREFIX}-fold1-seed42"


class _FakeRun:
    def __init__(self) -> None:
        self.summary: dict = {}


class _FakeWandb:
    """Records calls; `init` returns a run whose `summary` is a plain dict."""

    def __init__(self) -> None:
        self.run = _FakeRun()
        self.init_calls: list[dict] = []
        self.logged: list = []
        self.finished = False

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return self.run

    def Settings(self, **kwargs) -> dict:  # noqa: N802 - mirrors wandb.Settings
        return kwargs

    def define_metric(self, *args, **kwargs) -> None:
        pass

    def log(self, *args, **kwargs) -> None:
        self.logged.append((args, kwargs))

    def finish(self) -> None:
        self.finished = True


def _cfg() -> dict:
    return {
        "seed": 42,
        "model": {"backbone": "resnet18", "pretrained": False, "num_classes": 2},
        "training": {"batch_size": 2, "num_workers": 0, "epochs": 1},
        "lodo": {},
        "hitfinder": {"backend": "mock"},
        "wandb": {"project": "scratch"},
    }


def _fold_args() -> dict:
    return dict(
        fold={"fold_id": 1, "test_detector": "AGIPD"},
        split_artifact={
            "splits": {
                "s_train": SPLIT_TRAIN,
                "s_val": SPLIT_VAL,
                "s_in": SPLIT_IN_DOMAIN_TEST,
                "s_cross": SPLIT_CROSS_DETECTOR,
            }
        },
        session_map={},
        hitfinder=None,
        device="cpu",
        run_name_prefix=RUN_PREFIX,
    )


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """chdir to a temp dir, fake wandb, make building the training loader fatal."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)

    def _no_training_loader(**kwargs):
        raise AssertionError("the training loader must not be built")

    monkeypatch.setattr(lodo, "asymmetric_loader", _no_training_loader)
    monkeypatch.setattr(lodo, "build_supervised_model", lambda **kw: nn.Linear(4, 2))
    monkeypatch.setattr(lodo, "run_patch_agg", lambda *a, **kw: dict(METRICS))
    ckpt_dir = tmp_path / "checkpoints" / RUN_NAME

    def make_checkpoint(backbone: str = "resnet18", results: str | None = None):
        ckpt_dir.mkdir(parents=True)
        torch.save(
            {
                "epoch": 7,
                "model_state_dict": nn.Linear(4, 2).state_dict(),
                "val_f1": 0.61,
                "inference_threshold": 0.4,
                "backbone": backbone,
                "num_classes": 2,
            },
            ckpt_dir / "best.pt",
        )
        if results is not None:
            (ckpt_dir / RESULTS_NAME).write_text(results)

    return fake, ckpt_dir, make_checkpoint


class TestTrainFoldInferenceOnly:
    def test_skips_the_training_loader_and_writes_results_json_when_absent(
        self, harness
    ):
        fake, ckpt_dir, make_checkpoint = harness
        make_checkpoint()
        result = lodo._train_fold(cfg=_cfg(), inference_only=True, **_fold_args())
        data = json.loads((ckpt_dir / RESULTS_NAME).read_text())
        assert data["fold_id"] == 1
        assert data["inference_threshold"] == 0.4
        assert data["cross"]["ap"] == 0.9
        assert data["in_domain"]["f1"] == 0.8
        assert "inference" not in data
        assert not (ckpt_dir / INFERENCE_RESULTS_NAME).exists()
        assert result["ap"] == 0.9

    def test_keeps_an_existing_results_json_and_writes_the_side_file(self, harness):
        fake, ckpt_dir, make_checkpoint = harness
        original = '{"fold_id": 1, "cross": {"ap": 0.123}}'
        make_checkpoint(results=original)
        lodo._train_fold(cfg=_cfg(), inference_only=True, **_fold_args())
        assert (ckpt_dir / RESULTS_NAME).read_text() == original
        side = json.loads((ckpt_dir / INFERENCE_RESULTS_NAME).read_text())
        assert side["cross"]["ap"] == 0.9
        assert side["inference"]["aggregation"] == "vote"
        assert side["inference"]["checkpoint_epoch"] == 7

    def test_a_second_inference_overwrites_only_the_side_file(
        self, harness, monkeypatch
    ):
        fake, ckpt_dir, make_checkpoint = harness
        make_checkpoint(results="{}")
        state = {"ap": 0.9}
        monkeypatch.setattr(
            lodo, "run_patch_agg", lambda *a, **kw: {**METRICS, "ap": state["ap"]}
        )
        lodo._train_fold(cfg=_cfg(), inference_only=True, **_fold_args())
        first = json.loads((ckpt_dir / INFERENCE_RESULTS_NAME).read_text())
        state["ap"] = 0.5
        lodo._train_fold(cfg=_cfg(), inference_only=True, **_fold_args())
        second = json.loads((ckpt_dir / INFERENCE_RESULTS_NAME).read_text())
        assert first["cross"]["ap"] == 0.9
        assert second["cross"]["ap"] == 0.5
        assert (ckpt_dir / RESULTS_NAME).read_text() == "{}"

    def test_writes_summary_keys_and_logs_no_training_metrics(self, harness):
        fake, ckpt_dir, make_checkpoint = harness
        make_checkpoint()
        lodo._train_fold(cfg=_cfg(), inference_only=True, **_fold_args())
        assert len(fake.init_calls) == 1
        init = fake.init_calls[0]
        assert init["id"] == RUN_NAME and init["name"] == RUN_NAME
        assert init["resume"] == "allow"
        assert "config" not in init  # never rewrite the closed run's config
        # the resumed run must not re-upload metadata, console output, stats or code
        assert init["settings"] == {
            "console": "off",
            "x_disable_stats": True,
            "x_disable_meta": True,
            "save_code": False,
        }
        assert fake.run.summary["inference/cross/ap"] == 0.9
        assert fake.run.summary["inference/in_domain/auc"] == 0.95
        assert fake.logged == []
        assert fake.finished is True

    @pytest.mark.parametrize("mode", ["disabled", "offline"])
    def test_sends_nothing_when_wandb_is_off(self, harness, monkeypatch, mode):
        fake, ckpt_dir, make_checkpoint = harness
        monkeypatch.setenv("WANDB_MODE", mode)
        make_checkpoint()
        lodo._train_fold(cfg=_cfg(), inference_only=True, **_fold_args())
        assert fake.init_calls == []
        assert (ckpt_dir / RESULTS_NAME).exists()

    def test_a_failed_evaluation_leaves_the_wandb_run_untouched(
        self, harness, monkeypatch
    ):
        fake, ckpt_dir, make_checkpoint = harness
        make_checkpoint()

        def _boom(*args, **kwargs):
            raise RuntimeError("evaluation failed")

        monkeypatch.setattr(lodo, "run_patch_agg", _boom)
        with pytest.raises(RuntimeError, match="evaluation failed"):
            lodo._train_fold(cfg=_cfg(), inference_only=True, **_fold_args())
        assert fake.init_calls == []  # a crashed pass must not mark the closed run
        assert not (ckpt_dir / RESULTS_NAME).exists()

    def test_backbone_mismatch_is_still_rejected(self, harness):
        fake, ckpt_dir, make_checkpoint = harness
        make_checkpoint(backbone="resnet50")
        with pytest.raises(RuntimeError, match="backbone"):
            lodo._train_fold(cfg=_cfg(), inference_only=True, **_fold_args())

    def test_missing_checkpoint_raises_and_creates_nothing(self, harness, tmp_path):
        fake, ckpt_dir, make_checkpoint = harness
        with pytest.raises(FileNotFoundError, match="best.pt"):
            lodo._train_fold(cfg=_cfg(), inference_only=True, **_fold_args())
        assert not ckpt_dir.exists()


class TestTrainFoldRefusesSilentRetraining:
    def test_existing_checkpoint_without_a_mode_is_refused(self, harness):
        fake, ckpt_dir, make_checkpoint = harness
        make_checkpoint()
        with pytest.raises(RuntimeError, match="resume_training or inference_only"):
            lodo._train_fold(cfg=_cfg(), **_fold_args())


def _run_module(module: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )


class TestTrack1Cli:
    BASE = [
        "--config",
        "configs/supervised/resnet18_asymmetric.yaml",
        "--run-name-prefix",
        "resnet18-asymmetric-v2",
    ]

    @pytest.mark.parametrize("other", ["--resume-training", "--override-training"])
    def test_flag_is_mutually_exclusive_with_the_other_two(self, other):
        proc = _run_module(
            "src.training.train_asymmetric", *self.BASE, "--inference-only", other
        )
        assert proc.returncode == 2
        assert "not allowed with argument" in proc.stderr

    def test_help_lists_the_flag(self):
        proc = _run_module("src.training.train_asymmetric", "--help")
        assert "--inference-only" in proc.stdout


class TestTrack1Wrapper:
    def test_inference_only_passes_with_an_existing_checkpoint(
        self, tmp_path, monkeypatch
    ):
        from src.training.train_asymmetric import _check_checkpoint_collisions

        monkeypatch.chdir(tmp_path)
        run_dir = tmp_path / "checkpoints" / RUN_NAME
        run_dir.mkdir(parents=True)
        (run_dir / "best.pt").write_bytes(b"x")
        (run_dir / RESULTS_NAME).write_text("{}")
        _check_checkpoint_collisions(
            [1], _cfg(), RUN_PREFIX, False, False, inference_only=True
        )
        assert (run_dir / "best.pt").exists()
        assert (run_dir / RESULTS_NAME).exists()

    def test_inference_only_without_a_checkpoint_exits(self, tmp_path, monkeypatch):
        from src.training.train_asymmetric import _check_checkpoint_collisions

        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit):
            _check_checkpoint_collisions(
                [1], _cfg(), RUN_PREFIX, False, False, inference_only=True
            )


class TestSslFinetuneCli:
    BASE = [
        "--config",
        "configs/ssl/mae_finetune.yaml",
        "--fold",
        "1",
        "--run-name-prefix",
        "vits16-mae-v2",
    ]

    @pytest.mark.parametrize("other", ["--resume-training", "--override-training"])
    def test_flag_is_mutually_exclusive_with_the_other_two(self, other):
        proc = _run_module(
            "src.training.train_ssl_finetune", *self.BASE, "--inference-only", other
        )
        assert proc.returncode == 2
        assert "not allowed with argument" in proc.stderr

    def test_pretrain_checkpoint_is_required_without_the_flag(self):
        proc = _run_module("src.training.train_ssl_finetune", *self.BASE)
        assert proc.returncode == 2
        assert "--pretrain-checkpoint" in proc.stderr

    def test_help_lists_the_flag(self):
        proc = _run_module("src.training.train_ssl_finetune", "--help")
        assert "--inference-only" in proc.stdout


def test_pretrain_cli_has_no_inference_flag():
    proc = _run_module(
        "src.training.train_ssl_pretrain",
        "--config",
        "configs/ssl/mae_pretrain.yaml",
        "--fold",
        "1",
        "--run-name-prefix",
        "mae-vits16-v2",
        "--inference-only",
    )
    assert proc.returncode == 2
    assert "unrecognized arguments: --inference-only" in proc.stderr
