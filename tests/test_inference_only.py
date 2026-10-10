"""--inference-only: result helpers, the _train_fold inference path, CLI wiring.

No real training, SLURM or W&B: wandb is faked, the training loader raises if built.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
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
    LOSS_FLAT_REL_TOL,
    LOSS_FLAT_WINDOW,
    RESULTS_NAME,
    assess_training,
    fetch_wandb_history,
    inference_block,
    inference_result_path,
    loss_flatness,
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
            training_check={"status": "unknown"},
        )
        assert block["aggregation"] == "vote"
        assert block["patch_stride"] == 224
        assert block["min_hit_patches"] == 3
        assert block["checkpoint_epoch"] == 12
        assert block["checkpoint_val_f1"] == 0.77
        assert block["evaluated_at"].endswith("+00:00")
        assert block["training_check"] == {"status": "unknown"}

    def test_tolerates_a_checkpoint_without_epoch_or_f1(self):
        block = inference_block(
            aggregation="max",
            patch_stride=112,
            min_hit_patches=3,
            checkpoint={},
            training_check={"status": "unknown"},
        )
        assert block["checkpoint_epoch"] is None
        assert block["checkpoint_val_f1"] is None


def _history(state="crashed", losses=None, f1s=None, start_epoch=1):
    losses = losses if losses is not None else [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    f1s = f1s if f1s is not None else [0.5] * len(losses)
    return {
        "state": state,
        "epochs": list(range(start_epoch, start_epoch + len(losses))),
        "train_loss": losses,
        "val_f1": f1s,
    }


class TestLossFlatness:
    def test_still_decreasing_is_not_flat(self):
        out = loss_flatness([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
        assert out["flat"] is False
        assert out["window"] == LOSS_FLAT_WINDOW
        assert out["relative_improvement"] == pytest.approx((0.9 - 0.5) / 0.9)

    def test_a_plateau_is_flat(self):
        out = loss_flatness([1.0, 0.5, 0.300, 0.2995, 0.2992, 0.2990, 0.2989])
        assert out["flat"] is True
        assert out["relative_improvement"] < LOSS_FLAT_REL_TOL

    def test_the_boundary_is_not_flat(self):
        # exactly LOSS_FLAT_REL_TOL improvement over the window -> still improving
        start = 1.0
        end = start * (1 - LOSS_FLAT_REL_TOL)
        out = loss_flatness([start, 0.995, 0.993, 0.992, end])
        assert out["flat"] is False

    def test_a_rising_loss_is_flat(self):
        assert loss_flatness([0.5, 0.5, 0.5, 0.6, 0.7])["flat"] is True

    def test_too_few_points_is_unknown(self):
        assert loss_flatness([1.0, 0.9, 0.8, 0.7]) is None

    def test_non_finite_values_are_ignored(self):
        out = loss_flatness([1.0, float("nan"), 0.9, 0.8, 0.7, 0.6, 0.5])
        assert out["flat"] is False

    def test_zero_start_does_not_divide_by_zero(self):
        assert loss_flatness([0.0] * 6)["flat"] is True


class TestAssessTraining:
    def test_finished_run_is_complete_and_silent(self):
        check = assess_training(
            configured_epochs=100,
            checkpoint_epoch=40,
            history=_history(state="finished"),
        )
        assert check["status"] == "complete"
        assert check["warnings"] == []
        assert check["configured_epochs"] == 100
        assert check["checkpoint_epoch"] == 40

    def test_crashed_run_still_improving_warns_about_an_undertrained_checkpoint(self):
        check = assess_training(
            configured_epochs=100, checkpoint_epoch=9, history=_history()
        )
        assert check["status"] == "incomplete"
        assert check["wandb_state"] == "crashed"
        assert check["last_logged_epoch"] == 7
        assert check["train_loss_flat"]["flat"] is False
        text = " ".join(check["warnings"])
        assert "did not finish" in text and "7/100" in text
        assert "still decreasing" in text

    def test_crashed_run_with_flat_loss_says_so(self):
        losses = [1.0, 0.5, 0.3, 0.2990, 0.2989, 0.2988, 0.2988, 0.2987]
        check = assess_training(
            configured_epochs=100,
            checkpoint_epoch=6,
            history=_history(losses=losses),
        )
        assert check["status"] == "incomplete"
        assert check["train_loss_flat"]["flat"] is True
        assert "flattened" in " ".join(check["warnings"])

    def test_crashed_run_with_too_few_epochs_cannot_judge_the_trend(self):
        check = assess_training(
            configured_epochs=100,
            checkpoint_epoch=2,
            history=_history(losses=[1.0, 0.9]),
        )
        assert check["train_loss_flat"] is None
        assert "too few" in " ".join(check["warnings"])

    def test_no_history_and_an_early_checkpoint_warns_it_cannot_verify(self):
        check = assess_training(configured_epochs=100, checkpoint_epoch=9, history=None)
        assert check["status"] == "unknown"
        text = " ".join(check["warnings"])
        assert "epoch 9 of 100" in text and "cannot verify" in text

    def test_no_history_and_the_last_epoch_checkpoint_is_quiet(self):
        check = assess_training(
            configured_epochs=100, checkpoint_epoch=100, history=None
        )
        assert check["status"] == "unknown"
        assert check["warnings"] == []

    def test_missing_checkpoint_epoch_is_tolerated(self):
        check = assess_training(
            configured_epochs=100, checkpoint_epoch=None, history=None
        )
        assert check["status"] == "unknown"
        assert check["checkpoint_epoch"] is None


class _FakeApiRun:
    def __init__(self, state="crashed", rows=None):
        self.state = state
        self._rows = (
            rows
            if rows is not None
            else [
                {"epoch": e, "train/loss": 1.0 / e, "val/f1": 0.1 * e}
                for e in (3, 1, 2)
            ]
        )

    def scan_history(self, keys=None):
        self.scan_keys = keys
        return iter(self._rows)


class _FakeApiWandb:
    """wandb stand-in exposing a read-only Api; records the requested path."""

    def __init__(self, run=None, error=None):
        self._run = run or _FakeApiRun()
        self._error = error
        self.paths: list[str] = []
        self.run_inits = 0

    def Api(self, timeout=None):  # noqa: N802 - mirrors wandb.Api
        outer = self

        class _Api:
            def run(self, path):
                outer.paths.append(path)
                if outer._error:
                    raise outer._error
                return outer._run

        return _Api()

    def init(self, **kwargs):  # must never be used to read history
        self.run_inits += 1

    def setup(self):
        return types.SimpleNamespace(settings=types.SimpleNamespace(mode="online"))


class TestFetchWandbHistory:
    def test_reads_state_and_sorted_per_epoch_series_read_only(self, monkeypatch):
        monkeypatch.delenv("WANDB_MODE", raising=False)
        fake = _FakeApiWandb()
        monkeypatch.setitem(sys.modules, "wandb", fake)
        out = fetch_wandb_history("proj", "ent", "run-1")
        assert fake.paths == ["ent/proj/run-1"]
        assert fake.run_inits == 0
        assert out["state"] == "crashed"
        assert out["epochs"] == [1, 2, 3]
        assert out["train_loss"] == [1.0, 0.5, pytest.approx(1 / 3)]
        assert out["val_f1"] == [
            pytest.approx(0.1),
            pytest.approx(0.2),
            pytest.approx(0.3),
        ]

    def test_no_entity_uses_the_default_path(self, monkeypatch):
        monkeypatch.delenv("WANDB_MODE", raising=False)
        fake = _FakeApiWandb()
        monkeypatch.setitem(sys.modules, "wandb", fake)
        fetch_wandb_history("proj", None, "run-1")
        assert fake.paths == ["proj/run-1"]

    def test_any_api_failure_returns_none(self, monkeypatch, capsys):
        monkeypatch.delenv("WANDB_MODE", raising=False)
        monkeypatch.setitem(
            sys.modules, "wandb", _FakeApiWandb(error=ConnectionError("no network"))
        )
        assert fetch_wandb_history("proj", None, "run-1") is None
        assert "could not read the W&B history" in capsys.readouterr().out

    @pytest.mark.parametrize("mode", ["offline", "disabled"])
    def test_offline_modes_never_touch_the_network(self, monkeypatch, mode):
        monkeypatch.setenv("WANDB_MODE", mode)
        fake = _FakeApiWandb()
        monkeypatch.setitem(sys.modules, "wandb", fake)
        assert fetch_wandb_history("proj", None, "run-1") is None
        assert fake.paths == []


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


class _SetupWandb:
    """Stand-in for the wandb module: `setup().settings.mode` is configurable."""

    def __init__(self, mode: str | None = "online", broken: bool = False) -> None:
        self._mode = mode
        self._broken = broken

    def setup(self):
        if self._broken:
            raise RuntimeError("wandb service unavailable")
        return types.SimpleNamespace(settings=types.SimpleNamespace(mode=self._mode))


class TestWandbEnabled:
    @pytest.mark.parametrize("mode", ["offline", "disabled", "OFFLINE", "Disabled"])
    def test_off_for_offline_env_modes(self, monkeypatch, mode):
        monkeypatch.setenv("WANDB_MODE", mode)
        monkeypatch.setitem(sys.modules, "wandb", _SetupWandb("online"))
        assert wandb_enabled() is False

    @pytest.mark.parametrize("env", [None, "", "online"])
    def test_on_when_nothing_turns_it_off(self, monkeypatch, env):
        if env is None:
            monkeypatch.delenv("WANDB_MODE", raising=False)
        else:
            monkeypatch.setenv("WANDB_MODE", env)
        monkeypatch.setitem(sys.modules, "wandb", _SetupWandb("online"))
        assert wandb_enabled() is True

    @pytest.mark.parametrize("mode", ["offline", "disabled"])
    def test_off_when_set_with_the_wandb_offline_command(self, monkeypatch, mode):
        """`wandb offline` stores the mode in a settings file, not in the environment."""
        monkeypatch.delenv("WANDB_MODE", raising=False)
        monkeypatch.setitem(sys.modules, "wandb", _SetupWandb(mode))
        assert wandb_enabled() is False

    def test_assumed_on_when_the_settings_cannot_be_read(self, monkeypatch):
        monkeypatch.delenv("WANDB_MODE", raising=False)
        monkeypatch.setitem(sys.modules, "wandb", _SetupWandb(broken=True))
        assert wandb_enabled() is True


RUN_PREFIX = "resnet18-asymmetric-v2"
RUN_NAME = f"{RUN_PREFIX}-fold1-seed42"
BUILD_KWARGS: list[dict] = []


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
        self.fail_init = False

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        if self.fail_init:
            raise ConnectionError("no network")
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
    BUILD_KWARGS.clear()

    def _build(**kwargs):
        BUILD_KWARGS.append(kwargs)
        return nn.Linear(4, 2)

    monkeypatch.setattr(lodo, "build_supervised_model", _build)
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
        # the provenance block is always written, so a provisional results.json is
        # distinguishable from one produced by a finished training run
        assert data["inference"]["checkpoint_epoch"] == 7
        assert data["inference"]["training_check"]["status"] == "unknown"
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
        assert "training_check" in side["inference"]

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

    def test_an_unfinished_run_is_flagged_in_the_block_and_warned_about(
        self, harness, monkeypatch, capsys
    ):
        fake, ckpt_dir, make_checkpoint = harness
        make_checkpoint()
        monkeypatch.setattr(lodo, "fetch_wandb_history", lambda *a, **kw: _history())
        lodo._train_fold(cfg=_cfg(), inference_only=True, **_fold_args())
        data = json.loads((ckpt_dir / RESULTS_NAME).read_text())
        check = data["inference"]["training_check"]
        assert check["status"] == "incomplete"
        assert check["checkpoint_epoch"] == 7
        assert check["configured_epochs"] == 1
        out = capsys.readouterr().out
        assert "[inference] WARNING" in out
        assert "did not finish" in out

    def test_a_finished_run_prints_no_warning(self, harness, monkeypatch, capsys):
        fake, ckpt_dir, make_checkpoint = harness
        make_checkpoint()
        monkeypatch.setattr(
            lodo, "fetch_wandb_history", lambda *a, **kw: _history(state="finished")
        )
        lodo._train_fold(cfg=_cfg(), inference_only=True, **_fold_args())
        assert "WARNING" not in capsys.readouterr().out
        data = json.loads((ckpt_dir / RESULTS_NAME).read_text())
        assert data["inference"]["training_check"]["status"] == "complete"

    def test_the_history_is_read_for_the_resolved_run_name(self, harness, monkeypatch):
        fake, ckpt_dir, make_checkpoint = harness
        make_checkpoint()
        seen = {}

        def _spy(project, entity, run_id):
            seen.update(project=project, entity=entity, run_id=run_id)

        monkeypatch.setattr(lodo, "fetch_wandb_history", _spy)
        lodo._train_fold(cfg=_cfg(), inference_only=True, **_fold_args())
        assert seen == {"project": "scratch", "entity": None, "run_id": RUN_NAME}

    def test_a_wandb_failure_never_loses_the_result(self, harness, capsys):
        fake, ckpt_dir, make_checkpoint = harness
        fake.fail_init = True
        make_checkpoint()
        result = lodo._train_fold(cfg=_cfg(), inference_only=True, **_fold_args())
        assert (ckpt_dir / RESULTS_NAME).exists()  # written before W&B is touched
        assert result["ap"] == 0.9
        assert (
            "[wandb] could not record the inference summary" in capsys.readouterr().out
        )

    def test_no_pretrained_weights_are_requested_for_inference(self, harness):
        fake, ckpt_dir, make_checkpoint = harness
        make_checkpoint()
        cfg = _cfg()
        cfg["model"]["pretrained"] = True  # the default in the real configs
        lodo._train_fold(cfg=cfg, inference_only=True, **_fold_args())
        assert BUILD_KWARGS[-1]["pretrained"] is False

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

    def test_override_also_removes_a_previous_inference_result(
        self, tmp_path, monkeypatch
    ):
        from src.training.train_asymmetric import _check_checkpoint_collisions

        monkeypatch.chdir(tmp_path)
        run_dir = tmp_path / "checkpoints" / RUN_NAME
        run_dir.mkdir(parents=True)
        for name in ("best.pt", RESULTS_NAME, INFERENCE_RESULTS_NAME):
            (run_dir / name).write_text("x")
        _check_checkpoint_collisions([1], _cfg(), RUN_PREFIX, False, True)
        assert not (run_dir / "best.pt").exists()
        assert not (run_dir / RESULTS_NAME).exists()
        assert not (run_dir / INFERENCE_RESULTS_NAME).exists()


class TestRecordedWandbId:
    def test_inference_uses_the_recorded_id_for_history_and_summary(
        self, harness, monkeypatch
    ):
        fake, ckpt_dir, make_checkpoint = harness
        make_checkpoint()
        (ckpt_dir / "wandb_id.txt").write_text(f"{RUN_NAME}-o2\n")
        fetched: list[str] = []

        def _history(project, entity, run_id):
            fetched.append(run_id)
            return None

        monkeypatch.setattr(lodo, "fetch_wandb_history", _history)
        lodo._train_fold(cfg=_cfg(), inference_only=True, **_fold_args())
        assert fetched == [f"{RUN_NAME}-o2"]
        assert len(fake.init_calls) == 1
        assert fake.init_calls[0]["id"] == f"{RUN_NAME}-o2"
        assert fake.init_calls[0]["name"] == RUN_NAME

    def test_without_the_file_the_run_name_is_the_id(self, harness, monkeypatch):
        fake, ckpt_dir, make_checkpoint = harness
        make_checkpoint()
        fetched: list[str] = []
        monkeypatch.setattr(
            lodo,
            "fetch_wandb_history",
            lambda project, entity, run_id: fetched.append(run_id),
        )
        lodo._train_fold(cfg=_cfg(), inference_only=True, **_fold_args())
        assert fetched == [RUN_NAME]
        assert fake.init_calls[0]["id"] == RUN_NAME


class TestTrainFoldFreshStartWandbId:
    """A fresh start must not reuse a W&B run id that already exists."""

    class _Stop(Exception):
        pass

    def _arrange(self, harness, monkeypatch, existing):
        from types import SimpleNamespace

        from src.training import wandb_identity as wi

        fake, ckpt_dir, make_checkpoint = harness
        monkeypatch.setattr(wi, "wandb_enabled", lambda: True)
        monkeypatch.setattr(
            lodo, "asymmetric_loader", lambda **kw: SimpleNamespace(dataset=[])
        )

        class Api:
            def __init__(self, **kw):
                pass

            def run(self, path):
                if path in existing:
                    return SimpleNamespace(tags=[], update=lambda: None)
                raise RuntimeError("not found")

        fake.Api = Api
        stop = self._Stop

        def _init(**kwargs):
            fake.init_calls.append(kwargs)
            raise stop()

        fake.init = _init
        return fake, make_checkpoint

    def test_fresh_start_with_existing_run_rotates(self, harness, monkeypatch):
        fake, _ = self._arrange(harness, monkeypatch, {f"scratch/{RUN_NAME}"})
        with pytest.raises(self._Stop):
            lodo._train_fold(cfg=_cfg(), **_fold_args())
        assert fake.init_calls[0]["id"] == f"{RUN_NAME}-o2"

    def test_genuine_resume_keeps_current_id(self, harness, monkeypatch):
        fake, make_checkpoint = self._arrange(
            harness, monkeypatch, {f"scratch/{RUN_NAME}"}
        )
        make_checkpoint()
        with pytest.raises(Exception):
            lodo._train_fold(cfg=_cfg(), resume_training=True, **_fold_args())
        assert fake.init_calls[0]["id"] == RUN_NAME
