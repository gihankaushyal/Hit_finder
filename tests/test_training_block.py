"""results.json records how far a training run got (`training` block).

The dashboard's Epochs column reads it. A real fold is driven with scripted validation
F1 scores; no GPU, data or W&B: wandb, the loader, the training step and patch evaluation
are faked.
"""

from __future__ import annotations

import copy
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.evaluation.benchmark import (
    SPLIT_CROSS_DETECTOR,
    SPLIT_IN_DOMAIN_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
)
from src.evaluation.metrics import RECALL_FPR_KEYS
from src.training import lodo
from src.training.inference_results import RESULTS_NAME, SCORES_NAME, training_block

METRICS = {
    "ap": 0.9,
    "auc_roc": 0.95,
    "f1": 0.8,
    "threshold": 0.4,
}
RECALL = {
    k: 0.5 for k in RECALL_FPR_KEYS
}  # only with return_recall=True, as the real one
RUN_PREFIX = "resnet18-asymmetric-v2"
RUN_NAME = f"{RUN_PREFIX}-fold1-seed42"


def _fake_scores() -> dict:
    return {
        "y_true": np.array([1, 0], dtype=np.int8),
        "vote": np.array([0.9, 0.1], dtype=np.float32),
        "max": np.array([0.95, 0.2], dtype=np.float32),
        "session": np.array(["s", "s"]),
    }


def _fake_test_agg(*a, **kw) -> dict:
    """run_patch_agg on a test split: METRICS, plus recall and scores when asked for."""
    return {
        **METRICS,
        **(RECALL if kw.get("return_recall") else {}),
        **({"scores": _fake_scores()} if kw.get("return_scores") else {}),
    }


class TestTrainingBlock:
    def test_records_the_run_as_plain_json_numbers(self):
        block = training_block(
            configured_epochs=200, last_epoch=187, best_epoch=150, early_stopped=True
        )
        assert block == {
            "configured_epochs": 200,
            "last_epoch": 187,
            "best_epoch": 150,
            "early_stopped": True,
        }
        json.dumps(block)

    def test_numpy_and_float_epochs_become_ints(self):
        import numpy as np

        block = training_block(
            configured_epochs=np.int64(10),
            last_epoch=np.int64(7),
            best_epoch=7.0,
            early_stopped=False,
        )
        assert block["last_epoch"] == 7 and type(block["last_epoch"]) is int
        assert block["best_epoch"] == 7 and type(block["best_epoch"]) is int
        assert block["configured_epochs"] == 10

    @pytest.mark.parametrize(
        "bad", [None, float("nan"), float("inf"), "x", True, False, 7.9]
    )
    def test_an_unknown_epoch_is_null_not_an_error(self, bad):
        block = training_block(
            configured_epochs=10, last_epoch=bad, best_epoch=bad, early_stopped=False
        )
        assert block["last_epoch"] is None and block["best_epoch"] is None

    def test_a_best_epoch_after_the_last_epoch_is_unknown(self):
        # the fallback checkpoint (no val-F1 improvement) is stamped with the configured epochs
        block = training_block(
            configured_epochs=200, last_epoch=12, best_epoch=200, early_stopped=True
        )
        assert block["last_epoch"] == 12 and block["best_epoch"] is None


class _FakeRun:
    def __init__(self) -> None:
        self.summary: dict = {}


class _FakeWandb:
    def __init__(self) -> None:
        self.run = _FakeRun()

    def init(self, **kwargs):
        return self.run

    def Settings(self, **kwargs) -> dict:  # noqa: N802 - mirrors wandb.Settings
        return kwargs

    def define_metric(self, *a, **kw) -> None:
        pass

    def log(self, *a, **kw) -> None:
        pass

    def finish(self) -> None:
        pass


class _FakeDataset:
    def __len__(self) -> int:
        return 4

    def set_epoch(self, epoch: int) -> None:
        pass


class _FakeLoader:
    dataset = _FakeDataset()


def _cfg(epochs: int, patience: int = 10) -> dict:
    return {
        "seed": 42,
        "model": {"backbone": "resnet18", "pretrained": False, "num_classes": 2},
        "training": {
            "batch_size": 2,
            "num_workers": 0,
            "epochs": epochs,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "early_stopping_patience": patience,
        },
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
def run_fold(tmp_path, monkeypatch):
    """Returns run(cfg, val_f1s, **kwargs) -> (results dict, number of val evaluations)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setitem(sys.modules, "wandb", _FakeWandb())
    monkeypatch.setattr(lodo, "asymmetric_loader", lambda **kw: _FakeLoader())
    monkeypatch.setattr(
        lodo, "train_one_epoch", lambda *a, **kw: {"loss": 0.5, "hit_frac": 0.5}
    )
    monkeypatch.setattr(lodo, "build_supervised_model", lambda **kw: nn.Linear(4, 2))
    monkeypatch.setattr(lodo, "wandb_id_for_training", lambda *a, **kw: "fake-id")

    def run(cfg, val_f1s, **kwargs):
        scores = iter(val_f1s)
        evaluations: list[int] = []

        def fake_agg(model, session_map, ids, **kw):
            if ids == ["s_val"]:
                evaluations.append(1)
                recall = RECALL if kw.get("return_recall") else {}
                return {**METRICS, **recall, "f1": next(scores), "threshold": 0.5}
            return _fake_test_agg(**kw)

        monkeypatch.setattr(lodo, "run_patch_agg", fake_agg)
        lodo._train_fold(cfg=cfg, **{**_fold_args(), **kwargs})
        path = Path("checkpoints") / RUN_NAME / RESULTS_NAME
        return json.loads(path.read_text()), len(evaluations)

    return run


class TestTrainFoldWritesTheTrainingBlock:
    def test_results_carry_recall_at_fixed_fpr(self, run_fold):
        data, _ = run_fold(_cfg(epochs=1), [0.5])
        for split in ("cross", "in_domain"):
            for key in RECALL_FPR_KEYS:
                assert data[split][key] == 0.5

    def test_the_test_evaluation_saves_per_frame_scores(self, run_fold, tmp_path):
        run_fold(_cfg(epochs=1), [0.5])
        with np.load(tmp_path / "checkpoints" / RUN_NAME / SCORES_NAME) as z:
            assert z["cross_y_true"].tolist() == [1, 0]
            assert z["in_domain_vote"].dtype == np.float32
            assert set(z.files) == {
                f"{s}_{k}"
                for s in ("cross", "in_domain")
                for k in ("y_true", "vote", "max", "session")
            }

    def test_recall_at_fixed_fpr_is_logged_for_the_test_not_per_epoch(
        self, run_fold, monkeypatch
    ):
        logged: list[dict] = []
        monkeypatch.setattr(
            sys.modules["wandb"], "log", lambda payload, **kw: logged.append(payload)
        )
        run_fold(_cfg(epochs=2), [0.5, 0.6])
        keys = {k for payload in logged for k in payload}
        assert "cross/recall_at_fpr_1e-3" in keys
        assert "in_domain/max_recall_at_fpr_1e-2" in keys
        assert not any(k.startswith("val/") and "fpr" in k for k in keys)

    def test_a_run_that_trains_every_epoch(self, run_fold):
        data, n = run_fold(_cfg(epochs=4), [0.5, 0.6, 0.7, 0.8])
        assert n == 4
        assert data["training"] == {
            "configured_epochs": 4,
            "last_epoch": 4,
            "best_epoch": 4,
            "early_stopped": False,
        }

    def test_the_best_epoch_is_not_the_last_epoch(self, run_fold):
        data, _ = run_fold(_cfg(epochs=4, patience=10), [0.5, 0.9, 0.6, 0.55])
        assert data["training"] == {
            "configured_epochs": 4,
            "last_epoch": 4,
            "best_epoch": 2,
            "early_stopped": False,
        }

    def test_early_stopping_ends_the_run_early_and_says_so(self, run_fold):
        # best at epoch 2; patience 2 -> epochs 3 and 4 do not improve -> stop after epoch 4
        data, n = run_fold(_cfg(epochs=10, patience=2), [0.5, 0.9, 0.6, 0.55, 0.99])
        assert n == 4
        assert data["training"] == {
            "configured_epochs": 10,
            "last_epoch": 4,
            "best_epoch": 2,
            "early_stopped": True,
        }

    def test_resuming_a_finished_run_keeps_where_it_got_to(self, run_fold, tmp_path):
        ckpt_dir = tmp_path / "checkpoints" / RUN_NAME
        ckpt_dir.mkdir(parents=True)
        torch.save(
            {
                "epoch": 5,
                "model_state_dict": nn.Linear(4, 2).state_dict(),
                "optimizer_state_dict": torch.optim.AdamW(
                    nn.Linear(4, 2).parameters()
                ).state_dict(),
                "val_f1": 0.7,
                "epochs_no_improve": 0,
                "inference_threshold": 0.5,
                "backbone": "resnet18",
                "num_classes": 2,
            },
            ckpt_dir / "best.pt",
        )
        data, n = run_fold(_cfg(epochs=5), [], resume_training=True)
        assert n == 0  # nothing left to train
        assert data["training"] == {
            "configured_epochs": 5,
            "last_epoch": 5,
            "best_epoch": 5,
            "early_stopped": False,
        }

    def test_no_validation_improvement_leaves_the_best_epoch_unknown(self, run_fold):
        # every val F1 is NaN: best.pt is the fallback checkpoint stamped with epoch 10
        nan = float("nan")
        data, n = run_fold(_cfg(epochs=10, patience=3), [nan] * 4)
        assert n == 3
        assert data["training"] == {
            "configured_epochs": 10,
            "last_epoch": 3,
            "best_epoch": None,
            "early_stopped": True,
        }

    def test_resuming_with_epochs_left_describes_the_session_that_finished(
        self, run_fold, tmp_path
    ):
        ckpt_dir = tmp_path / "checkpoints" / RUN_NAME
        ckpt_dir.mkdir(parents=True)
        torch.save(
            {
                "epoch": 2,
                "model_state_dict": nn.Linear(4, 2).state_dict(),
                "optimizer_state_dict": torch.optim.AdamW(
                    nn.Linear(4, 2).parameters()
                ).state_dict(),
                "val_f1": 0.7,
                "epochs_no_improve": 0,
                "inference_threshold": 0.5,
                "backbone": "resnet18",
                "num_classes": 2,
            },
            ckpt_dir / "best.pt",
        )
        data, n = run_fold(_cfg(epochs=4), [0.9, 0.95], resume_training=True)
        assert n == 2  # epochs 3 and 4
        assert data["training"] == {
            "configured_epochs": 4,
            "last_epoch": 4,
            "best_epoch": 4,
            "early_stopped": False,
        }

    def test_extra_results_may_not_set_the_reserved_training_key(self, run_fold):
        with pytest.raises(ValueError, match="training"):
            run_fold(
                _cfg(epochs=2),
                [0.5, 0.6],
                extra_results={"training": {"anything": 1}},
            )

    def test_an_inference_only_pass_does_not_claim_to_have_trained(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("WANDB_MODE", raising=False)
        monkeypatch.setitem(sys.modules, "wandb", _FakeWandb())
        monkeypatch.setattr(
            lodo, "build_supervised_model", lambda **kw: nn.Linear(4, 2)
        )
        monkeypatch.setattr(lodo, "run_patch_agg", _fake_test_agg)
        ckpt_dir = tmp_path / "checkpoints" / RUN_NAME
        ckpt_dir.mkdir(parents=True)
        torch.save(
            {
                "epoch": 7,
                "model_state_dict": nn.Linear(4, 2).state_dict(),
                "val_f1": 0.61,
                "inference_threshold": 0.4,
                "backbone": "resnet18",
                "num_classes": 2,
            },
            ckpt_dir / "best.pt",
        )
        lodo._train_fold(cfg=_cfg(epochs=10), inference_only=True, **_fold_args())
        data = json.loads((ckpt_dir / RESULTS_NAME).read_text())
        assert "training" not in data
        assert data["inference"]["checkpoint_epoch"] == 7


class TestWandbIdFollowsTheCheckpointNotTheFlag:
    """--resume-training with no checkpoint trains from scratch, so it needs a fresh W&B id."""

    def _record(self, monkeypatch) -> list[int | None]:
        calls: list[int | None] = []

        def fake(*args, restored_epoch, **kwargs):
            calls.append(restored_epoch)
            return "fake-id"

        monkeypatch.setattr(lodo, "wandb_id_for_training", fake)
        return calls

    def test_flag_without_a_checkpoint_is_a_fresh_start(self, run_fold, monkeypatch):
        calls = self._record(monkeypatch)
        run_fold(_cfg(epochs=1), [0.5], resume_training=True)
        assert calls == [None]

    def test_flag_with_a_checkpoint_passes_the_restored_epoch(
        self, run_fold, monkeypatch
    ):
        calls = self._record(monkeypatch)
        run_fold(_cfg(epochs=1), [0.5])
        run_fold(_cfg(epochs=2), [0.6], resume_training=True)
        assert calls == [None, 1]


def _ckpt_dir(tmp_path: Path) -> Path:
    return tmp_path / "checkpoints" / RUN_NAME


class TestResumeFromLastEpoch:
    """last.pt is written every epoch, so a resume continues where the run stopped."""

    def test_last_pt_records_the_newest_epoch_and_the_counters(
        self, run_fold, tmp_path
    ):
        run_fold(_cfg(epochs=3), [0.5, 0.9, 0.6])
        last = torch.load(_ckpt_dir(tmp_path) / "last.pt", weights_only=False)
        assert last["epoch"] == 3
        assert last["best_f1"] == 0.9
        assert last["epochs_no_improve"] == 1
        assert last["early_stopped"] is False
        assert "optimizer_state_dict" in last
        assert (
            torch.load(_ckpt_dir(tmp_path) / "best.pt", weights_only=False)["epoch"]
            == 2
        )

    def test_resume_continues_after_the_last_epoch_not_the_best(self, run_fold):
        run_fold(_cfg(epochs=3), [0.5, 0.9, 0.6])
        # best was epoch 2; a best.pt resume would retrain epoch 3 and need 3 scores
        data, n = run_fold(_cfg(epochs=5), [0.95, 0.7], resume_training=True)
        assert n == 2
        assert data["training"]["last_epoch"] == 5
        assert data["training"]["best_epoch"] == 4

    def test_resume_keeps_the_best_f1_so_far(self, run_fold, tmp_path):
        run_fold(_cfg(epochs=3), [0.5, 0.9, 0.6])
        run_fold(_cfg(epochs=4), [0.8], resume_training=True)  # below 0.9
        best = torch.load(_ckpt_dir(tmp_path) / "best.pt", weights_only=False)
        assert best["epoch"] == 2

    def test_an_early_stopped_run_does_not_train_again(self, run_fold):
        run_fold(_cfg(epochs=10, patience=2), [0.5, 0.9, 0.6, 0.55])
        data, n = run_fold(_cfg(epochs=10, patience=2), [], resume_training=True)
        assert n == 0
        assert data["training"]["last_epoch"] == 4
        assert data["training"]["early_stopped"] is True

    def test_the_checkpoint_is_written_before_the_epoch_is_logged(
        self, run_fold, tmp_path, monkeypatch
    ):
        seen: list[tuple[int, int]] = []

        def _log(payload, step=None, **kw):
            if "train/loss" in payload:
                last = torch.load(_ckpt_dir(tmp_path) / "last.pt", weights_only=False)
                seen.append((step, last["epoch"]))

        monkeypatch.setattr(sys.modules["wandb"], "log", _log)
        run_fold(_cfg(epochs=3), [0.5, 0.9, 0.6])
        assert seen == [(1, 1), (2, 2), (3, 3)]

    def test_a_legacy_best_pt_only_dir_resumes_after_its_epoch(
        self, run_fold, tmp_path
    ):
        run_fold(_cfg(epochs=3), [0.5, 0.9, 0.6])
        (_ckpt_dir(tmp_path) / "last.pt").unlink()  # a run dir from before last.pt
        # best.pt is epoch 2 (F1 0.9): epochs 3 and 4 train, and 0.85 must not beat it
        data, n = run_fold(_cfg(epochs=4), [0.85, 0.7], resume_training=True)
        assert n == 2
        assert data["training"]["last_epoch"] == 4
        best = torch.load(_ckpt_dir(tmp_path) / "best.pt", weights_only=False)
        assert best["epoch"] == 2

    def test_resume_keeps_the_patience_counter(self, run_fold):
        # patience 2; epoch 3 did not improve, so one more non-improving epoch stops it
        run_fold(_cfg(epochs=3, patience=2), [0.5, 0.9, 0.6])
        data, n = run_fold(
            _cfg(epochs=10, patience=2), [0.7, 0.99], resume_training=True
        )
        assert n == 1
        assert data["training"]["early_stopped"] is True

    def test_a_larger_patience_lets_an_early_stopped_run_continue(self, run_fold):
        run_fold(_cfg(epochs=10, patience=2), [0.5, 0.9, 0.6, 0.55])  # stops at 4
        data, n = run_fold(
            _cfg(epochs=10, patience=4), [0.5, 0.5, 0.5], resume_training=True
        )
        assert n == 2  # epochs 5 and 6 bring the count from 2 to 4
        assert data["training"]["last_epoch"] == 6
        assert data["training"]["early_stopped"] is True

    def test_resume_loads_last_pt_weights_and_optimizer_state(
        self, run_fold, tmp_path, monkeypatch
    ):
        captured: list[tuple[dict, dict]] = []

        def _step(model, loader, optimizer, criterion, device):
            if resumed:
                captured.append(
                    (
                        {k: v.clone() for k, v in model.state_dict().items()},
                        # clone: step() below updates the live state tensors in place
                        copy.deepcopy(optimizer.state_dict()["state"]),
                    )
                )
            optimizer.zero_grad()
            model(torch.ones(1, 4)).sum().backward()
            optimizer.step()
            return {"loss": 0.5, "hit_frac": 0.5}

        resumed = False
        monkeypatch.setattr(lodo, "train_one_epoch", _step)
        run_fold(_cfg(epochs=2), [0.9, 0.5])  # best.pt = epoch 1, last.pt = epoch 2
        last = torch.load(_ckpt_dir(tmp_path) / "last.pt", weights_only=False)
        best = torch.load(_ckpt_dir(tmp_path) / "best.pt", weights_only=False)
        assert not torch.equal(
            last["model_state_dict"]["bias"], best["model_state_dict"]["bias"]
        )

        resumed = True
        run_fold(_cfg(epochs=3), [0.4], resume_training=True)
        weights, opt_state = captured[0]
        for k, v in last["model_state_dict"].items():
            assert torch.equal(weights[k], v)
        saved = last["optimizer_state_dict"]["state"]
        assert opt_state.keys() == saved.keys()
        for i in saved:
            assert torch.equal(opt_state[i]["exp_avg"], saved[i]["exp_avg"])

    def test_a_kill_between_best_and_last_resumes_from_best_pt(
        self, run_fold, tmp_path, monkeypatch
    ):
        real_save = lodo.atomic_torch_save

        def _killed_at_epoch_2(obj, path):
            if Path(path).name == "last.pt" and obj["epoch"] == 2:
                raise KeyboardInterrupt  # preempted after best.pt (epoch 2) was saved
            real_save(obj, path)

        monkeypatch.setattr(lodo, "atomic_torch_save", _killed_at_epoch_2)
        with pytest.raises(KeyboardInterrupt):
            run_fold(_cfg(epochs=4), [0.5, 0.9])
        monkeypatch.setattr(lodo, "atomic_torch_save", real_save)
        # last.pt is epoch 1 (best F1 0.5); best.pt is epoch 2 (F1 0.9): resume from
        # best.pt, so epoch 2 is not trained again and only epochs 3 and 4 run.
        # A third score, so a resume that retrains epoch 2 fails on n, not the fake.
        data, n = run_fold(_cfg(epochs=4), [0.7, 0.8, 0.85], resume_training=True)
        assert n == 2
        assert data["training"]["best_epoch"] == 2
        best = torch.load(_ckpt_dir(tmp_path) / "best.pt", weights_only=False)
        assert best["epoch"] == 2
        assert best["val_f1"] == 0.9

    def _kill_after_best_pt_at_epoch_2(self, run_fold, monkeypatch, cfg):
        real_save = lodo.atomic_torch_save

        def _killed(obj, path):
            if Path(path).name == "last.pt" and obj["epoch"] == 2:
                raise KeyboardInterrupt
            real_save(obj, path)

        monkeypatch.setattr(lodo, "atomic_torch_save", _killed)
        with pytest.raises(KeyboardInterrupt):
            run_fold(cfg, [0.5, 0.9])
        monkeypatch.setattr(lodo, "atomic_torch_save", real_save)

    def test_a_best_pt_restore_loads_its_weights_and_optimizer_state(
        self, run_fold, tmp_path, monkeypatch
    ):
        captured: list[tuple[dict, dict]] = []

        def _step(model, loader, optimizer, criterion, device):
            captured.append(
                (
                    {k: v.clone() for k, v in model.state_dict().items()},
                    copy.deepcopy(optimizer.state_dict()["state"]),
                )
            )
            optimizer.zero_grad()
            model(torch.ones(1, 4)).sum().backward()
            optimizer.step()
            return {"loss": 0.5, "hit_frac": 0.5}

        monkeypatch.setattr(lodo, "train_one_epoch", _step)
        self._kill_after_best_pt_at_epoch_2(run_fold, monkeypatch, _cfg(epochs=4))
        best = torch.load(_ckpt_dir(tmp_path) / "best.pt", weights_only=False)
        captured.clear()
        _, n = run_fold(_cfg(epochs=4), [0.7, 0.8], resume_training=True)
        assert n == 2
        weights, opt_state = captured[0]
        for k, v in best["model_state_dict"].items():
            assert torch.equal(weights[k], v)
        saved = best["optimizer_state_dict"]["state"]
        assert saved and opt_state.keys() == saved.keys()
        for i in saved:
            assert torch.equal(opt_state[i]["exp_avg"], saved[i]["exp_avg"])

    def test_a_best_pt_restore_logs_its_epoch_to_wandb(
        self, run_fold, tmp_path, monkeypatch
    ):
        """Epoch 2 was killed before its W&B log: its val F1 is logged on resume."""
        self._kill_after_best_pt_at_epoch_2(run_fold, monkeypatch, _cfg(epochs=4))
        logs: list[tuple[int | None, dict]] = []
        monkeypatch.setattr(
            sys.modules["wandb"],
            "log",
            lambda payload, step=None, **kw: logs.append((step, payload)),
        )
        run_fold(_cfg(epochs=4), [0.7, 0.8], resume_training=True)
        assert logs[0] == (2, {"epoch": 2, "val/f1": 0.9})
        assert [s for s, p in logs if "train/loss" in p] == [3, 4]

    def test_a_kill_before_the_epoch_snapshot_still_writes_it(
        self, run_fold, tmp_path, monkeypatch
    ):
        real_save = lodo.atomic_torch_save

        def _killed_at_snapshot(obj, path):
            if Path(path).name == "epoch2.pt":
                raise KeyboardInterrupt
            real_save(obj, path)

        cfg = _cfg(epochs=4)
        cfg["training"]["save_epochs"] = [2]
        monkeypatch.setattr(lodo, "atomic_torch_save", _killed_at_snapshot)
        with pytest.raises(KeyboardInterrupt):
            run_fold(cfg, [0.5, 0.9])  # epoch 2 improves and is a snapshot epoch
        monkeypatch.setattr(lodo, "atomic_torch_save", real_save)
        run_fold(cfg, [0.9, 0.7, 0.8], resume_training=True)
        assert (_ckpt_dir(tmp_path) / "epoch2.pt").exists()

    def test_a_resume_ignores_the_nan_fallback_best_pt(self, run_fold):
        """The fallback best.pt is stamped with the configured epochs, not trained ones."""
        nan = float("nan")
        run_fold(
            _cfg(epochs=10, patience=3), [nan] * 3
        )  # stops at 3; best.pt "epoch 10"
        data, n = run_fold(_cfg(epochs=10, patience=5), [nan] * 3, resume_training=True)
        assert n == 2  # epochs 4 and 5 bring the count from 3 to 5
        assert data["training"]["last_epoch"] == 5
        assert data["training"]["best_epoch"] is None

    def test_a_corrupt_last_pt_is_refused_even_with_a_newer_best_pt(
        self, run_fold, tmp_path, monkeypatch
    ):
        self._kill_after_best_pt_at_epoch_2(run_fold, monkeypatch, _cfg(epochs=4))
        path = _ckpt_dir(tmp_path) / "last.pt"
        last = torch.load(path, weights_only=False)
        del last["best_f1"]
        torch.save(last, path)
        with pytest.raises(RuntimeError, match="best_f1"):
            run_fold(_cfg(epochs=4), [0.7, 0.8], resume_training=True)

    def test_the_epoch_snapshot_is_saved_before_last_pt(
        self, run_fold, tmp_path, monkeypatch
    ):
        real_save = lodo.atomic_torch_save

        def _killed_before_last_pt(obj, path):
            if Path(path).name == "last.pt" and obj["epoch"] == 2:
                raise KeyboardInterrupt
            real_save(obj, path)

        monkeypatch.setattr(lodo, "atomic_torch_save", _killed_before_last_pt)
        cfg = _cfg(epochs=3)
        cfg["training"]["save_epochs"] = [2]
        with pytest.raises(KeyboardInterrupt):
            run_fold(cfg, [0.5, 0.4])
        assert (_ckpt_dir(tmp_path) / "epoch2.pt").exists()

    def test_a_resume_logs_its_first_epoch(self, run_fold, monkeypatch):
        """W&B drops a step below its counter: no unstepped log may precede an epoch."""
        run_fold(_cfg(epochs=3), [0.5, 0.9, 0.6])
        steps: list[int | None] = []
        monkeypatch.setattr(
            sys.modules["wandb"],
            "log",
            lambda payload, step=None, **kw: steps.append(step),
        )
        run_fold(_cfg(epochs=4), [0.7], resume_training=True)
        assert steps[0] == 4

    def _record_loads(self, monkeypatch) -> list[tuple[str, dict]]:
        real_load = lodo.torch.load
        loads: list[tuple[str, dict]] = []

        def _load(path, *args, **kwargs):
            loads.append((Path(path).name, kwargs))
            return real_load(path, *args, **kwargs)

        monkeypatch.setattr(lodo.torch, "load", _load)
        return loads

    def test_checkpoints_are_loaded_on_the_cpu_for_evaluation_too(
        self, run_fold, monkeypatch
    ):
        loads = self._record_loads(monkeypatch)
        # a torch.device, so a load mapped to the training device differs from "cpu"
        run_fold(_cfg(epochs=2), [0.5, 0.6], device=torch.device("cpu"))
        run_fold(
            _cfg(epochs=3), [0.7], resume_training=True, device=torch.device("cpu")
        )
        assert loads and all(kw["map_location"] == "cpu" for _, kw in loads)

    def test_an_older_best_pt_is_only_peeked_at_on_resume(self, run_fold, monkeypatch):
        """last.pt wins: best.pt's weights and optimizer state are not read in."""
        run_fold(_cfg(epochs=3), [0.5, 0.9, 0.6])  # best.pt epoch 2, last.pt epoch 3
        loads = self._record_loads(monkeypatch)
        run_fold(_cfg(epochs=4), [0.7], resume_training=True)
        restore = loads[: [n for n, _ in loads].index("last.pt") + 2]
        best = [kw for n, kw in restore if n == "best.pt"]
        assert best and all(kw.get("mmap") for kw in best)

    @pytest.mark.parametrize(
        "key", ["epoch", "best_f1", "epochs_no_improve", "optimizer_state_dict"]
    )
    def test_a_last_pt_missing_a_key_is_refused(self, run_fold, tmp_path, key):
        run_fold(_cfg(epochs=2), [0.5, 0.6])
        path = _ckpt_dir(tmp_path) / "last.pt"
        last = torch.load(path, weights_only=False)
        del last[key]
        torch.save(last, path)
        with pytest.raises(RuntimeError, match=key):
            run_fold(_cfg(epochs=3), [0.7], resume_training=True)


def test_asymmetric_override_also_deletes_last_pt(tmp_path, monkeypatch):
    from src.training import train_asymmetric

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    run_dir = _ckpt_dir(tmp_path)
    run_dir.mkdir(parents=True)
    scores = ("scores.npz", "scores.inference.npz")
    for name in ("best.pt", "last.pt", "last.pt.tmp", RESULTS_NAME, *scores):
        (run_dir / name).write_bytes(b"x")
    train_asymmetric._check_checkpoint_collisions(
        [1],
        {"seed": 42, "wandb": {"project": "p"}},
        RUN_PREFIX,
        resume_training=False,
        override_training=True,
    )
    assert not (run_dir / "last.pt").exists()
    assert not (run_dir / "last.pt.tmp").exists()  # left by a SLURM kill mid-save
    assert not (run_dir / "best.pt").exists()
    assert not any((run_dir / name).exists() for name in scores)
