"""results.json records how far a training run got (`training` block).

The dashboard's Epochs column reads it. A real fold is driven with scripted validation
F1 scores; no GPU, data or W&B: wandb, the loader, the training step and patch evaluation
are faked.
"""

from __future__ import annotations

import json
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
from src.training.inference_results import RESULTS_NAME, training_block

METRICS = {"ap": 0.9, "auc_roc": 0.95, "f1": 0.8, "threshold": 0.4}
RUN_PREFIX = "resnet18-asymmetric-v2"
RUN_NAME = f"{RUN_PREFIX}-fold1-seed42"


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

    @pytest.mark.parametrize("bad", [None, float("nan"), "x"])
    def test_an_unknown_epoch_is_null_not_an_error(self, bad):
        block = training_block(
            configured_epochs=10, last_epoch=bad, best_epoch=bad, early_stopped=False
        )
        assert block["last_epoch"] is None and block["best_epoch"] is None


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
                return {**METRICS, "f1": next(scores), "threshold": 0.5}
            return dict(METRICS)

        monkeypatch.setattr(lodo, "run_patch_agg", fake_agg)
        lodo._train_fold(cfg=cfg, **_fold_args(), **kwargs)
        path = Path("checkpoints") / RUN_NAME / RESULTS_NAME
        return json.loads(path.read_text()), len(evaluations)

    return run


class TestTrainFoldWritesTheTrainingBlock:
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

    def test_an_inference_only_pass_does_not_claim_to_have_trained(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("WANDB_MODE", raising=False)
        monkeypatch.setitem(sys.modules, "wandb", _FakeWandb())
        monkeypatch.setattr(
            lodo, "build_supervised_model", lambda **kw: nn.Linear(4, 2)
        )
        monkeypatch.setattr(lodo, "run_patch_agg", lambda *a, **kw: dict(METRICS))
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
