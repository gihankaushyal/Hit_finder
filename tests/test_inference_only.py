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
