"""A requeued SSL launch driven through the real ``main()`` entry points.

The gate's "continued" result only works if ``main`` passes ``continued`` to the real
gate call and turns it into a resume. These tests drive ``main()`` twice under one
``SLURM_JOB_ID`` (the second time as a requeue) with the heavy parts faked, so a
``main`` that drops the signal fails here instead of on the cluster.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PT_NAME = "mae-vits16-v2-fold1-seed42"
FT_NAME = "vits16-mae-finetune-v2-fold1-seed42"


@pytest.fixture(autouse=True)
def _slurm_env(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    monkeypatch.setattr(
        "src.training.wandb_identity.tag_overridden",
        lambda project, entity, run_id: True,
    )


def _fold():
    return {"fold_id": 1, "test_detector": "AGIPD"}


def _pretrain_main(tmp_path, monkeypatch, seen):
    from src.training import train_ssl_pretrain as tp

    cfg = {
        "seed": 42,
        "wandb": {"project": "p"},
        "checkpoint_dir": str(tmp_path / "ckpt"),
        "training": {"epochs": 1},
        "lodo": {"detector_dirs": {}},
    }
    monkeypatch.setattr(tp, "load_config", lambda path: cfg)
    monkeypatch.setattr(tp, "frame_cache_from_cfg", lambda c: None)
    monkeypatch.setattr(tp, "build_sessions", lambda lodo: ([], {}))
    monkeypatch.setattr("src.evaluation.benchmark.build_lodo_folds", lambda: [_fold()])

    def fake_run_pretrain(cfg, session_map, ids, run_name, device, resume, **kw):
        seen.append(resume)
        (Path(cfg["checkpoint_dir"]) / run_name / "last.pt").write_bytes(b"new")
        return {}

    monkeypatch.setattr(tp, "run_pretrain", fake_run_pretrain)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--config",
            "c.yaml",
            "--fold",
            "1",
            "--run-name-prefix",
            "mae-vits16-v2",
            "--override-training",
            "--device",
            "cpu",
        ],
    )
    return tp, Path(cfg["checkpoint_dir"]) / PT_NAME / "last.pt"


def _finetune_main(tmp_path, monkeypatch, seen):
    from src.training import train_ssl_finetune as tf

    cfg = {"seed": 42, "wandb": {"project": "p"}}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tf, "load_config", lambda path: cfg)
    monkeypatch.setattr(tf, "read_pretrain_epoch", lambda path: 7)
    monkeypatch.setattr(tf, "frame_cache_from_cfg", lambda c: None)
    monkeypatch.setattr(tf, "get_hitfinder", lambda c: None)
    monkeypatch.setattr(tf, "build_sessions", lambda lodo: ([], {}))
    monkeypatch.setattr(tf, "build_lodo_folds", lambda: [_fold()])
    monkeypatch.setattr(tf, "build_session_stratified_split", lambda *a, **k: {})
    monkeypatch.setattr(tf, "save_split_artifact", lambda *a, **k: None)
    monkeypatch.setattr(tf, "SPLIT_DIR", tmp_path / "splits")
    monkeypatch.setattr(tf, "build_finetune_model_builder", lambda *a, **k: None)
    cfg["lodo"] = {}

    def fake_train_fold(fold, *a, resume_training, **k):
        seen.append(resume_training)
        (tmp_path / "checkpoints" / FT_NAME / "best.pt").write_bytes(b"new")
        return {}

    monkeypatch.setattr(tf, "_train_fold", fake_train_fold)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--config",
            "c.yaml",
            "--fold",
            "1",
            "--run-name-prefix",
            "vits16-mae-v2",
            "--pretrain-checkpoint",
            "p.pt",
            "--override-training",
            "--device",
            "cpu",
        ],
    )
    return tf, tmp_path / "checkpoints" / FT_NAME / "best.pt"


@pytest.mark.parametrize("which", ["pretrain", "finetune"])
def test_requeue_keeps_the_new_checkpoint_and_resumes(tmp_path, monkeypatch, which):
    seen: list[bool] = []
    build = _pretrain_main if which == "pretrain" else _finetune_main
    module, ckpt = build(tmp_path, monkeypatch, seen)
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"old attempt")

    module.main()  # first start: the override applies
    assert seen == [False]
    assert ckpt.read_bytes() == b"new"  # the fake wrote the new attempt's checkpoint

    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")  # SLURM requeues the job
    module.main()
    assert seen == [False, True]
    assert ckpt.read_bytes() == b"new"


@pytest.mark.parametrize("which", ["pretrain", "finetune"])
def test_requeue_before_the_first_save_resumes_without_a_checkpoint(
    tmp_path, monkeypatch, which
):
    seen: list[bool] = []
    build = _pretrain_main if which == "pretrain" else _finetune_main
    module, ckpt = build(tmp_path, monkeypatch, seen)
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"old attempt")

    module.main()
    ckpt.unlink()  # preempted before the new attempt saved anything
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    module.main()
    assert seen == [False, True]


@pytest.mark.parametrize("which", ["pretrain", "finetune"])
def test_a_new_sbatch_overrides_again(tmp_path, monkeypatch, which):
    seen: list[bool] = []
    build = _pretrain_main if which == "pretrain" else _finetune_main
    module, ckpt = build(tmp_path, monkeypatch, seen)
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"old attempt")

    module.main()
    monkeypatch.setenv("SLURM_JOB_ID", "456")  # a deliberate new submission
    module.main()
    assert seen == [False, False]
