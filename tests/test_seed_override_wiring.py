"""``--seed`` reaches the run name and the collision gate in both training entry points.

The override must be applied before the checkpoint-collision gate and ``fold_run_name`` read
``cfg["seed"]``; otherwise a seed-43 launch would check (or, with ``--override-training``,
delete) the seed-42 checkpoint. A finished seed-42 run is planted, so a misplaced override
fails the gate here.
"""

from __future__ import annotations

import sys

import pytest

FOLD = {"fold_id": 1, "test_detector": "AGIPD"}


def _plant_finished_run(tmp_path, run_name):
    d = tmp_path / "checkpoints" / run_name
    d.mkdir(parents=True)
    (d / "best.pt").write_bytes(b"seed 42")


def _finetune(tmp_path, monkeypatch, extra_argv):
    from src.training import train_ssl_finetune as tf

    cfg = {"seed": 42, "wandb": {"project": "p"}, "lodo": {}}
    seen = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tf, "load_config", lambda path: cfg)
    monkeypatch.setattr(tf, "read_pretrain_epoch", lambda path: 7)
    monkeypatch.setattr(tf, "frame_cache_from_cfg", lambda c: None)
    monkeypatch.setattr(tf, "get_hitfinder", lambda c: None)
    monkeypatch.setattr(tf, "build_sessions", lambda lodo: ([], {}))
    monkeypatch.setattr(tf, "build_lodo_folds", lambda: [FOLD])
    monkeypatch.setattr(tf, "build_session_stratified_split", lambda *a, **k: {})
    monkeypatch.setattr(tf, "save_split_artifact", lambda *a, **k: None)
    monkeypatch.setattr(tf, "SPLIT_DIR", tmp_path / "splits")
    monkeypatch.setattr(tf, "build_finetune_model_builder", lambda *a, **k: None)
    monkeypatch.setattr(
        tf, "_train_fold", lambda fold, split, smap, c, *a, **k: seen.append(c["seed"])
    )
    argv = ["x", "--config", "c.yaml", "--fold", "1"]
    argv += ["--run-name-prefix", "vits16-mae-v2", "--pretrain-checkpoint", "p.pt"]
    argv += ["--device", "cpu", *extra_argv]
    monkeypatch.setattr(sys, "argv", argv)
    return tf, seen


def test_finetune_seed_flag_names_the_run_and_clears_the_gate(tmp_path, monkeypatch):
    _plant_finished_run(tmp_path, "vits16-mae-finetune-v2-fold1-seed42")
    tf, seen = _finetune(tmp_path, monkeypatch, ["--seed", "43"])
    tf.main()
    assert seen == [43]


def test_finetune_without_the_flag_still_hits_the_seed_42_gate(tmp_path, monkeypatch):
    _plant_finished_run(tmp_path, "vits16-mae-finetune-v2-fold1-seed42")
    tf, seen = _finetune(tmp_path, monkeypatch, [])
    with pytest.raises(SystemExit, match="seed42"):  # collision gate
        tf.main()
    assert seen == []


def _asymmetric(tmp_path, monkeypatch):
    from src.training import train_asymmetric as ta

    cfg = {
        "seed": 42,
        "wandb": {"project": "p"},
        "training": {"num_workers": 0},
        "hitfinder": {"backend": "pf8"},
        "lodo": {"detector_dirs": {"AGIPD": "x"}},
    }
    seen = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ta, "load_config", lambda path: cfg)
    monkeypatch.setattr(ta, "check_crop_options", lambda c: None)
    monkeypatch.setattr(ta, "get_hitfinder", lambda c: None)
    monkeypatch.setattr(ta, "frame_cache_from_cfg", lambda c: None)
    monkeypatch.setattr(
        ta,
        "build_sessions",
        lambda lodo: ([{"detector": "AGIPD", "frame_count": 1}], {}),
    )
    monkeypatch.setattr(ta, "build_lodo_folds", lambda: [FOLD])
    monkeypatch.setattr(ta, "build_session_stratified_split", lambda *a, **k: {})
    monkeypatch.setattr(ta, "save_split_artifact", lambda *a, **k: None)
    monkeypatch.setattr(
        ta,
        "_train_fold",
        lambda fold, split, smap, c, *a, **k: seen.append(c["seed"])
        or {"ap": 0.5, "test_detector": "AGIPD"},
    )
    return ta, seen


def test_asymmetric_seed_argument_names_the_run_and_clears_the_gate(
    tmp_path, monkeypatch
):
    _plant_finished_run(tmp_path, "resnet18-asymmetric-v2-fold1-seed42")
    ta, seen = _asymmetric(tmp_path, monkeypatch)
    ta.main("c.yaml", "resnet18-asymmetric-v2", folds=[1], device="cpu", seed=43)
    assert seen == [43]


def test_asymmetric_without_a_seed_still_hits_the_seed_42_gate(tmp_path, monkeypatch):
    _plant_finished_run(tmp_path, "resnet18-asymmetric-v2-fold1-seed42")
    ta, seen = _asymmetric(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="seed42"):
        ta.main("c.yaml", "resnet18-asymmetric-v2", folds=[1], device="cpu")
    assert seen == []


# --- the seed must reach the data pipeline, not only the run name ---------------------


class _Stop(Exception):
    pass


def _loader_kwargs_for_seed(tmp_path, monkeypatch, seed):
    """Run lodo._train_fold up to the training loader and return the kwargs it passed."""
    from src.training import lodo

    captured = {}

    def fake_loader(**kwargs):
        captured.update(kwargs)
        raise _Stop

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(lodo, "asymmetric_loader", fake_loader)
    cfg = {
        "seed": seed,
        "model": {"backbone": "resnet18"},
        "training": {"batch_size": 2, "num_workers": 0, "epochs": 1},
        "lodo": {},
        "wandb": {},
    }
    split = {"splits": {"s1": "train"}}
    with pytest.raises(_Stop):
        lodo._train_fold(FOLD, split, {"s1": "x"}, cfg, None, "cpu")
    return captured


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_train_fold_hands_the_run_seed_to_the_training_loader(
    tmp_path, monkeypatch, seed
):
    assert _loader_kwargs_for_seed(tmp_path, monkeypatch, seed)["seed"] == seed


def test_asymmetric_loader_forwards_the_seed_to_the_dataset(monkeypatch):
    from src.data import dataloader

    seen = {}

    class FakeDataset(list):
        def __init__(self, **kwargs):
            super().__init__([0])
            seen.update(kwargs)

    monkeypatch.setattr(dataloader, "AsymmetricCXIDataset", FakeDataset)
    dataloader.asymmetric_loader({}, [], None, batch_size=1, num_workers=0, seed=44)
    assert seen["seed"] == 44
    dataloader.asymmetric_loader({}, [], None, batch_size=1, num_workers=0)
    assert (
        seen["seed"] == 42
    )  # the default keeps every existing seed-42 run reproducible


def test_different_seeds_draw_different_per_sample_streams():
    from src.data.dataset import _epoch_seed

    assert _epoch_seed(42, 0, 5, 100) != _epoch_seed(43, 0, 5, 100)


# --- shell wrappers -------------------------------------------------------------------

import os  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _bash(script, args, **env):
    return subprocess.run(
        ["bash", str(REPO / "scripts" / script), *args],
        cwd=REPO,
        env={**os.environ, **env},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize(
    "script,args",
    [
        (
            "submit_asymmetric_lodo_all.sh",
            ["--run-name-prefix", "resnet18-asymmetric-v2"],
        ),
        (
            "submit_ssl_finetune_all.sh",
            [
                "--run-name-prefix",
                "vits16-mae-v2",
                "--pretrain-run-prefix",
                "mae-vits16-v2",
            ],
        ),
    ],
)
def test_orchestrators_refuse_train_seed(script, args):
    r = _bash(script, args, TRAIN_SEED="43")
    assert r.returncode == 1
    assert "TRAIN_SEED" in r.stderr


@pytest.mark.parametrize(
    "env,msg",
    [
        ({"SEED": "43x"}, "SEED must be"),
        ({"SUFFIX": "a/b"}, "SUFFIX must be"),
        ({"SUFFIX": "../x"}, "SUFFIX must be"),
        ({"SUFFIX": "a..b"}, "'..'"),
    ],
)
def test_crop_diag_rejects_unsafe_seed_and_suffix(env, msg):
    r = _bash("submit_crop_shortcut_diag.sh", [], **env)
    assert r.returncode == 1
    assert msg in r.stderr
