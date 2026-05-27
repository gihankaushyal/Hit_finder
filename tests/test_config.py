"""Tests for config loader."""

import pytest
import yaml
from src.utils.config import load_config


def test_load_config_returns_dict(tmp_path):
    base = tmp_path / "base.yaml"
    model = tmp_path / "model.yaml"
    base.write_text("seed: 42\ntraining:\n  lr: 0.001\n  epochs: 50\n")
    model.write_text("training:\n  lr: 0.0001\n")
    cfg = load_config(model, base_path=base)
    assert isinstance(cfg, dict)


def test_model_values_override_base(tmp_path):
    base = tmp_path / "base.yaml"
    model = tmp_path / "model.yaml"
    base.write_text("seed: 42\ntraining:\n  lr: 0.001\n  epochs: 50\n")
    model.write_text("training:\n  lr: 0.0001\n")
    cfg = load_config(model, base_path=base)
    assert cfg["training"]["lr"] == 0.0001
    assert cfg["training"]["epochs"] == 50


def test_base_fills_missing_keys(tmp_path):
    base = tmp_path / "base.yaml"
    model = tmp_path / "model.yaml"
    base.write_text("seed: 42\nwandb:\n  project: sfx\n")
    model.write_text("model:\n  backbone: resnet18\n")
    cfg = load_config(model, base_path=base)
    assert cfg["seed"] == 42
    assert cfg["wandb"]["project"] == "sfx"
    assert cfg["model"]["backbone"] == "resnet18"


def test_load_config_default_base_path(tmp_path, monkeypatch):
    import src.utils.config as cfg_module

    fake_base = tmp_path / "base.yaml"
    fake_base.write_text("seed: 99\n")
    monkeypatch.setattr(cfg_module, "_DEFAULT_BASE", fake_base)
    model = tmp_path / "model.yaml"
    model.write_text("model:\n  backbone: resnet18\n")
    cfg = load_config(model)
    assert cfg["seed"] == 99


def test_load_config_missing_file_raises(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("seed: 42\n")
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(missing, base_path=base)


def test_load_config_malformed_yaml_raises(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("seed: 42\n")
    bad = tmp_path / "bad.yaml"
    bad.write_text("key: [unclosed\n")
    with pytest.raises(yaml.YAMLError):
        load_config(bad, base_path=base)
