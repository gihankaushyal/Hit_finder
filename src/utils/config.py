"""YAML config loader: deep-merges configs/base.yaml with a model-specific config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_DEFAULT_BASE = Path(__file__).parents[2] / "configs" / "base.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_config(
    config_path: str | Path,
    base_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load a model YAML and deep-merge configs/base.yaml underneath it.

    Model values win; base fills in missing keys at every nesting level.
    """
    config_path = Path(config_path)
    base_path = Path(base_path) if base_path is not None else _DEFAULT_BASE

    with open(base_path) as f:
        base_cfg: dict[str, Any] = yaml.safe_load(f) or {}
    with open(config_path) as f:
        model_cfg: dict[str, Any] = yaml.safe_load(f) or {}

    return _deep_merge(base_cfg, model_cfg)
