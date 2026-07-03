# src/hitfinders/__init__.py
"""Pluggable hitfinder backends for the asymmetric training pipeline."""

from __future__ import annotations

from src.hitfinders.base import Hitfinder, MockHitfinder
from src.hitfinders.gpu import GPUHitfinder
from src.hitfinders.pf8 import PF8Hitfinder

__all__ = ["Hitfinder", "MockHitfinder", "PF8Hitfinder", "GPUHitfinder", "get_hitfinder"]


def get_hitfinder(cfg: dict) -> Hitfinder:
    """Instantiate the hitfinder specified in the config dict.

    Config key: cfg["hitfinder"]["backend"] — one of "pf8", "gpu", "mock".

    Args:
        cfg: Config dict as returned by load_config().

    Returns:
        A Hitfinder instance.

    Raises:
        ValueError: If the backend name is unrecognised.
    """
    hf_cfg = cfg.get("hitfinder", {})
    backend = hf_cfg.get("backend", "pf8")

    if backend == "pf8":
        return PF8Hitfinder(
            threshold_snr=hf_cfg.get("pf8_threshold_snr", 5.0),
            min_peaks=hf_cfg.get("pf8_min_peaks", 1),
        )
    if backend == "gpu":
        return GPUHitfinder()
    if backend == "mock":
        return MockHitfinder()
    raise ValueError(
        f"Unknown hitfinder backend {backend!r}. "
        "Valid options: 'pf8', 'gpu', 'mock'."
    )
