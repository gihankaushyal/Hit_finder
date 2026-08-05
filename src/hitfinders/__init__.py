# src/hitfinders/__init__.py
"""Pluggable hitfinder backends for the asymmetric training pipeline."""

from __future__ import annotations

from src.hitfinders.base import Hitfinder, MockHitfinder
from src.hitfinders.gpu import GPUHitfinder
from src.hitfinders.numpy_pf8 import NumpyPF8Hitfinder
from src.hitfinders.pf8 import PF8Hitfinder

__all__ = [
    "Hitfinder",
    "MockHitfinder",
    "PF8Hitfinder",
    "NumpyPF8Hitfinder",
    "GPUHitfinder",
    "get_hitfinder",
]


def get_hitfinder(cfg: dict) -> Hitfinder:
    """Instantiate the hitfinder specified in the config dict.

    Config key: cfg["hitfinder"]["backend"] — one of:
        "pf8"       CrystFEL libcrystfel.so via C bridge (requires compiled .so)
        "pf8_numpy" Pure Python/NumPy reimplementation of PeakFinder8
        "gpu"       User-provided PyTorch script (gpu_script_path required)
        "mock"      Deterministic test fixture (MockHitfinder)

    Args:
        cfg: Config dict as returned by load_config().

    Returns:
        A Hitfinder instance.

    Raises:
        ValueError: If the backend name is missing or unrecognised.
    """
    hf_cfg = cfg.get("hitfinder", {})
    backend = hf_cfg.get("backend")
    if not backend:
        raise ValueError(
            "cfg['hitfinder']['backend'] is required. "
            "Valid options: 'pf8', 'pf8_numpy', 'gpu', 'mock'."
        )

    if backend == "pf8":
        return PF8Hitfinder(
            threshold=hf_cfg.get("pf8_threshold", 800.0),
            min_snr=hf_cfg.get("pf8_min_snr", 5.0),
            min_pix_count=hf_cfg.get("pf8_min_pix_count", 2),
            max_pix_count=hf_cfg.get("pf8_max_pix_count", 200),
            local_bg_radius=hf_cfg.get("pf8_local_bg_radius", 3),
            min_res=hf_cfg.get("pf8_min_res", 0),
            max_res=hf_cfg.get("pf8_max_res", 0),
            use_saturated=bool(hf_cfg.get("pf8_use_saturated", False)),
        )
    if backend == "pf8_numpy":
        from src.hitfinders.numpy_pf8 import NumpyPF8Hitfinder

        return NumpyPF8Hitfinder(
            threshold=hf_cfg.get("pf8_threshold", 800.0),
            min_snr=hf_cfg.get("pf8_min_snr", 5.0),
            min_pix_count=hf_cfg.get("pf8_min_pix_count", 2),
            max_pix_count=hf_cfg.get("pf8_max_pix_count", 200),
            local_bg_radius=hf_cfg.get("pf8_local_bg_radius", 3),
            min_res=hf_cfg.get("pf8_min_res", 0),
            max_res=hf_cfg.get("pf8_max_res", 0),
        )
    if backend == "gpu":
        script_path = hf_cfg.get("gpu_script_path", "")
        if not script_path:
            raise ValueError(
                "cfg['hitfinder']['gpu_script_path'] is required for backend='gpu'."
            )
        return GPUHitfinder(
            script_path=script_path,
            device=hf_cfg.get("gpu_device", "cuda"),
        )
    if backend == "mock":
        return MockHitfinder()
    raise ValueError(
        f"Unknown hitfinder backend {backend!r}. "
        "Valid options: 'pf8', 'pf8_numpy', 'gpu', 'mock'."
    )
