"""Optimizer and LR-schedule recipe for fine-tuning (shared by `_train_fold`).

Every knob is read from `cfg["training"]` and defaults to the v2 behaviour, so a config
that sets none of them builds exactly `AdamW(model.parameters(), lr, weight_decay)` with a
constant learning rate:

    warmup_epochs   linear LR warm-up over this many epochs (default 0)
    lr_schedule     "constant" (default) or "cosine" (decays towards 0 after warm-up; the last
                    epoch keeps a small residual factor, ~0.0012 for 45 post-warm-up epochs;
                    the peak LR is held for two epochs, `warmup` and `warmup + 1`)
    layer_decay     per-layer LR scale for ViT blocks, e.g. 0.75 (default 1.0 = off)
    wd_skip_norm_bias  no weight decay on biases/norm/cls params (default False; implied by
                    layer_decay, MAE convention)
"""

from __future__ import annotations

import math
import re

import torch
import torch.nn as nn

LR_CONSTANT = "constant"
DEFAULT_WARMUP_EPOCHS = 0
DEFAULT_LAYER_DECAY = 1.0
LR_COSINE = "cosine"
_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.")
_NO_DECAY_NAMES = ("cls_token", "pos_embed")


def _layer_id(name: str, num_layers: int) -> int:
    """0 for the embeddings, i+1 for block i, num_layers+1 for final norm and head."""
    m = _BLOCK_RE.match(name)
    if m:
        return int(m.group(1)) + 1
    if name.startswith(("patch_embed.", *_NO_DECAY_NAMES)):
        return 0
    return num_layers + 1


def build_optimizer(model: nn.Module, train_cfg: dict) -> torch.optim.Optimizer:
    """AdamW, optionally with layer-wise LR decay and no wd on norm/bias parameters."""
    lr = train_cfg["learning_rate"]
    wd = train_cfg["weight_decay"]
    layer_decay = train_cfg.get("layer_decay", DEFAULT_LAYER_DECAY)
    skip_norm_bias = (
        train_cfg.get("wd_skip_norm_bias", False) or layer_decay != DEFAULT_LAYER_DECAY
    )
    if layer_decay == DEFAULT_LAYER_DECAY and not skip_norm_bias:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    block_ids = [
        int(m.group(1))
        for n, _ in model.named_parameters()
        if (m := _BLOCK_RE.match(n))
    ]
    num_layers = max(block_ids) + 1 if block_ids else 0
    groups: dict[tuple[int, bool], dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        layer = _layer_id(name, num_layers)
        no_decay = skip_norm_bias and (p.ndim == 1 or name in _NO_DECAY_NAMES)
        scale = layer_decay ** (num_layers + 1 - layer)
        key = (layer, no_decay)
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": lr * scale,
                "base_lr": lr * scale,
                "weight_decay": 0.0 if no_decay else wd,
            }
        groups[key]["params"].append(p)
    return torch.optim.AdamW(list(groups.values()), lr=lr, weight_decay=wd)


def lr_factor(epoch: int, total_epochs: int, train_cfg: dict) -> float:
    """LR multiplier for 1-indexed `epoch`: linear warm-up, then constant or cosine."""
    warmup = train_cfg.get("warmup_epochs", DEFAULT_WARMUP_EPOCHS)
    schedule = train_cfg.get("lr_schedule", LR_CONSTANT)
    if schedule not in (LR_CONSTANT, LR_COSINE):
        raise ValueError(
            f"lr_schedule must be 'constant' or 'cosine', got {schedule!r}"
        )
    if epoch <= warmup:
        return epoch / warmup
    if schedule == LR_CONSTANT:
        return 1.0
    span = max(total_epochs - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup - 1) / span))


def apply_lr_schedule(
    optimizer: torch.optim.Optimizer, epoch: int, total_epochs: int, train_cfg: dict
) -> float:
    """Set each group's LR to base_lr * lr_factor(epoch). No-op for the default recipe."""
    if not train_cfg.get("warmup_epochs", DEFAULT_WARMUP_EPOCHS) and (
        train_cfg.get("lr_schedule", LR_CONSTANT) == LR_CONSTANT
    ):
        return 1.0
    factor = lr_factor(epoch, total_epochs, train_cfg)
    for g in optimizer.param_groups:
        g.setdefault("base_lr", g["lr"])
        g["lr"] = g["base_lr"] * factor
    return factor
