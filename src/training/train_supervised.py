"""Shared training utilities used by train_lodo.py."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str | torch.device,
) -> dict[str, float]:
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.float().to(device), y.long().to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return {"loss": total_loss / max(n, 1)}
