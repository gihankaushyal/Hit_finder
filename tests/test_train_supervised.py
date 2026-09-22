"""Smoke tests for the supervised training loop."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.training.train_supervised import train_one_epoch


def tiny_loader():
    X = torch.randn(4, 1, 224, 224)
    y = torch.tensor([0, 1, 0, 1])
    return DataLoader(TensorDataset(X, y), batch_size=2)


def test_train_one_epoch_returns_loss():
    from src.models.supervised import build_supervised_model

    model = build_supervised_model("resnet18", pretrained=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    metrics = train_one_epoch(model, tiny_loader(), optimizer, criterion, device="cpu")
    assert "loss" in metrics
    assert metrics["loss"] >= 0.0


def test_train_one_epoch_updates_weights():
    from src.models.supervised import build_supervised_model

    model = build_supervised_model("resnet18", pretrained=False)
    params_before = [p.clone() for p in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    train_one_epoch(model, tiny_loader(), optimizer, criterion, device="cpu")
    assert any(not torch.equal(a, b) for a, b in zip(params_before, model.parameters()))


def test_train_one_epoch_returns_hit_frac():
    from src.models.supervised import build_supervised_model

    X = torch.randn(10, 1, 224, 224)
    # 3 hits (label=1) out of 10 samples -> expected hit_frac = 0.3
    y = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    loader = DataLoader(TensorDataset(X, y), batch_size=3)

    model = build_supervised_model("resnet18", pretrained=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    metrics = train_one_epoch(model, loader, optimizer, criterion, device="cpu")

    assert "hit_frac" in metrics
    assert metrics["hit_frac"] == pytest.approx(0.3)
