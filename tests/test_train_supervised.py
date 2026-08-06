"""Smoke tests for the supervised training loop."""

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
