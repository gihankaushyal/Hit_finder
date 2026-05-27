"""Tests for supervised model architectures."""

import pytest
import torch
from src.models.supervised import build_supervised_model


class TestBuildSupervisedModel:
    def test_resnet18_output_shape(self):
        model = build_supervised_model("resnet18", pretrained=False, num_classes=2)
        x = torch.zeros(2, 1, 224, 224)
        out = model(x)
        assert out.shape == (2, 2)

    def test_resnet50_output_shape(self):
        model = build_supervised_model("resnet50", pretrained=False, num_classes=2)
        x = torch.zeros(2, 1, 224, 224)
        out = model(x)
        assert out.shape == (2, 2)

    def test_model_deterministic_in_inference_mode(self):
        model = build_supervised_model("resnet18", pretrained=False, num_classes=2)
        model.train(False)
        x = torch.randn(1, 1, 224, 224)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_invalid_backbone_raises(self):
        with pytest.raises(RuntimeError):
            build_supervised_model("not_a_model", pretrained=False, num_classes=2)
