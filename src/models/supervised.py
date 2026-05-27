"""ResNet18/50 fine-tuning via timm. Track 1 — supervised baseline."""

from __future__ import annotations

import timm
import torch.nn as nn


def build_supervised_model(
    backbone: str,
    pretrained: bool,
    num_classes: int = 2,
) -> nn.Module:
    """Create a single-channel ResNet classifier.

    Uses timm in_chans=1 to average pretrained RGB first-layer weights into a
    single-channel equivalent. num_classes=2 for hit/non-hit binary classification.

    Raises RuntimeError if backbone name is not recognized by timm.
    """
    try:
        model = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=1,
            num_classes=num_classes,
        )
    except (ValueError, RuntimeError, ImportError) as e:
        raise RuntimeError(f"timm could not create model '{backbone}': {e}") from e
    return model
