"""Transfer learning model factories for CardioIA Vision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torchvision import models


TransferModelName = Literal[
    "resnet50",
    "efficientnet_b0",
    "efficientnet_b3",
    "densenet121",
]


@dataclass(frozen=True)
class TransferLearningConfig:
    """Configuration for transfer learning model creation."""

    model_name: TransferModelName
    pretrained: bool = True
    freeze_backbone: bool = True
    dropout: float = 0.30


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    """Set trainability for all parameters in a module."""
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad


def _weights_or_none(weights_class: object, pretrained: bool):
    if not pretrained:
        return None
    return weights_class.DEFAULT


def create_resnet50(
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout: float = 0.30,
) -> nn.Module:
    """Create ResNet50 with a binary classification head."""
    weights = _weights_or_none(models.ResNet50_Weights, pretrained)
    model = models.resnet50(weights=weights)

    if freeze_backbone:
        set_requires_grad(model, False)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 1),
    )
    return model


def create_efficientnet_b0(
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout: float = 0.30,
) -> nn.Module:
    """Create EfficientNetB0 with a binary classification head."""
    weights = _weights_or_none(models.EfficientNet_B0_Weights, pretrained)
    model = models.efficientnet_b0(weights=weights)

    if freeze_backbone:
        set_requires_grad(model, False)

    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 1),
    )
    return model


def create_efficientnet_b3(
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout: float = 0.30,
) -> nn.Module:
    """Create EfficientNetB3 with a binary classification head."""
    weights = _weights_or_none(models.EfficientNet_B3_Weights, pretrained)
    model = models.efficientnet_b3(weights=weights)

    if freeze_backbone:
        set_requires_grad(model, False)

    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 1),
    )
    return model


def create_densenet121(
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout: float = 0.30,
) -> nn.Module:
    """Create DenseNet121 with a binary classification head."""
    weights = _weights_or_none(models.DenseNet121_Weights, pretrained)
    model = models.densenet121(weights=weights)

    if freeze_backbone:
        set_requires_grad(model, False)

    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 1),
    )
    return model


def create_transfer_model(
    model_name: TransferModelName,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout: float = 0.30,
) -> nn.Module:
    """Create one of the supported transfer learning models."""
    factories = {
        "resnet50": create_resnet50,
        "efficientnet_b0": create_efficientnet_b0,
        "efficientnet_b3": create_efficientnet_b3,
        "densenet121": create_densenet121,
    }
    if model_name not in factories:
        raise ValueError(f"Unsupported transfer model: {model_name}")

    return factories[model_name](
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
    )


def unfreeze_final_blocks(model: nn.Module, model_name: TransferModelName) -> None:
    """Unfreeze final feature blocks for partial fine-tuning."""
    if model_name == "resnet50":
        set_requires_grad(model.layer4, True)
        set_requires_grad(model.fc, True)
        return

    if model_name in {"efficientnet_b0", "efficientnet_b3"}:
        set_requires_grad(model.features[-1], True)
        set_requires_grad(model.classifier, True)
        return

    if model_name == "densenet121":
        set_requires_grad(model.features.denseblock4, True)
        set_requires_grad(model.features.norm5, True)
        set_requires_grad(model.classifier, True)
        return

    raise ValueError(f"Unsupported transfer model: {model_name}")


def get_trainable_parameter_names(model: nn.Module) -> list[str]:
    """Return names of trainable parameters for notebook diagnostics."""
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def transfer_model_summary(model: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(total - trainable),
    }
