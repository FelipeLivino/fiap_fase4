"""Vision Transformer model factories for CardioIA Vision."""

from __future__ import annotations

from typing import Literal

from torch import nn
from torchvision import models


VisionTransformerName = Literal["vit_b_16", "swin_t"]


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    """Set trainability for all parameters in a module."""
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad


def _weights_or_none(weights_class: object, pretrained: bool):
    if not pretrained:
        return None
    return weights_class.DEFAULT


def create_vit_b_16(
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Create ViT-B/16 with a binary classification head."""
    weights = _weights_or_none(models.ViT_B_16_Weights, pretrained)
    model = models.vit_b_16(weights=weights)

    if freeze_backbone:
        set_requires_grad(model, False)

    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, 1)
    return model


def create_swin_t(
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Create Swin Transformer Tiny with a binary classification head."""
    weights = _weights_or_none(models.Swin_T_Weights, pretrained)
    model = models.swin_t(weights=weights)

    if freeze_backbone:
        set_requires_grad(model, False)

    in_features = model.head.in_features
    model.head = nn.Linear(in_features, 1)
    return model


def create_vision_transformer(
    model_name: VisionTransformerName = "vit_b_16",
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Create a supported transformer-based vision model."""
    if model_name == "vit_b_16":
        return create_vit_b_16(pretrained=pretrained, freeze_backbone=freeze_backbone)
    if model_name == "swin_t":
        return create_swin_t(pretrained=pretrained, freeze_backbone=freeze_backbone)
    raise ValueError(f"Unsupported vision transformer model: {model_name}")


def unfreeze_transformer_final_blocks(
    model: nn.Module,
    model_name: VisionTransformerName = "vit_b_16",
) -> None:
    """Unfreeze final transformer blocks for partial fine-tuning."""
    if model_name == "vit_b_16":
        set_requires_grad(model.encoder.layers[-2:], True)
        set_requires_grad(model.heads, True)
        return

    if model_name == "swin_t":
        set_requires_grad(model.features[-2:], True)
        set_requires_grad(model.head, True)
        return

    raise ValueError(f"Unsupported vision transformer model: {model_name}")


def transformer_model_summary(model: nn.Module) -> dict[str, int]:
    """Return total, trainable, and frozen parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(total - trainable),
    }


def get_trainable_parameter_names(model: nn.Module) -> list[str]:
    """Return trainable parameter names for notebook diagnostics."""
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
