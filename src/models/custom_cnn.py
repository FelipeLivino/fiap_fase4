"""Custom CNN architecture created for the CardioIA Vision PBL."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Convolutional block used by the custom CardioIA CNN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CardioIACustomCNN(nn.Module):
    """CNN designed by the project team for binary chest X-ray classification.

    Output:
        A single logit per image. Use with `nn.BCEWithLogitsLoss`.
    """

    def __init__(self, dropout: float = 0.35) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, dropout=0.05),
            ConvBlock(32, 64, dropout=0.10),
            ConvBlock(64, 128, dropout=0.15),
            ConvBlock(128, 256, dropout=0.20),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x).squeeze(1)


def create_custom_cnn(dropout: float = 0.35) -> CardioIACustomCNN:
    """Factory used by notebooks and training scripts."""
    return CardioIACustomCNN(dropout=dropout)
