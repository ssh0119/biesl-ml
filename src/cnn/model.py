"""Compact multichannel 1D-CNN for 40-second physiological windows."""
from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, *, stride: int = 1, pool: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel, stride=stride, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool) if pool > 1 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Phase2CNN1D(nn.Module):
    """Input ``(batch, 3, 10000)``; output five class logits."""

    def __init__(self, in_channels: int = 3, n_classes: int = 5, dropout: float = 0.35):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, 15, stride=2, pool=2),
            ConvBlock(32, 64, 9, stride=2, pool=2),
            ConvBlock(64, 128, 5, stride=2, pool=2),
            ConvBlock(128, 128, 3, stride=1, pool=2),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(96, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        z = torch.cat((self.avg_pool(z), self.max_pool(z)), dim=1).squeeze(-1)
        return self.classifier(z)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
