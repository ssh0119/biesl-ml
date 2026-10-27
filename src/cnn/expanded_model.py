"""Wider feature extractor with the original embedding interface."""
from torch import nn
from src.cnn.model import ConvBlock, Phase2CNN1D


class ExpandedPhase2CNN1D(Phase2CNN1D):
    """Double the first three channel widths; retain 256 pooled / 96 hidden features."""

    def __init__(self, in_channels=3, n_classes=5, dropout=0.35):
        super().__init__(in_channels, n_classes, dropout)
        self.features = nn.Sequential(
            ConvBlock(in_channels, 64, 15, stride=2, pool=2),
            ConvBlock(64, 128, 9, stride=2, pool=2),
            ConvBlock(128, 256, 5, stride=2, pool=2),
            ConvBlock(256, 128, 3, stride=1, pool=2),
        )
