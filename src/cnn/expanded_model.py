"""Wider feature extractor with the original embedding interface."""
from torch import nn
from src.cnn.model import ConvBlock, Phase2CNN1D


class ExpandedPhase2CNN1D(Phase2CNN1D):
    """64/128/256/256 convolution channels and a 256-unit hidden layer."""

    def __init__(self, in_channels=3, n_classes=5, dropout=0.35):
        super().__init__(in_channels, n_classes, dropout)
        self.features = nn.Sequential(
            ConvBlock(in_channels, 64, 15, stride=2, pool=2),
            ConvBlock(64, 128, 9, stride=2, pool=2),
            ConvBlock(128, 256, 5, stride=2, pool=2),
            ConvBlock(256, 256, 3, stride=1, pool=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2), nn.Linear(256, n_classes),
        )
