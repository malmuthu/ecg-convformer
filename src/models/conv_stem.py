"""
conv_stem.py
Residual convolutional stem for ECG-ConvFormer.

Takes a raw ECG beat of shape (batch, 1, 187) and produces a sequence of
local feature vectors of shape (batch, d_model, L) where L is the compressed
temporal length after strided convolutions.

Architecture:
    Input (1, 187)
        ↓
    ConvBlock 1 - kernel 7, stride 2, out_channels 32 -> (32, 94)
        ↓
    ConvBlock 2 - kernel 5, stride 2, out_channels 64 -> (64, 47)
        ↓
    ConvBlock 3 - kernel 3, stride 1, out_channels 128 -> (128, 47)
        ↓
    Projection - kernel 1, out_channels d_model -> (d_model, 47)
"""

import torch
import torch.nn as nn
from typing import Tuple

class ResidualBlock(nn.Module):
    """
    One residual convolutional block.

    Structure:
        x -> Conv1d -> BatchNorm -> ReLU -> Conv1d -> BatchNorm -> +x -> ReLU

    When stride > 1 or channels change, the skip connection uses a 1x1 conv to
    match dimensions before adding.

    Args:
        in_channels : number of input channels
        out_channels: number of output channels
        kernel_size : convolutional kernel size
        stride      : stride for the first conv
        dropout     : dropout probability applied after final ReLU
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, dropout: float = 0.1):
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                               stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape (batch, in_channels, length)
        Returns:
            shape (batch, out_channels, new_length)
        """

        residual = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out + residual)
        return self.dropout(out)
    
class ConvStem(nn.Module):
    """
    Full convolutional stem: stacks three ResidualBlocks then projects to
    d_model channels for input to the transformer encoder.

    Args:
        d_model : output channel dimension
        dropout : dropout applies inside each residual block
    """

    def __init__(self, d_model: int = 128, dropout: float = 0.1):
        super().__init__()

        self.blocks = nn.Sequential(
            ResidualBlock(in_channels=1, out_channels=32, kernel_size=7, stride=2, dropout=dropout),
            ResidualBlock(in_channels=32, out_channels=64, kernel_size=5, stride=2, dropout=dropout),
            ResidualBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, dropout=dropout),
        )

        self.project = nn.Sequential(
            nn.Conv1d(128, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : shape (batch, 1, 187)
        Returns:
            shape (batch, d_model, 47) 
        """

        x1 = self.blocks(x)
        x2 = self.project(x1)
        return x2
    
def get_conv_stem_out_len(input_len: int = 187) -> int:
    """
    Calculate the temporal length L after the conv stem, given an input of
    'input_length' samples.
    """
    # Stride 2, kernel 7, padding 3
    L = (input_len + 2 * 3 - 7) // 2 + 1
    # Stride 2, kernel 5, padding 2
    L = (L + 2 * 2 - 5) // 2 + 1
    return L