"""
baselines.py
Baseline models for ablation study.

ConvOnlyBaseline: ConvStem + Global Average Pooling + Classifier
    No transformer encoder - tests whether the transformer adds value over
    pure convolutional feature extraction.
"""

from src.models.conv_stem import ConvStem
import torch
import torch.nn as nn

class ConvOnlyBaseline(nn.Module):
    """
    Ablation baseline: convolutional stem only, no transformer.

    Identitcal classifier head to ECGConvFormer for a fair comparison.
    The only difference is the absence of the TransformerEncoder.

    Args:
        n_classes   : number of output classes (5 for AAMI EC57)
        d_model     : convolutional stem output channels
        dropout     : dropout probability
    """

    def __init__(self, n_classes: int = 5, d_model: int = 128, dropout: float = 0.1):
        super().__init__()

        self.conv_stem = ConvStem(d_model=d_model, dropout=dropout)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : shape (batch, 1, 187)
        Returns:
            logits  : shape (batch, n_classes)
        """

        x = self.conv_stem(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        logits = self.classifier(x)
        return logits
    
    def count_parameters(self) -> int:
        """
        Returns total number of trainable parameters.
        """

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params