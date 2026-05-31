"""
convformer.py
Full ECG-ConvFormer model.

Assembles ConvStem + TransformerEncoder + Classification Head into a 
single nn.Module for end-to-end training.

Architecture summary:
    Input (batch, 1, 187)
        ↓ ConvStem
    (batch, d_model, L)
        ↓ TransformerEncoder
    (batch, d_model, L)
        ↓ Global Average Pooling
    (batch, d_model)
        ↓ Classification Head
    (batch, n_classes)
"""

from src.models.conv_stem import ConvStem
from src.models.transformer_encoder import TransformerEncoder
import torch
import torch.nn as nn

class ECGConvFormer(nn.Module):
    """
    Hybrid convolutional-transformer model for ECG beat classification.

    The convolutional stem extracts local waveform morphology features.
    The transformer encoder captures global temporal relationships.
    Global average pooling aggregates the sequence into a fixed vector.
    The classification head maps to class probabilities.

    Args:
        n_classes   : number of output classes (5 fot AAMI EC57)
        d_model     : internal embedding dimension
        n_heads     : number of transformer attention heads
        d_ff        : transformer feed-forward hidden dimension
        n_layers    : number of transformer encoder layers
        dropout     : dropout probability throughout the model
    """

    def __init__(self, n_classes: int = 5, d_model: int = 128, n_heads: int = 4,
                 d_ff: int = 256, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        self.conv_stem = ConvStem(d_model=d_model, dropout=dropout)

        self.transformer = TransformerEncoder(d_model=d_model, n_heads=n_heads,
                                              d_ff=d_ff, n_layers=n_layers, dropout=dropout)
        
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
            x   : shape (batch, 1, 187) - raw ECG beat, single channel
        Returns:
            logits  : shape (batch, n_classes) - raw scores before softmax
        """

        x = self.conv_stem(x)
        x = self.transformer(x)
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