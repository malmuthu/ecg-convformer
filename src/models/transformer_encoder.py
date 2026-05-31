"""
tranformer_encoder.py
Positional encoding and transformer encoder for ECG-ConvFormer.

Takes the convolutional stem output of shape (batch, d_model, L) and
pproduces a context-enriched sequence of shape (batch, d_model, L).

The transformer lets each temporal position attend to all others, capturing
long-range dependencies that convolutions miss.
"""

import math
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding from 'Attention is All You Need'.

    Adds position information to each tokem so the transformer knows the temporal order
    of the sequence. Without this, the transformer treats the input as an
    unordered set.

    Encoding formula for position pos, dimension i:
        PE[pos, 2i]     = sin(pos / 10000^(2i/d_model))
        PE[pos, 1i+1]   = cos(pos / 10000^(2i/d_model))

    Args:
        d_model : embedding dimension (match conv stem output)
        max_len : maximum sequence length to pre-compute 
        dropout : dropout applied after adding positional encoding
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Pre-compute sinusoidal positional encoding matrix
        # Buils PE matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)    # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)    # even dims
        pe[:, 1::2] = torch.cos(position * div_term)    # odd dims

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : shape (batch, seq_len, d_model)
        Returns:
            shape (batch, seq_len, d_model)
        """

        x = x + self.pe[:, :x.size(1), :]
        out = self.dropout(x)
        return out
    
class TransformerEncoderBlock(nn.Module):
    """
    Single transformer encoder layer.

    Structure:
        x -> LayerNorm -> MutiHeadAttenton -> +x (residual)
          -> LayerNorm -> FeedForward -> +x (residual)

    This uses layer norm before attention for more stable training.

    Args:
        d_model : embedding dimension
        n_heads : number of attention heads
        d_ff    : feed-forward hidden dimension
        dropout : dropout applied inside attenton and feed-forward
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads,
                                          dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : shape (batch, seq_len, d_model)
        Returns:
            shape (batch, seq_len, d_model)
        """

        # Attention sub-layer
        out = self.norm1(x)
        attn_out, _ = self.attn(out, out, out)
        x = x + self.dropout(attn_out)

        # Feed-forward sub-layer
        out = self.norm2(x)
        out = self.ff(out)
        x = x + self.dropout(out)

        return x
    
class TransformerEncoder(nn.Module):
    """
    Full transformer encoder: positional encoding + N stacked layers

    Args:
        d_model     : embedding dimension
        n_heads     : attention heads per layer
        d_ff        : feed-forward hidden dimension
        n_layers    : number of stacked TransformerEncoderBlocks
        max_len     : max sequence length for positional encoding
        dropout     : dropout probability
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4, d_ff: int = 256,
                 n_layers: int = 2, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                                    dropout=dropout) for i in range(n_layers)])
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : shape (batch, d_model, L)
        Returns:
            shape (batch, d_model, L)
        """

        x = x.transpose(1, 2)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        out = x.transpose(1, 2)
        return out


