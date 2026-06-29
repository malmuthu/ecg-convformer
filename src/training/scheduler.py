"""
scheduler.py
Learning rate scheduling for ECG-ConvFormer training.

Uses cosine annealing with linear warmup. Warmup starts with a small LR and
gradually increases it, giving the model time to stabilize before full-speed training.

Schedule shape:
    Epochs 0 -> warmup_epochs   : LR increases linearly 0 -> base_lr
    Epochs warmup -> max_epochs : LR decreases following a cosine curve down to min_lr
"""

import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer: Optimizer, warmup_epochs: int,
                                    max_epochs: int, min_lr_ratio: float = 0.01,
                                    ) -> LambdaLR:
    """
    Cosine annealing schedule with linear warmup.

    Args:
        optimizer       : the optimizer to schedule
        warmup_epochs   : number of epochs for linear warmup
        max_epochs      : total training epochs
        min_lr_ratio    : minimum LR as fraction of base LR

    Returns:
        LambdaLR scheduler - call scheduler.step() once per epoch
    """

    def lr_lambda(current_epoch: int) -> float:
        """
        Returns a multiplicative factor for the base learning rate.
        The optimizer mutiplies base_lr by this factor each epoch.

        Args:
            current_epoch   : current training epoch
        Returns:
            float multiplier in range [min_lr_ratio, 1.0]
        """

        if current_epoch < warmup_epochs:
            # Linear increase
            return (current_epoch + 1) / warmup_epochs
        
        progress = (current_epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

        return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)

def get_optimizer(model, lr: float = 1e-3, weight_decay: float = 1e-4):
    """
    AdamW optimizer with sensible defaults for ECG-ConvFormer.

    Args:
        model       : nn.Module to optimize
        lr          : base learning rate
        weight_decay: L2 regularization strength
    """

    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)