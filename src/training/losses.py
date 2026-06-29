"""
losses.py
Loss functions for ECG beat classification.

Two options:
    1. Weighted Cross-Entropy - standard approach, inverse frequency weights
    2. Focal Loss - down-weights easy examples, focuses training on hard ones
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_weighted_cross_entropy(class_weights: torch.Tensor) -> nn.CrossEntropyLoss:
    """
    Standard cross-entropy loss with inverse frequency weights.
    The weight for each class scales the loss contribution of that class -
    rare classes contribute more to the total loss.

    Args:
        class_weights: shape (n_classes,) - one weight per class

    Returns:
        nn.CrossEntropyLoss with weight registered
    """

    return nn.CrossEntropyLoss(weight=class_weights)

class FocalLoss(nn.Module):
    """
    From Lin et al. 2017

    Modifies cross-entropy by down-weighting easy examples so training focuses on
    hard, misclassified examples. Effective for severe class imbalance.

    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

        Where:
            p_t     : model's estimated probability for the correct class
            gamma   : focusing parameter
            alpha   : per-class weight
    Args:
        gamma           : focusing parameter (default 2.0 from literature)
        class_weights   : Optional per-class weights shape (n_classes,)
        reduction       : 'mean' or 'sum'
    """

    def __init__(self, gamma: float = 2.0, class_weights: torch.Tensor = None,
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : shape (batch, n_classes) - raw model output before softmax
            targets : shape (batch,) - integer class labels
        Returns:
            scalar loss value
        """

        # One loss value per sample
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction='none')

        probs = torch.exp(-ce_loss)

        focal_weight = (1 - probs) ** self.gamma

        focal_loss = ce_loss * focal_weight

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return ValueError(f"reduction must be 'mean' or 'sum'")