"""Loss functions for head training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        intersection = (probs_flat * targets_flat).sum(dim=1)
        denom = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    BCE (with optional pos_weight for class imbalance) + lambda * Dice.

    pos_weight: scalar tensor = bg_pixels / fg_pixels, computed from the training manifest.
                Upweights the foreground in BCE to counteract background dominance.
    """

    def __init__(self, dice_lambda: float = 0.5, pos_weight: torch.Tensor = None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()
        self.dice_lambda = dice_lambda

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, targets) + self.dice_lambda * self.dice(logits, targets)
