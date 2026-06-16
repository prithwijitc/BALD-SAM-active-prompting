"""Segmentation metrics: IoU and Dice computed from logits."""

import torch


def iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.0) -> float:
    """Compute mean IoU over a batch. logits and targets are (B, 1, H, W)."""
    preds = (logits > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets).clamp(0, 1).sum(dim=(1, 2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def dice_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.0) -> float:
    """Compute mean Dice over a batch."""
    preds = (logits > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    denom = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + 1e-6) / (denom + 1e-6)
    return dice.mean().item()
