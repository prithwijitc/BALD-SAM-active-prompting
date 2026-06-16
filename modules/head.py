"""NoImageHead: lightweight CNN refinement head for prompt-aware segmentation."""

import torch
import torch.nn as nn


class NoImageHead(nn.Module):
    """
    Takes [sam_logits, pos_map, neg_map] (3 x H x W) and outputs refined logits (1 x H x W).

    Architecture:
        Conv(3 -> H, 3x3, pad=1) + ReLU
        Conv(H -> H, 3x3, pad=1) + ReLU
        Conv(H -> 1, 1x1)  <- Laplace target (final_conv)

    Deliberately small: SAM does the heavy lifting; this head only refines.
    No BatchNorm — avoids frozen-BN / Laplace interaction issues.
    """

    def __init__(self, channels_in: int = 3, hidden_channels: int = 16):
        super().__init__()
        H = hidden_channels

        self.backbone = nn.Sequential(
            nn.Conv2d(channels_in, H, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(H, H, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Final prediction layer — only this layer is Bayesianized by Laplace
        self.final_conv = nn.Conv2d(H, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) — [sam_logits, pos_map, neg_map]
        Returns:
            (B, 1, H, W) refined logits
        """
        return self.final_conv(self.backbone(x))

    def final_layer_parameters(self):
        """Return parameters of final_conv only (for Laplace)."""
        return list(self.final_conv.parameters())
