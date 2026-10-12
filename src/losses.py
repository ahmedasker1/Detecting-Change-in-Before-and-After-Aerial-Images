# -*- coding: utf-8 -*-
"""Loss functions suited to highly imbalanced change masks.

`BCEDiceLoss` = BCE_WEIGHT * BCEWithLogitsLoss + DICE_WEIGHT * soft Dice.
The Dice term directly optimises overlap of the (rare) changed pixels, which
keeps training stable even when <10% of pixels are labelled as change.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Soft Dice loss on probabilities (works on raw logits)."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.flatten(1)
        targets = targets.flatten(1)
        inter = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """BCE (optionally with pos_weight for class imbalance) + soft Dice."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5,
                 pos_weight: float | None = None):
        super().__init__()
        pw = torch.tensor([pos_weight], dtype=torch.float32) if pos_weight else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (self.bce_weight * self.bce(logits, targets)
                + self.dice_weight * self.dice(logits, targets))
