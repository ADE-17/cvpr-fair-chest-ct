"""Focal loss for imbalanced classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss. Handles class imbalance via alpha and gamma."""

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean", label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # logits: (B, C), targets: (B,)
        num_classes = logits.size(-1)
        probs = F.softmax(logits, dim=-1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        if self.label_smoothing > 0:
            targets_one_hot = (
                targets_one_hot * (1 - self.label_smoothing)
                + self.label_smoothing / num_classes
            )
        pt = (probs * targets_one_hot).sum(dim=-1)
        focal_weight = (1 - pt) ** self.gamma
        ce = F.cross_entropy(logits, targets, reduction="none")
        loss = focal_weight * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
