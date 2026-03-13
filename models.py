"""SOTA 2D backbone + classifier for slice-level prediction and Attention MIL with optional adversarial gender head."""

import torch
import torch.nn as nn
from torch.autograd import Function

try:
    import timm
except ImportError:
    timm = None


def build_model(num_classes=4, model_name="convnext_tiny", pretrained=True):
    """
    Build classifier using timm backbone.
    Options: convnext_tiny, efficientnetv2_s, swin_tiny_patch4_window7_224
    """
    if timm is None:
        raise ImportError("timm is required. Install: pip install timm")

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=3,
    )
    return model


def get_backbone_feature_dim(model_name="convnext_tiny"):
    """Return feature dim for backbone with num_classes=0."""
    if timm is None:
        raise ImportError("timm is required")
    model = timm.create_model(model_name, pretrained=False, num_classes=0, in_chans=3)
    with torch.no_grad():
        out = model(torch.randn(1, 3, 224, 224))
    return out.shape[-1]


class SliceClassifier(nn.Module):
    """Wrapper for timm model; can add custom head if needed."""

    def __init__(self, num_classes=4, model_name="convnext_tiny", pretrained=True):
        super().__init__()
        self.backbone = build_model(num_classes, model_name, pretrained)
        self.num_classes = num_classes

    def forward(self, x):
        return self.backbone(x)


# --- Gradient Reversal for adversarial training ---
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambd=1.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambd)


# --- Attention-based Multiple Instance Learning (MIL) ---
class AttentionMIL(nn.Module):
    """
    Learn to weight slices: attention over slice features so the model can ignore
    empty slices and focus on slices with anomalies (e.g. tumor in 5–10 of 150).
    """

    def __init__(self, feature_dim, num_classes=4):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, features, mask=None, return_features=False):
        """
        features: (B, N, D)
        mask: (B, N), 1 = real slice, 0 = padding. If None, all positions used.
        Returns: logits (B, num_classes) and optionally aggregated features (B, D)
        """
        # Attention scores per slice
        scores = self.attention(features)  # (B, N, 1)
        if mask is not None:
            scores = scores.squeeze(-1).masked_fill(mask == 0, -1e9)
            scores = scores.unsqueeze(-1)
        weights = torch.softmax(scores, dim=1)  # (B, N, 1)
        aggregated = (weights * features).sum(dim=1)  # (B, D)
        logits = self.classifier(aggregated)
        if return_features:
            return logits, aggregated
        return logits


class ScanLevelMIL(nn.Module):
    """
    Backbone (no head) + Attention MIL. Forward: (B, N, 3, H, W) + mask -> (B, num_classes).
    """

    def __init__(self, num_classes=4, model_name="convnext_tiny", pretrained=True, grl_lambda=0.0):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required")
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=3,
        )
        feat_dim = get_backbone_feature_dim(model_name)
        self.mil = AttentionMIL(feat_dim, num_classes)
        # adversarial gender head (2 classes: male/female)
        self.grl = GradientReversal(lambd=grl_lambda)
        self.gender_head = nn.Linear(feat_dim, 2)
        self.num_classes = num_classes

    def forward(self, images, mask=None, return_gender=False):
        """
        images: (B, N, 3, H, W)
        mask: (B, N)
        """
        B, N, C, H, W = images.shape
        x = images.view(B * N, C, H, W)
        features = self.backbone(x)  # (B*N, D)
        features = features.view(B, N, -1)
        logits, agg_feat = self.mil(features, mask, return_features=True)
        if return_gender:
            rev = self.grl(agg_feat)
            gender_logits = self.gender_head(rev)
            return logits, gender_logits
        return logits
