"""Baseline 模型: B1 ResNet50 (+EDL), B2 ResNet50+CORAL, B3 ViT (+EDL)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .edl_head import EDLHead


class BaselineResNet50(nn.Module):
    """B1: ResNet50 + EDL Head (use_edl=True) or Linear head."""

    def __init__(self, num_classes=4, pretrained=True,
                 use_edl=True, edl_hidden=256, edl_dropout=0.3):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        w = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet50(weights=w)
        self.backbone.fc = nn.Identity()
        self.use_edl = use_edl
        feat_dim = 2048
        if use_edl:
            self.edl_head = EDLHead(feat_dim, num_classes, edl_hidden, edl_dropout)
        else:
            self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        f = self.backbone(x)
        if self.use_edl:
            out = self.edl_head(f)
            out["features"] = f
            return out
        return self.head(f)


class BaselineResNet50CORAL(nn.Module):
    """B2: ResNet50 + CORAL ordinal regression head.

    CORAL: K-1 binary classifiers sharing backbone.
    Outputs P(y > k) for k = 0..K-2.
    For compatibility with train.py, converts to EDL-like format.
    """

    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        w = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet50(weights=w)
        self.backbone.fc = nn.Identity()
        self.num_classes = num_classes
        self.head = nn.Linear(2048, num_classes - 1)

    def forward(self, x):
        f = self.backbone(x)
        probs = torch.sigmoid(self.head(f))  # [B, K-1]
        pred = probs.sum(dim=-1).round().long().clamp(0, self.num_classes - 1)
        # Synthetic alpha for loss compatibility (not a true Dirichlet)
        alpha = F.softmax(torch.cat([probs, torch.zeros_like(probs[:, :1])], dim=-1), dim=-1) * 10 + 1
        S = alpha.sum(dim=-1, keepdim=True)
        return {
            "alpha": alpha,
            "belief": (alpha - 1) / S.clamp(min=1e-6),
            "u": alpha.size(1) / S.squeeze(-1).clamp(min=1e-6),
            "pred": pred,
            "features": f,
        }


class BaselineViT(nn.Module):
    """B3: ViT-B/16 + EDL Head (use_edl=True) or MLP head."""

    def __init__(self, num_classes=4, pretrained=False, img_size=224,
                 use_edl=True, edl_hidden=256, edl_dropout=0.3):
        super().__init__()
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        w = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = vit_b_16(weights=w, image_size=img_size)
        self.backbone.heads = nn.Identity()
        self.use_edl = use_edl
        feat_dim = 768
        if use_edl:
            self.edl_head = EDLHead(feat_dim, num_classes, edl_hidden, edl_dropout)
        else:
            self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        f = self.backbone(x)
        if self.use_edl:
            out = self.edl_head(f)
            out["features"] = f
            return out
        return self.head(f)


def create_baseline(name: str, **kwargs) -> nn.Module:
    """Factory: 'resnet50' | 'resnet50_coral' | 'vit'."""
    registry = {
        "resnet50": BaselineResNet50,
        "resnet50_coral": BaselineResNet50CORAL,
        "vit": BaselineViT,
    }
    if name not in registry:
        raise ValueError(f"Unknown baseline: {name}. Choose from {list(registry.keys())}")
    return registry[name](**kwargs)
