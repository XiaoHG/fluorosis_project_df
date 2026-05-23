"""EDL 分类头: α → u, b_k, pred。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EDLHead(nn.Module):
    """Dirichlet 证据分类头 (Sensoy et al., NeurIPS 2018)。

    Input:  z [B, D]
    Output: dict {alpha, belief, u, pred}
    """

    def __init__(self, in_dim: int, num_classes: int = 4,
                 hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, z: torch.Tensor) -> dict:
        evidence = self.fc2(self.drop(self.act(self.fc1(z))))
        alpha = F.softplus(evidence) + 1  # α ≥ 1, 比 exp 更稳定
        S = alpha.sum(dim=-1, keepdim=True)
        return {
            "alpha": alpha,
            "belief": (alpha - 1) / S.clamp(min=1e-6),
            "u": alpha.size(1) / S.squeeze(-1).clamp(min=1e-6),
            "pred": alpha.argmax(dim=-1),
        }
