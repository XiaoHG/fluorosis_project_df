"""损失函数: L_EDL + L_ord + L_cont + L_boundary, 按 04_Model_Design/03_loss_functions.md."""

import torch
import torch.nn.functional as F


def edl_loss(alpha: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
    """标准证据损失 (Sensoy et al., NeurIPS 2018).

    L_EDL = Σ_k (y_k-b_k)² + Σ_k y_k·(S-α_k)²/S
    """
    S = alpha.sum(dim=-1, keepdim=True)
    b = (alpha - 1) / S.clamp(min=1e-6)
    mse = ((y_onehot - b) ** 2).sum(dim=-1)
    evidence_reg = (y_onehot * (S - alpha) ** 2 / S.clamp(min=1e-6)).sum(dim=-1)
    return (mse + evidence_reg).mean()


def ordinal_regularization(alpha: torch.Tensor) -> torch.Tensor:
    """有序信念正则化: 禁止 b_k 跳跃。L_ord = Σ_{k=1}^{K-2} max(0, b_{k-1}+b_{k+1}-2b_k)"""
    S = alpha.sum(dim=-1, keepdim=True)
    b = (alpha - 1) / S.clamp(min=1e-6)
    K = b.size(1)
    penalty = sum(F.relu(b[:, k-1] + b[:, k+1] - 2*b[:, k]) for k in range(1, K-1))
    return penalty.mean()


def ordinal_contrastive_loss(z: torch.Tensor, y: torch.Tensor,
                             temperature: float = 0.07) -> torch.Tensor:
    """有序对比损失: 拉近相邻等级, 推远跨级。"""
    z = F.normalize(z, p=2, dim=-1)
    sim = torch.matmul(z, z.T) / temperature
    y_int = y if y.dim() == 1 else y.argmax(dim=-1)
    pos_mask = (y_int.unsqueeze(0) - y_int.unsqueeze(1)).abs() <= 1
    pos_mask.fill_diagonal_(False)
    loss = 0.0
    for i in range(z.size(0)):
        pos = sim[i][pos_mask[i]]
        if pos.numel() == 0:
            continue
        loss -= torch.log(pos.exp().sum() / sim[i].exp().sum().clamp(min=1e-8))
    if isinstance(loss, float):
        return torch.tensor(loss, device=z.device)
    return loss / z.size(0)


def boundary_uncertainty_loss(alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """边界不确定性先验: 邻级样本 u 允许升高。L_boundary = -Σ_{|y_j-y_i|=1} log(u_i)"""
    S = alpha.sum(dim=-1)
    u = alpha.size(1) / S.clamp(min=1e-6)
    y_int = y if y.dim() == 1 else y.argmax(dim=-1)
    adj_mask = (y_int.unsqueeze(0) - y_int.unsqueeze(1)).abs() == 1
    if adj_mask.sum() == 0:
        return torch.tensor(0.0, device=alpha.device)
    return -torch.log(u[adj_mask.any(dim=1)].clamp(min=1e-6)).mean()


def compute_total_loss(alpha: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                       loss_cfg: dict) -> tuple[torch.Tensor, dict]:
    """总损失 = w1·L_EDL + w2·L_ord + w3·L_cont + w4·L_boundary.

    Args:
        alpha: [B, K] Dirichlet 浓度参数。
        y: [B] 整数标签 (0-3)。
        z: [B, D] 特征向量。
        loss_cfg: {"edl": {"weight":1.0}, "ordinal":{"weight":0.1}, ...}
    """
    y_onehot = F.one_hot(y, num_classes=alpha.size(1)).float()
    w = loss_cfg

    l_edl = edl_loss(alpha, y_onehot)
    l_ord = ordinal_regularization(alpha)
    l_cont = ordinal_contrastive_loss(
        z, y, w.get("contrastive", {}).get("temperature", 0.07))
    l_bound = boundary_uncertainty_loss(alpha, y)

    total = (
        w.get("edl", {}).get("weight", 1.0) * l_edl +
        w.get("ordinal", {}).get("weight", 0.1) * l_ord +
        w.get("contrastive", {}).get("weight", 0.05) * l_cont +
        w.get("boundary", {}).get("weight", 0.01) * l_bound
    )

    comps = {"L_EDL": l_edl.item(), "L_ord": l_ord.item(),
             "L_cont": l_cont.item(), "L_bound": l_bound.item()}
    return total, comps
