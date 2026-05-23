"""损失函数: L_EDL + L_KL + L_ord + L_cont + L_boundary, 按 04_Model_Design/03_loss_functions.md."""

import torch
import torch.nn.functional as F


def cross_entropy_loss(alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on Dirichlet belief distribution.

    belief = (alpha-1)/S  is a valid probability simplex.
    Using log-belief as log-prob for CE is numerically stable
    and provides a direct classification gradient, bypassing
    EDL's problematic evidence regularizer on small datasets.
    """
    S = alpha.sum(dim=-1, keepdim=True)
    log_belief = torch.log(alpha - 1 + 1e-8) - torch.log(S + 1e-8)
    if y.dim() == 2:
        y_idx = y.argmax(dim=-1)
    else:
        y_idx = y.long()
    return F.nll_loss(log_belief, y_idx)


def edl_loss(alpha: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
    """标准证据损失 (Sensoy et al., NeurIPS 2018).

    L_EDL = Σ_k (y_k-b_k)² + Σ_k y_k·(S-α_k)²/S
    """
    S = alpha.sum(dim=-1, keepdim=True)
    b = (alpha - 1) / S.clamp(min=1e-6)
    mse = ((y_onehot - b) ** 2).sum(dim=-1)
    evidence_reg = (y_onehot * (S - alpha) ** 2 / S.clamp(min=1e-6)).sum(dim=-1)
    return (mse + evidence_reg).mean()


def kl_regularization(alpha: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
    """KL 散度正则: 惩罚错误类别的证据累积, 防止 EDL 塌缩.

    KL[Dir(π|α̃) || Dir(π|1)]  where  α̃ = y + (1-y)⊙α
    只约束错误类别, 不影响正确类别的证据增长.
    """
    K = alpha.size(1)
    alpha_tilde = y_onehot + (1 - y_onehot) * alpha
    S_tilde = alpha_tilde.sum(dim=-1, keepdim=True)
    kl = (torch.lgamma(S_tilde.squeeze(-1))
          - torch.lgamma(alpha_tilde).sum(dim=-1)
          - torch.lgamma(torch.tensor(K, dtype=alpha.dtype, device=alpha.device))
          + ((alpha_tilde - 1) * (torch.digamma(alpha_tilde)
                                  - torch.digamma(S_tilde))).sum(dim=-1))
    return kl.mean()


def _schedule_weight(full_weight: float, epoch: int, start: int, ramp: int) -> float:
    """Linear ramp from 0 to full_weight between [start, start+ramp] epochs."""
    if epoch < start:
        return 0.0
    if epoch >= start + ramp:
        return full_weight
    return full_weight * (epoch - start) / ramp


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
                       loss_cfg: dict, epoch: int = 0) -> tuple[torch.Tensor, dict]:
    """总损失 = w1·L_EDL + w2·L_KL + w3·L_ord + w4·L_cont + w5·L_boundary.

    辅助损失 (L_ord/L_cont/L_boundary) 通过 schedule 延迟介入,
    给 backbone 时间学习基础特征, 避免早期训练崩塌.

    Args:
        alpha: [B, K] Dirichlet 浓度参数。
        y: [B] 整数标签 或 [B, K] soft labels (CutMix)。
        z: [B, D] 特征向量。
        loss_cfg: 含 weights 和可选的 schedule {aux_start, aux_ramp}。
        epoch: 当前 epoch, 用于 schedule 计算。
    """
    if y.dim() == 2 and y.size(1) == alpha.size(1):
        y_onehot = y.float()
    else:
        y_onehot = F.one_hot(y.long(), num_classes=alpha.size(1)).float()
    w = loss_cfg
    sch = loss_cfg.get("schedule", {})
    aux_start = sch.get("aux_start", 10)
    aux_ramp = sch.get("aux_ramp", 10)

    # primary losses
    w_ce = _schedule_weight(
        w.get("ce", {}).get("weight", 1.0), epoch,
        sch.get("ce_start", 0), sch.get("ce_ramp", 1))
    w_edl = _schedule_weight(
        w.get("edl", {}).get("weight", 0.0), epoch,
        sch.get("edl_start", 0), sch.get("edl_ramp", 1))
    w_kl = _schedule_weight(
        w.get("kl", {}).get("weight", 0.0), epoch,
        sch.get("kl_start", 0), sch.get("kl_ramp", 1))
    # auxiliary losses
    w_ord = _schedule_weight(
        w.get("ordinal", {}).get("weight", 0.0), epoch, aux_start, aux_ramp)
    w_cont = _schedule_weight(
        w.get("contrastive", {}).get("weight", 0.0), epoch, aux_start, aux_ramp)
    w_bound = _schedule_weight(
        w.get("boundary", {}).get("weight", 0.0), epoch, aux_start, aux_ramp)

    l_ce = cross_entropy_loss(alpha, y) if w_ce > 0 else torch.tensor(0.0, device=alpha.device)
    l_edl = edl_loss(alpha, y_onehot) if w_edl > 0 else torch.tensor(0.0, device=alpha.device)
    l_kl = kl_regularization(alpha, y_onehot) if w_kl > 0 else torch.tensor(0.0, device=alpha.device)

    total = w_ce * l_ce + w_edl * l_edl + w_kl * l_kl

    l_ord = torch.tensor(0.0, device=alpha.device)
    l_cont = torch.tensor(0.0, device=alpha.device)
    l_bound = torch.tensor(0.0, device=alpha.device)

    if w_ord > 0:
        l_ord = ordinal_regularization(alpha)
        total = total + w_ord * l_ord
    if w_cont > 0:
        l_cont = ordinal_contrastive_loss(
            z, y, w.get("contrastive", {}).get("temperature", 0.07))
        total = total + w_cont * l_cont
    if w_bound > 0:
        l_bound = boundary_uncertainty_loss(alpha, y)
        total = total + w_bound * l_bound

    comps = {"L_CE": l_ce.item(), "L_EDL": l_edl.item(), "L_KL": l_kl.item(),
             "L_ord": l_ord.item(), "L_cont": l_cont.item(),
             "L_bound": l_bound.item()}
    return total, comps
