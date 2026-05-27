"""SymMamba: 对称性感知双路径 Mamba 网络.

Arch Scan + Cross Scan + CGF×3 + EDL Head. ~17M params.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from .edl_head import EDLHead


# ---- Mamba 核心 --------------------------------------------------

class MambaBlock(nn.Module):
    """单个 Mamba (S6) 块: SSM + residual + LayerNorm."""

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        inner = dim * expand
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, inner * 2)
        self.conv1d = nn.Conv1d(inner, inner, d_conv, padding=d_conv - 1,
                                groups=inner)
        # x_proj: delta + B + C
        n_ssm = d_state * 2 + inner
        self.x_proj = nn.Linear(inner, n_ssm)
        self.dt_scale = nn.Parameter(torch.ones(inner))
        self.A_log = nn.Parameter(
            -torch.log(torch.arange(1, d_state + 1).float().view(1, -1)))
        self.D = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        # 1D conv + SiLU
        x_c = self.conv1d(x_in.transpose(1, 2))[..., :L].transpose(1, 2)
        x_c = F.silu(x_c)

        # SSM 参数
        proj = self.x_proj(x_c)
        inner = x_in.size(-1)
        d_state = self.A_log.size(-1)

        delta_b_c = proj  # [B, L, d_state*2 + inner]
        delta = F.softplus(delta_b_c[..., :inner] * self.dt_scale)
        B_ssm = delta_b_c[..., inner:inner + d_state]
        C_ssm = delta_b_c[..., inner + d_state:inner + d_state * 2]

        A = -torch.exp(self.A_log)  # [1, d_state]

        # checkpoint 丢弃 forward 中间状态, backward 时重算 (O(1) 显存)
        y_ssm = cp.checkpoint(self._ssm_scan, delta, A, B_ssm, C_ssm,
                              self.D, use_reentrant=False)

        y = y_ssm * F.silu(z)
        y = self.out_proj(y)
        return residual + self.dropout(y)

    @staticmethod
    def _ssm_scan(delta, A, B_ssm, C_ssm, D):
        """逐帧 SSM 扫描 — O(1) 中间状态.

        每步操作 [B, inner, d_state] = [B, 192, 16] ≈ 0.4 MB.
        通过 checkpoint 丢弃 autograd 保存的 h 状态, backward 重算.
        """
        B_sz, L, inner = delta.shape
        d_state = A.size(-1)
        A = A.view(1, d_state)  # [1, d_state]

        h = torch.zeros(B_sz, inner, d_state, device=delta.device)
        ys = []
        for t in range(L):
            d_t = delta[:, t, :].unsqueeze(-1)        # [B, inner, 1]
            dA_t = torch.exp(d_t * A)                  # [B, inner, d_state]
            dB_t = d_t * B_ssm[:, t, :].unsqueeze(-2) # [B, inner, d_state]
            h = dA_t * h + dB_t
            y_t = (h * C_ssm[:, t, :].unsqueeze(-2)).sum(-1) + D  # [B, inner]
            ys.append(y_t)
        return torch.stack(ys, dim=1)


# ---- Patch Embedding ----------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=96, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, patch_size, stride=patch_size)
        self.norm = nn.BatchNorm2d(embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.proj(x)))


# ---- Arch Scan ---------------------------------------------------

class ArchScanStage(nn.Module):
    def __init__(self, dim, num_blocks, d_state=16, d_conv=4, expand=2,
                 dropout=0.1, downsample=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            MambaBlock(dim, d_state, d_conv, expand, dropout)
            for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Conv2d(dim, dim * 2, 2, stride=2) if downsample else None

    def forward(self, x):
        B, C, H, W = x.shape
        x_seq = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        for blk in self.blocks:
            x_seq = blk(x_seq)
        x_seq = self.norm(x_seq)
        x = x_seq.reshape(B, H, W, C).permute(0, 3, 1, 2)
        new_dim = C * 2 if self.down else C
        if self.down:
            x = self.down(x)
        return x, new_dim


# ---- Cross Scan --------------------------------------------------

class CrossScanStage(nn.Module):
    def __init__(self, dim, num_blocks, d_state=16, d_conv=4, expand=2,
                 dropout=0.1, downsample=False, use_sym_head=True):
        super().__init__()
        self.mamba = nn.ModuleList([
            MambaBlock(dim, d_state, d_conv, expand, dropout)
            for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Conv2d(dim, dim * 2, 2, stride=2) if downsample else None
        self.use_sym_head = use_sym_head

    def forward(self, x):
        B, C, H, W = x.shape
        half = W // 2

        def scan_half(xh):
            B_, C_, H_, W_ = xh.shape
            xs = xh.permute(0, 2, 3, 1).reshape(B_, H_ * W_, C_)
            for blk in self.mamba:
                xs = blk(xs)
            xs = self.norm(xs)
            return xs.reshape(B_, H_, W_, C_).permute(0, 3, 1, 2)

        left_out = scan_half(x[:, :, :, :half])
        right_out = scan_half(x[:, :, :, half:])

        s_sym = torch.zeros(B, device=x.device)
        if self.use_sym_head:
            lg = F.adaptive_avg_pool2d(left_out, 1).flatten(1)
            rg = F.adaptive_avg_pool2d(right_out, 1).flatten(1)
            s_sym = F.cosine_similarity(
                F.normalize(lg, dim=-1), F.normalize(rg, dim=-1))

        x_out = torch.cat([left_out, right_out], dim=-1)
        new_dim = C * 2 if self.down else C
        if self.down:
            x_out = self.down(x_out)
        return x_out, new_dim, s_sym


# ---- CGF ---------------------------------------------------------

class CrossGatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(self, f_arch, f_cross):
        g = self.gate(torch.cat([f_arch, f_cross], dim=1))
        g = g.unsqueeze(-1).unsqueeze(-1)
        return g * f_arch + (1 - g) * f_cross


# ---- Full SymMamba -----------------------------------------------

class SymMamba(nn.Module):
    """SymMamba V1: Arch + Cross Scan + CGF×3 + EDL Head.

    Args:
        in_channels: 3 (RGB).
        num_classes: 4 (Dean 0-3).
        embed_dim: 96, patch embedding dim.
        depths: (4,2,2) Mamba blocks per stage.
        d_state: 16, SSM state dim.
        d_conv: 4, SSM conv kernel.
        expand: 2, channel expansion.
        dropout: 0.1, Mamba dropout.
        edl_hidden: 256, EDL head hidden dim.
        edl_dropout: 0.3, EDL head dropout.
        use_sym_head: True for Cross Scan symmetry score.
    """

    def __init__(self, in_channels=3, num_classes=4, embed_dim=96,
                 depths=(4, 2, 2), d_state=16, d_conv=4, expand=2,
                 dropout=0.1, edl_hidden=256, edl_dropout=0.3,
                 use_sym_head=True, patch_size=4):
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)
        dim = embed_dim

        # Stage 1
        self.arch_s1 = ArchScanStage(dim, depths[0], d_state, d_conv, expand, dropout)
        self.cross_s1 = CrossScanStage(dim, depths[0], d_state, d_conv, expand,
                                       dropout, use_sym_head=use_sym_head)
        self.cgf1 = CrossGatedFusion(dim)

        # Stage 2
        self.arch_s2 = ArchScanStage(dim, depths[1], d_state, d_conv, expand,
                                     dropout, downsample=True)
        self.cross_s2 = CrossScanStage(dim, depths[1], d_state, d_conv, expand,
                                       dropout, downsample=True, use_sym_head=use_sym_head)
        dim *= 2
        self.cgf2 = CrossGatedFusion(dim)

        # Stage 3
        self.arch_s3 = ArchScanStage(dim, depths[2], d_state, d_conv, expand,
                                     dropout, downsample=True)
        self.cross_s3 = CrossScanStage(dim, depths[2], d_state, d_conv, expand,
                                       dropout, downsample=True, use_sym_head=use_sym_head)
        dim *= 2
        self.cgf3 = CrossGatedFusion(dim)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.edl_head = EDLHead(dim * 3, num_classes, edl_hidden, edl_dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(x)  # [B,96,64,128]

        f_arch, _ = self.arch_s1(x)
        f_cross, _, s_sym = self.cross_s1(x)
        f = self.cgf1(f_arch, f_cross)

        f_arch, _ = self.arch_s2(f)
        f_cross, _, _ = self.cross_s2(f)
        f = self.cgf2(f_arch, f_cross)

        f_arch, _ = self.arch_s3(f)
        f_cross, _, _ = self.cross_s3(f)
        f_cgf3 = self.cgf3(f_arch, f_cross)

        fused = torch.cat([f_cgf3, f_arch, f_cross], dim=1)
        z = self.gap(fused).flatten(1)

        out = self.edl_head(z)
        out["features"] = z
        out["s_sym"] = s_sym
        return out
