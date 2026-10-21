"""QuatODE-Fusion：三模态多尺度四元数 ODE 融合模块。

RGB / Depth / Text 映射到四元数三虚轴，在 Hamilton 力场驱动下做辛积分
ODE 演化，并以全程模长轨迹聚合得到各尺度共识。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .label_router import PostFusionLabelRouter


def hamilt_mat(a):
    """四元数 a 的 Hamilton 左乘矩阵 L(a)：L(a) @ b = a ⊗ b
    分量约定：q_state = (w_consensus, rgb_axis, depth_axis, text_axis)"""
    a0, a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    row0 = torch.stack([a0, -a1, -a2, -a3], dim=-1)
    row1 = torch.stack([a1, a0, -a3, a2], dim=-1)
    row2 = torch.stack([a2, a3, a0, -a1], dim=-1)
    row3 = torch.stack([a3, -a2, a1, a0], dim=-1)
    return torch.stack([row0, row1, row2, row3], dim=-2)


class QuaternionLinear(nn.Module):
    """四元数线性层：q_out_j = Σ_i W_ji ⊗ q_in_i + b_j。"""

    def __init__(self, in_ch, out_ch, scale=0.05):
        super().__init__()
        self.w = nn.Parameter(torch.randn(out_ch, in_ch, 4) * scale)
        self.b = nn.Parameter(torch.zeros(out_ch, 4))

    def forward(self, q):  # q: [B, in_ch, 4]
        L = hamilt_mat(self.w)  # [out_ch, in_ch, 4, 4]
        out = torch.einsum('bic,oicd->bod', q, L)  # [B, out_ch, 4]
        return out + self.b.unsqueeze(0)


class QuatODEFusion(nn.Module):
    """四元数 Neural ODE 融合（辛积分 + 轨迹聚合）。"""

    def __init__(self, feature_dims=(160, 320, 640), text_dim=256,
                 latent_dim=384, num_ode_steps=4):
        super().__init__()
        self.feature_dims = feature_dims
        self.latent_dim = latent_dim
        self.num_ode_steps = num_ode_steps
        self.num_scales = len(feature_dims)

        self.rgb_proj = nn.ModuleList(
            [nn.Linear(c, latent_dim) for c in feature_dims])
        self.depth_proj = nn.ModuleList(
            [nn.Linear(c, latent_dim) for c in feature_dims])
        self.text_proj = nn.ModuleList(
            [nn.Linear(text_dim, latent_dim) for _ in feature_dims])

        for m in list(self.rgb_proj) + list(self.depth_proj) + list(self.text_proj):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

        self.force_field = nn.Sequential(
            QuaternionLinear(latent_dim, 2 * latent_dim),
            nn.ReLU(),
            QuaternionLinear(2 * latent_dim, latent_dim),
        )

        self.project_to_latent = nn.ModuleList(
            [nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim),
            ) for _ in feature_dims])

        self.merge = nn.Linear(latent_dim * self.num_scales, latent_dim)

    def _ode_step(self, q_state):
        """单步 ODE：d(q_state)/dt = f_θ(q_state)"""
        return self.force_field(q_state)

    def _integrate(self, q_state):
        """Leapfrog 辛积分演化 num_ode_steps 步。

          p_{n+1/2} = p_n + 0.5·dt·f(q_n)
          q_{n+1}   = q_n + dt·p_{n+1/2}
          p_{n+1}   = p_{n+1/2} + 0.5·dt·f(q_{n+1})

        返回终点状态与全程模长轨迹 [B, latent_dim, num_steps+1]。
        """
        q_traj = [q_state.norm(dim=-1)]

        p_state = torch.zeros_like(q_state)
        for _ in range(self.num_ode_steps):
            p_state = p_state + 0.5 * self._ode_step(q_state)
            q_state = q_state + p_state
            p_state = p_state + 0.5 * self._ode_step(q_state)
            q_traj.append(q_state.norm(dim=-1))

        return q_state, torch.stack(q_traj, dim=-1)

    def _aggregate_traj(self, traj):
        """对全程模长（含 q(0)）等权平均。"""
        return traj.mean(dim=-1)

    def forward(self, rgb_feats, depth_feats, text_feat):
        """返回多尺度融合向量与各尺度共识。"""
        consensuses = []
        for s in range(self.num_scales):
            r = F.relu(self.rgb_proj[s](rgb_feats[s].mean(dim=[2, 3])))
            d = F.relu(self.depth_proj[s](depth_feats[s].mean(dim=[2, 3])))
            t = F.relu(self.text_proj[s](text_feat))

            # q = (w, r, d, t)
            q_state = torch.stack([0.5 * (r + d), r, d, t, ], dim=-1)

            q_state, traj = self._integrate(q_state)
            c = self.project_to_latent[s](self._aggregate_traj(traj))
            consensuses.append(c)

        fused = self.merge(torch.cat(consensuses, dim=-1))
        return fused, consensuses


class NutritionHead(nn.Module):
    """营养预测头。输入多尺度融合向量 [B, latent_dim]，零残差。"""

    def __init__(self, latent_dim=384, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 5),
        )

    def forward(self, fused):
        return self.net(fused)


class QuatODEFusionModule(nn.Module):
    """QuatODE-Fusion + PostFusionLabelRouter 的完整融合模块。"""

    WORK_DIM = 512
    MOD_DROPOUT_P = 0.2

    def __init__(self, feature_dims=(160, 320, 640), text_dim=256,
                 latent_dim=384, head_hidden=128, head_dropout=0.3,
                 num_ode_steps=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.text_dim = text_dim
        self.work_dim = self.WORK_DIM
        self.mod_dropout_p = self.MOD_DROPOUT_P

        self.ode_fusion = QuatODEFusion(
            feature_dims=feature_dims,
            text_dim=text_dim,
            latent_dim=latent_dim,
            num_ode_steps=num_ode_steps,
        )

        self.head = NutritionHead(latent_dim, head_hidden, head_dropout)

        self.text_proj_lr = nn.Linear(text_dim, latent_dim)
        self.label_router = PostFusionLabelRouter(
            latent_dim=latent_dim,
            work_dim=self.work_dim,
            num_labels=5,
            num_scales=len(feature_dims),
            hidden=max(128, self.work_dim // 2),
            dropout=head_dropout,
        )
        self.null_token = nn.Parameter(
            torch.zeros(1, 1, latent_dim), requires_grad=False)

    def _build_modality_tokens(self, rgb_feats, depth_feats, text_feat):
        """三模态 GAP 表示 → [B, 3, latent_dim]：视觉三级投影求和 + 文本投影。"""
        B = text_feat.shape[0]
        rgb_gap = torch.zeros(B, self.latent_dim, device=text_feat.device,
                              dtype=text_feat.dtype)
        depth_gap = torch.zeros_like(rgb_gap)
        for s in range(len(rgb_feats)):
            rgb_gap = rgb_gap + F.relu(
                self.ode_fusion.rgb_proj[s](rgb_feats[s].mean(dim=[2, 3])))
            depth_gap = depth_gap + F.relu(
                self.ode_fusion.depth_proj[s](depth_feats[s].mean(dim=[2, 3])))
        text_gap = self.text_proj_lr(text_feat)
        return torch.stack([rgb_gap, depth_gap, text_gap], dim=1)

    def _modality_dropout(self, modality_tokens):
        """训练时对每个样本随机屏蔽一个模态 token（用 null_token 替换）。"""
        B = modality_tokens.shape[0]
        device = modality_tokens.device
        idx = torch.randint(0, 3, (B,), device=device)
        onehot = F.one_hot(idx, num_classes=3).to(modality_tokens.dtype)
        drop = (torch.rand(B, device=device) < self.mod_dropout_p) \
            .to(modality_tokens.dtype)
        mask = (drop.unsqueeze(-1) * onehot).unsqueeze(-1)  # [B, 3, 1]
        null = self.null_token.to(modality_tokens.dtype)
        return modality_tokens * (1.0 - mask) + null * mask

    def forward(self, rgb_feats, depth_feats, text_feat,
                return_intermediates=False):
        """
        Args:
            rgb_feats:   三级特征列表 (S1/S2/S3)
            depth_feats: 三级特征列表 (S1/S2/S3)
            text_feat:   [B, text_dim] 文本编码特征

        Returns:
            nutrition: [B, 5]
            （return_intermediates=True 时返回 (nutrition, dict)）
        """
        fused, consensuses = self.ode_fusion(rgb_feats, depth_feats, text_feat)
        nutrition = self.head(fused)

        inter = None
        if return_intermediates:
            inter = {
                'multi_scale_consensus': consensuses,
                'fused': fused,
            }

        # 残差精炼（恒等启动）
        modality_tokens = self._build_modality_tokens(
            rgb_feats, depth_feats, text_feat)
        if self.training and self.mod_dropout_p > 0:
            modality_tokens = self._modality_dropout(modality_tokens)

        delta, attn = self.label_router(fused, consensuses, modality_tokens)
        nutrition = nutrition + delta

        if return_intermediates:
            inter['label_attn'] = attn
            inter['delta'] = delta
            return nutrition, inter
        return nutrition

    @torch.no_grad()
    def explain(self, rgb_feats, depth_feats, text_feat):
        """推理时中间表示输出：多尺度融合共识（可解释性钩子）。"""
        self.eval()
        fused, consensuses = self.ode_fusion(rgb_feats, depth_feats, text_feat)
        return {
            'multi_scale_consensus': consensuses,
            'fused': fused,
        }
