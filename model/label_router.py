"""PostFusionLabelRouter：标签条件交叉注意力的后置精炼模块。

以三尺度共识（可拼接三模态 token）为 KV，输出 5 个营养标签的残差修正
与注意力归因。门控和残差头均零初始化，初始 delta = 0。
"""

import torch
import torch.nn as nn


class PostFusionLabelRouter(nn.Module):
    """标签条件后融合路由器。

    Args:
        latent_dim: Fusion 输出维度
        work_dim:   路由器内部工作维度
        num_labels: 标签数
        num_scales: 共识尺度数
        hidden:     残差头隐藏维度
        dropout:    残差头 dropout
        num_heads:  交叉注意力头数
    """

    def __init__(self,
                 latent_dim=384,
                 work_dim=512,
                 num_labels=5,
                 num_scales=3,
                 hidden=None,
                 dropout=0.3,
                 num_heads=4
                 ):
        super().__init__()
        self.latent_dim = latent_dim
        self.work_dim = work_dim
        self.num_labels = num_labels
        self.num_scales = num_scales
        self.hidden = hidden if hidden is not None else max(128, work_dim // 2)

        self.fused_proj = nn.Linear(latent_dim, work_dim)
        self.consensus_proj = nn.Linear(latent_dim, work_dim)
        self.modality_proj = nn.Linear(latent_dim, work_dim)

        self.label_queries = nn.Parameter(torch.randn(num_labels, work_dim) * 0.02)

        self.q_norm = nn.LayerNorm(work_dim)
        self.kv_norm = nn.LayerNorm(work_dim)

        # 查询=5 个标签，KV=三尺度共识 + 三模态 token
        self.attn = nn.MultiheadAttention(work_dim, num_heads, batch_first=True)

        # 恒等启动：注意力分支初始强度为 0
        self.gamma = nn.Parameter(torch.zeros(1))

        self.delta_head = nn.Sequential(
            nn.Linear(2 * work_dim, self.hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, 1),
        )

        # 恒等启动：末层零初始化 → 初始 delta = 0
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def forward(self, fused, consensuses, modality_tokens=None):
        """
        Args:
            fused:           [B, latent_dim] 全局融合向量
            consensuses:     长度 S 的列表，每项 [B, latent_dim]
            modality_tokens: [B, 3, latent_dim] 三模态 token（可选）

        Returns:
            delta: [B, num_labels] 残差修正
            attn:  [B, num_labels, K] 注意力权重，K = S 或 S+3
        """
        B = fused.shape[0]

        fused_w = self.fused_proj(fused)  # [B, work_dim]

        kv = torch.stack(
            [self.consensus_proj(c) for c in consensuses],
            dim=1
        )  # [B, S, W]
        if modality_tokens is not None:
            mod = self.modality_proj(modality_tokens)  # [B, 3, W]
            kv = torch.cat([kv, mod], dim=1)  # [B, S+3, W]

        q = self.q_norm(self.label_queries).unsqueeze(0).expand(B, -1, -1)
        kv = self.kv_norm(kv)

        attended, attn = self.attn(q, kv, kv, need_weights=True)
        label_feat = q + self.gamma * attended

        ctx = fused_w.unsqueeze(1).expand(-1, self.num_labels, -1)
        h = torch.cat([label_feat, ctx], dim=-1)  # [B, 5, 2W]

        delta = self.delta_head(h).squeeze(-1)  # [B, 5]
        return delta, attn