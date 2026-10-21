"""QuatODE-Net：四元数 Neural ODE 多模态营养预测网络。

RGB / Depth 经 MambaVision-T 提取三级特征，食材列表经文本编码器编码，
由 QuatODEFusionModule 融合后回归 5 项营养指标。
"""

import torch
import torch.nn as nn

from .mambavision_backbone import mamba_vision_T
from .text_encoder import IngredientTextEncoder
from .quat_ode_fusion import QuatODEFusionModule


class QuatODENet(nn.Module):
    """QuatODE-Net 主模型。"""

    def __init__(self, num_ingredients=198, text_dim=256, latent_dim=384,
                 max_ingredients=20, num_ode_steps=4, head_hidden=128,
                 head_dropout=0.3):
        super().__init__()

        self.backbone = 'tiny'
        self.feature_dims = (160, 320, 640)

        self.encoder_rgb = mamba_vision_T(return_features=True, num_levels=3)
        self.encoder_depth = mamba_vision_T(return_features=True, num_levels=3)

        self.text_encoder = IngredientTextEncoder(
            vocab_size=num_ingredients, embed_dim=text_dim,
            max_len=max_ingredients,
        )

        self.fusion = QuatODEFusionModule(
            feature_dims=self.feature_dims,
            text_dim=text_dim,
            latent_dim=latent_dim,
            head_hidden=head_hidden,
            head_dropout=head_dropout,
            num_ode_steps=num_ode_steps,
        )

    def forward(self, rgb, depth, ingr_ids, mask,
                return_intermediates=True):
        """
        Args:
            rgb:       [B, 3, 224, 224]
            depth:     [B, 3, 224, 224]
            ingr_ids:  [B, max_ingredients]
            mask:      [B, max_ingredients]
            return_intermediates: 是否返回中间量（训练时为 True）

        Returns:
            若 return_intermediates=True: (nutrition [B,5], intermediates dict)
            否则: nutrition [B,5]
        """
        rgb_feats = self.encoder_rgb(rgb)
        depth_feats = self.encoder_depth(depth)
        text_feat = self.text_encoder(ingr_ids, mask=mask)

        return self.fusion(
            rgb_feats, depth_feats, text_feat,
            return_intermediates=return_intermediates)

    @torch.no_grad()
    def explain(self, rgb, depth, ingr_ids, mask):
        """推理时中间表示输出：多尺度融合共识。"""
        self.eval()
        rgb_feats = self.encoder_rgb(rgb)[:3]
        depth_feats = self.encoder_depth(depth)[:3]
        text_feat = self.text_encoder(ingr_ids, mask=mask)
        return self.fusion.explain(rgb_feats, depth_feats, text_feat)
