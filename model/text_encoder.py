"""基于 Transformer 的食材列表编码器：ingredient IDs → [B, embed_dim]。

仅使用食材类别语义，不使用 grams，避免份量信息泄露。
"""

import torch
import torch.nn as nn


class IngredientTextEncoder(nn.Module):
    def __init__(self, vocab_size=556, embed_dim=256, max_len=50,
                 nhead=4, num_layers=2, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.pos_encoding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, ingr_ids, grams=None, mask=None):
        B, N = ingr_ids.shape
        x = self.embedding(ingr_ids)
        x = x + self.pos_encoding[:, :N, :]

        if mask is None:
            mask = (ingr_ids != 0).bool()

        x = self.transformer(x, src_key_padding_mask=~mask)
        x = masked_mean(x, mask)
        x = self.norm(x)
        x = self.out_proj(x)
        return x


def masked_mean(x, mask):
    mask = mask.unsqueeze(-1).float()
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
