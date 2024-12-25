import torch
import torch.nn as nn
from data import tokens
from jaxtyping import Float32
from torch import Tensor


class TokenEmbeddings(nn.Module):
    def __init__(self, embed_size: int, coord_dims: int = 2):
        self.cls_embed = nn.Embedding(self.num_classes, embed_size)
        self.coo_embed = nn.Linear(coord_dims, embed_size)
        self.reembed = nn.Linear(embed_size * 2, embed_size)
        self.norm = nn.LayerNorm(embed_size)

    @property
    def num_classes(self) -> int:
        return len(tokens.TokenCls)

    def forward(self, tokens: tokens.TokenBatch) -> Float32[Tensor, "B N C"]:
        cls_emb = self.cls_embed(tokens.cls)
        coo_emb = self.coo_embed(tokens.coord)
        emb = torch.cat([cls_emb, coo_emb], dim=-1)
        emb = self.reembed(emb)
        emb = self.norm(emb)

        return emb
