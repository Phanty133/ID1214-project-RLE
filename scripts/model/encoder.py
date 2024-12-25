from typing import cast

import torch.nn as nn
from jaxtyping import Float32
from torch import FloatTensor, Tensor
from transformers import Swinv2Config, Swinv2Model
from transformers.models.swinv2.modeling_swinv2 import Swinv2ModelOutput


class Encoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        swin_ckpt="microsoft/swinv2-base-patch4-window12to24-192to384-22kto1k-ft",
    ):
        super(Encoder, self).__init__()
        self.embed_dim = embed_dim
        self.img_encoder = Swinv2Model.from_pretrained(swin_ckpt)
        self.reproj = nn.Linear(self.img_encoder_config.hidden_size, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    @property
    def img_encoder_config(self) -> Swinv2Config:
        return self.img_encoder.config  # To cast to Swinv2Config

    def forward(self, image: Float32[Tensor, "B C H W"]) -> Float32[Tensor, "B N C"]:
        img_enc_out = cast(Swinv2ModelOutput, self.img_encoder.forward(cast(FloatTensor, image)))
        img_emb = self.reproj(img_enc_out.last_hidden_state)
        img_emb = self.norm(img_emb)

        return img_emb
