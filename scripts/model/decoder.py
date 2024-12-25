import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float32
from torch import Tensor


class Decoder(nn.Module):
    def __init__(self, embed_dim: int, num_layers: int, num_heads: int, max_len: int = 64):
        super(Decoder, self).__init__()
        dec_layer = nn.TransformerDecoderLayer(
            embed_dim,
            num_heads,
            dim_feedforward=3 * embed_dim,
            activation=F.gelu,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            bias=False,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers, norm=nn.LayerNorm(embed_dim))
        self.pos_embeds = nn.Embedding(max_len, embed_dim)  # TODO: Replace with AliBi or RoPE

    def forward(
        self,
        x: Float32[Tensor, "B N C"],
        padding_mask: Float32[Tensor, "B N"],
        enc_out: Float32[Tensor, "B N_enc C_enc"],
    ) -> Float32[Tensor, "B N C"]:
        _, N, _ = x.shape
        pos_embeds = self.pos_embeds(torch.arange(N, device=x.device))
        x += pos_embeds

        causal_mask = nn.Transformer.generate_square_subsequent_mask(N, device=x.device)
        x = self.decoder.forward(
            x, enc_out, tgt_mask=causal_mask, tgt_key_padding_mask=padding_mask, tgt_is_causal=True
        )

        return x
