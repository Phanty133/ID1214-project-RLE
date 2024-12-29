import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtune as tt
from jaxtyping import Float32
from torch import Tensor


class MHA(nn.MultiheadAttention):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.1,
        rope: bool = False,
    ):
        super(MHA, self).__init__(embed_dim, num_heads, dropout=dropout)
        head_dim = embed_dim // num_heads

        if rope:
            pos_embeds = tt.modules.RotaryPositionalEmbeddings(head_dim, max_seq_len=128)
        else:
            pos_embeds = None

        self.attn = tt.modules.MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            k_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            v_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            pos_embeddings=pos_embeds,
            is_causal=False,
            max_seq_len=128,
            kv_cache=None,
            attn_dropout=dropout,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = True,
        attn_mask: Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        B = query.size(0)
        N = query.size(1)
        dev = query.device
        mask = (
            nn.Transformer.generate_square_subsequent_mask(N, device=dev, dtype=torch.bool)
            .unsqueeze(0)
            .expand(B, -1, -1)
        )

        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool:
                key_padding_mask = key_padding_mask != 0

            key_padding_mask = key_padding_mask.view(B, 1, -1).expand(-1, N, -1)
            mask = mask | key_padding_mask

        return self.attn.forward(query, key, mask=~mask), None


class Decoder(nn.Module):
    def __init__(self, embed_dim: int, num_layers: int, num_heads: int, max_len: int = 64, rope=False):
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

        self.rope = rope

        if self.rope:
            dec_layer.self_attn = MHA(embed_dim, num_heads, rope=self.rope)
        else:
            self.pos_embeds = nn.Embedding(max_len, embed_dim)

        self.decoder = nn.TransformerDecoder(dec_layer, num_layers, norm=nn.LayerNorm(embed_dim))

    def forward(
        self,
        x: Float32[Tensor, "B N C"],
        padding_mask: Float32[Tensor, "B N"],
        enc_out: Float32[Tensor, "B N_enc C_enc"],
    ) -> Float32[Tensor, "B N C"]:
        _, N, _ = x.shape

        if not self.rope:
            pos_embeds = self.pos_embeds(torch.arange(N, device=x.device))
            x += pos_embeds

        causal_mask = nn.Transformer.generate_square_subsequent_mask(N, device=x.device)
        x = self.decoder.forward(
            x, enc_out, tgt_mask=causal_mask, tgt_key_padding_mask=padding_mask, tgt_is_causal=True
        )

        return x
