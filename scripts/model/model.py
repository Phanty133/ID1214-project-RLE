from typing import TypedDict, cast

import torch
import torch.nn as nn
from data import tokens
from jaxtyping import Float32
from model.decoder import Decoder
from model.encoder import Encoder
from model.heads import ClassHead, CoordHead
from model.token_embeddings import TokenEmbeddings
from torch import Tensor


class ModelOutput(TypedDict):
    cls: Float32[Tensor, "B N"]
    coord: Float32[Tensor, "B N 2"]


class ModelHeads(TypedDict):
    cls: ClassHead
    coord: CoordHead


class Model(nn.Module):
    def __init__(self, embed_size=768, num_layers=6, num_heads=12, max_len=64, compile_layers=False):
        super(Model, self).__init__()

        self.embed_size = embed_size
        self.encoder = Encoder(embed_size)
        self.decoder = Decoder(embed_size, num_layers, num_heads, max_len)
        self.embeds = TokenEmbeddings(embed_size)
        self.heads = cast(
            ModelHeads, nn.ModuleDict({"cls": ClassHead(embed_size), "coord": CoordHead(embed_size)})
        )

        if compile_layers:
            self.embeds = cast(TokenEmbeddings, torch.compile(self.embeds, fullgraph=True))
            self.heads["cls"] = cast(ClassHead, torch.compile(self.heads["cls"], fullgraph=True))
            self.heads["coord"] = cast(CoordHead, torch.compile(self.heads["coord"], fullgraph=True))
            self.encoder = cast(Encoder, torch.compile(self.encoder, fullgraph=True))
            # self.decoder = torch.compile(self.decoder, fullgraph=True) # TODO: might be a bit finnicky

    def forward_encoder(self, image: Float32[Tensor, "B C H W"]) -> Float32[Tensor, "B N_enc C_enc"]:
        return self.encoder.forward(image)

    def forward_decoder(
        self, tokens: tokens.TokenBatch, enc_out: Float32[Tensor, "B N_enc C_enc"]
    ) -> ModelOutput:
        embeds = self.embeds.forward(tokens)
        dec_out = self.decoder.forward(embeds, tokens["padding_mask"], enc_out)
        cls = self.heads["cls"].forward(dec_out)
        coord = self.heads["coord"].forward(dec_out)

        return {"cls": cls, "coord": coord}

    def forward(self, tokens: tokens.TokenBatch, image: Float32[Tensor, "B C H W"]) -> ModelOutput:
        enc_out = self.forward_encoder(image)
        dec_out = self.forward_decoder(tokens, enc_out)

        return dec_out
