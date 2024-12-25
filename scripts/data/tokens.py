from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypedDict

import torch
import torch.nn.functional as F
from jaxtyping import Float32, Int32
from torch import Tensor


class TokenCls(Enum):
    PAD = 0
    EOS = 1
    COO = 2


@dataclass
class Token:
    cls: TokenCls
    coord: Float32[Tensor, "2"]

    @staticmethod
    def eos() -> "Token":
        return Token(TokenCls.EOS, torch.tensor([0.0, 0.0], dtype=torch.float32))

    @staticmethod
    def pad() -> "Token":
        return Token(TokenCls.PAD, torch.tensor([0.0, 0.0], dtype=torch.float32))

    @staticmethod
    def coo(coord: Float32[Tensor, "2"] | tuple[float, float]) -> "Token":
        return Token(TokenCls.COO, torch.tensor(coord, dtype=torch.float32))


class TokenSequence(TypedDict):
    cls: Int32[Tensor, " N"]
    coord: Float32[Tensor, "N 2"]


def pack_tokens(tokens: list[Token]) -> TokenSequence:
    cls = torch.tensor([token.cls.value for token in tokens], dtype=torch.int32)
    coord = torch.stack([token.coord for token in tokens])
    return TokenSequence(cls, coord)


class TokenBatch(TypedDict):
    cls: Int32[Tensor, "B N"]
    coord: Float32[Tensor, "B N 2"]
    padding_mask: Float32[Tensor, "B N"]


def pack_token_sequences(sequences: list[TokenSequence] | list[list[Token]]) -> TokenBatch:
    if isinstance(sequences[0], list):
        sequences = [TokenSequence.pack_tokens(tokens) for tokens in sequences]

    max_seq_len = max(len(seq.cls) for seq in sequences)
    out_cls: list[Int32[Tensor, " N"]] = []
    out_coord: list[Float32[Tensor, "N 2"]] = []

    for seq in sequences:
        num_padded = max_seq_len - len(seq.cls)
        cls = F.pad(seq.cls, (0, num_padded))
        coord = F.pad(seq.coord, (0, 0, 0, num_padded))
        out_cls.append(cls)
        out_coord.append(coord)

    cls = torch.stack(out_cls)
    coord = torch.stack(out_coord)
    padding_mask = (cls == TokenCls.PAD.value).float() * -torch.inf

    return TokenBatch(cls, coord, padding_mask)
