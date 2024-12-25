from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypedDict, cast

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
    def eos(device: torch.device | None = None) -> "Token":
        return Token(TokenCls.EOS, torch.tensor([0.0, 0.0], dtype=torch.float32, device=device))

    @staticmethod
    def pad(device: torch.device | None = None) -> "Token":
        return Token(TokenCls.PAD, torch.tensor([0.0, 0.0], dtype=torch.float32, device=device))

    @staticmethod
    def coo(coord: Float32[Tensor, "2"] | tuple[float, float], device: torch.device | None = None) -> "Token":
        if isinstance(coord, tuple):
            coord = torch.tensor(coord, dtype=torch.float32, device=device)

        return Token(TokenCls.COO, coord)


class TokenSequence(TypedDict):
    cls: Int32[Tensor, " N"]
    coord: Float32[Tensor, "N 2"]


def pack_tokens(tokens: list[Token]) -> TokenSequence:
    if len(tokens) == 0:
        raise ValueError("Cannot pack empty token sequence")

    device = tokens[0].coord.device
    cls = torch.tensor([token.cls.value for token in tokens], dtype=torch.int32, device=device)
    coord = torch.stack([token.coord for token in tokens])
    seq: TokenSequence = {
        "cls": cls,
        "coord": coord,
    }

    return seq


class TokenBatch(TypedDict):
    cls: Int32[Tensor, "B N"]
    coord: Float32[Tensor, "B N 2"]
    padding_mask: Float32[Tensor, "B N"]


def pack_token_sequences(sequences: list[TokenSequence] | list[list[Token]]) -> TokenBatch:
    if isinstance(sequences[0], list):
        sequences = cast(list[list[Token]], sequences)
        sequences = [pack_tokens(tokens) for tokens in sequences]

    sequences = cast(list[TokenSequence], sequences)
    max_seq_len = max(len(seq["cls"]) for seq in sequences)
    out_cls: list[Int32[Tensor, " N"]] = []
    out_coord: list[Float32[Tensor, "N 2"]] = []

    for seq in sequences:
        num_padded = max_seq_len - len(seq["cls"])
        cls = F.pad(seq["cls"], (0, num_padded))
        coord = F.pad(seq["coord"], (0, 0, 0, num_padded))
        out_cls.append(cls)
        out_coord.append(coord)

    cls = torch.stack(out_cls)
    coord = torch.stack(out_coord)
    padding_mask = torch.where(cls == TokenCls.PAD.value, -torch.inf, 0.0)
    batch: TokenBatch = {
        "cls": cls,
        "coord": coord,
        "padding_mask": padding_mask,
    }

    return batch
