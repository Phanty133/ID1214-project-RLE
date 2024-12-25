import torch.nn as nn
from data.tokens import TokenCls
from jaxtyping import Float32
from torch import Tensor


class ClassHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.head = nn.Linear(in_channels, self.num_classes)

    @property
    def num_classes() -> int:
        return len(TokenCls)

    @staticmethod
    def get_classes(x: Float32[Tensor, "B N C"]) -> Float32[Tensor, "B N"]:
        return x.argmax(dim=-1)

    def forward(self, x: Float32[Tensor, "B N C"]) -> Float32[Tensor, "B N C"]:
        return self.head.forward(x)


class CoordHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.head = nn.Linear(in_channels, 2)

    def forward(self, x: Float32[Tensor, "B N C"]) -> Float32[Tensor, "B N 2"]:
        return self.head.forward(x)
