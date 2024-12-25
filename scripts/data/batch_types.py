from typing import Any, TypedDict

import tokens
from jaxtyping import Float32, Int32
from torch import Tensor


class ModelInputSample(TypedDict):
    image: Float32[Tensor, "C H W"]
    coords: tokens.TokenSequence


class ModelInputBatch(TypedDict):
    images: Float32[Tensor, "B C H W"]
    coords: tokens.TokenBatch


class Batch(TypedDict):
    idx: Int32[Tensor, " B"]
    model_input: ModelInputBatch
    target: tokens.TokenBatch
    metadata: dict[str, Any]
    image: Float32[Tensor, "B C H W"]
