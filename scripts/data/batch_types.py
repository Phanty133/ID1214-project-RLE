from typing import Any, TypedDict

import numpy as np
from data import tokens
from jaxtyping import Float32
from torch import Tensor


class ModelInputSample(TypedDict):
    image: Float32[Tensor, "C H W"]
    coords: tokens.TokenSequence


class ModelInputBatch(TypedDict):
    images: Float32[Tensor, "B C H W"]
    coords: tokens.TokenBatch


class Sample(TypedDict):
    idx: str
    model_input: ModelInputSample
    target: tokens.TokenSequence
    metadata: dict[str, Any]
    image: Float32[np.ndarray, "H W C"]


class Batch(TypedDict):
    idx: list[str]
    model_input: ModelInputBatch
    target: tokens.TokenBatch
    metadata: list[dict[str, Any]]
    images: list[Float32[np.ndarray, "H W C"]]
