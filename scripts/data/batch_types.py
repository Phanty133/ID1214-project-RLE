from typing import Any, Mapping, TypedDict

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
    metadata: Mapping[str, Any]
    image: Float32[np.ndarray, "H W C"]


class Batch(TypedDict):
    idx: list[str]
    model_input: ModelInputBatch
    target: tokens.TokenBatch
    metadata: list[Mapping[str, Any]]
    images: list[Float32[np.ndarray, "H W C"]]


def split_batch(batch: Batch) -> list[Sample]:
    out: list[Sample] = []

    for idx, sample_idx in enumerate(batch["idx"]):
        input_seq: tokens.TokenSequence = {
            "cls": batch["model_input"]["coords"]["cls"][idx],
            "coord": batch["model_input"]["coords"]["coord"][idx],
        }
        target_seq: tokens.TokenSequence = {
            "cls": batch["target"]["cls"][idx],
            "coord": batch["target"]["coord"][idx],
        }

        out.append(
            {
                "idx": sample_idx,
                "model_input": {
                    "image": batch["model_input"]["images"][idx],
                    "coords": input_seq,
                },
                "target": target_seq,
                "metadata": batch["metadata"][idx],
                "image": batch["images"][idx],
            }
        )

    return out
