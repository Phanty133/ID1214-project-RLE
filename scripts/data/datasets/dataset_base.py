from abc import abstractmethod
from pathlib import Path

import torch
from data import batch_types, tokens
from torch.utils.data import Dataset


class DatasetBase(Dataset[batch_types.Sample]):
    def __init__(self, ds_root_dir: Path, *args, **kwargs): ...

    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> batch_types.Sample: ...

    @staticmethod
    def collate_fn(samples: list[batch_types.Sample]) -> batch_types.Batch:
        images_torch = torch.stack([sample["model_input"]["image"] for sample in samples])
        images_np = [sample["image"] for sample in samples]
        input_coords = tokens.pack_token_sequences([sample["model_input"]["coords"] for sample in samples])
        target_coords = tokens.pack_token_sequences([sample["target"] for sample in samples])

        return {
            "idx": [sample["idx"] for sample in samples],
            "model_input": {
                "images": images_torch,
                "coords": input_coords,
            },
            "target": target_coords,
            "metadata": [sample["metadata"] for sample in samples],
            "images": images_np,
        }
