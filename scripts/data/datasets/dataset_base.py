from abc import abstractmethod
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from data import batch_types, tokens, transforms
from jaxtyping import Float32
from torch import Tensor
from torch.utils.data import Dataset


class DatasetBase(Dataset[batch_types.Sample]):
    def __init__(
        self,
        ds_root_dir: Path,
        image_size_wh: tuple[int, int],
        apply_augms: bool,
        split: Literal["train", "val"],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ds_root_dir = ds_root_dir
        self.image_size_wh = image_size_wh
        self.apply_augms = apply_augms
        self.split = split

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

    def _process_sample(
        self, idx: str, gt_layout: Float32[Tensor, "N 2"], image: Float32[np.ndarray, "H W 3"]
    ) -> batch_types.Sample:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size_wh)
        torch_image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if self.apply_augms:
            augmentations = transforms.gen_augmentation_values()
        else:
            augmentations = None

        if augmentations is not None:
            image = (
                transforms.apply_pano_augmentations(torch_image.clone(), augmentations)
                .permute(1, 2, 0)
                .numpy()
            ) * 255

        if augmentations is not None:
            torch_image = transforms.apply_pano_augmentations(torch_image, augmentations)
            gt_layout = transforms.apply_layout_augmentations(gt_layout, augmentations)

        F.normalize(torch_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)
        lowest_x_idx = torch.argmin(gt_layout[:, 0])
        gt_layout = torch.cat([gt_layout[lowest_x_idx:], gt_layout[:lowest_x_idx]])

        layout_tokens = [tokens.Token.coo(coord) for coord in gt_layout]
        input_seq = [tokens.Token.eos()] + layout_tokens
        gt_seq = layout_tokens + [tokens.Token.eos()]
        input_seq_torch = tokens.pack_tokens(input_seq)
        gt_seq_torch = tokens.pack_tokens(gt_seq)

        return {
            "idx": idx,
            "model_input": {
                "image": torch_image,
                "coords": input_seq_torch,
            },
            "target": gt_seq_torch,
            "metadata": augmentations or {},
            "image": image,
        }
