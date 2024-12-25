from typing import TypedDict

import torch
import torchvision.transforms.functional as F
from jaxtyping import Float32
from torch import Tensor


class Augmentations(TypedDict):
    flip: bool
    roll: float
    hue_jitter: float
    saturation_jitter: float
    brightness_jitter: float
    contrast_jitter: float


def gen_augmentation_values() -> Augmentations:
    augmentations: Augmentations = {
        "flip": bool(torch.rand(1).item() > 0.5),
        "roll": float(torch.rand(1).item()),
        "hue_jitter": float((torch.rand(1).item() - 0.5) * 0.2),
        "saturation_jitter": float(1 + (torch.rand(1).item() - 0.5) * 0.2),
        "brightness_jitter": float(1 + (torch.rand(1).item() - 0.5) * 0.2),
        "contrast_jitter": float(1 + (torch.rand(1).item() - 0.5) * 0.2),
    }

    return augmentations


def apply_pano_augmentations(
    image: Float32[Tensor, "C H W"], augmentations: Augmentations, normalize: bool = True
) -> Float32[Tensor, "C H W"]:
    if augmentations["flip"]:
        image = torch.flip(image, dims=(2,))

    image = torch.roll(image, shifts=int(augmentations["roll"] * image.shape[1]), dims=2)
    image = F.adjust_hue(image, augmentations["hue_jitter"])
    image = F.adjust_saturation(image, augmentations["saturation_jitter"])
    image = F.adjust_brightness(image, augmentations["brightness_jitter"])
    image = F.adjust_contrast(image, augmentations["contrast_jitter"])

    if normalize:
        # ImageNet normalization
        F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)

    return image


def apply_layout_augmentations(
    layout: Float32[Tensor, "N 2"], augmentations: Augmentations
) -> Float32[Tensor, "N 2"]:
    if augmentations["flip"]:
        layout = torch.flip(layout, dims=(0,))
        layout[:, 0] = 1 - layout[:, 0]

    layout[:, 0] = (layout[:, 0] + augmentations["roll"]) % 1

    return layout
