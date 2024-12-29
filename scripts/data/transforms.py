import functools
from typing import TypedDict

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as Fvision
from jaxtyping import Float, Float32, UInt8
from torch import Tensor
from utils.pano import poly_2d_to_3d, poly_3d_to_2d


class Augmentations(TypedDict):
    flip: bool
    roll: float
    hue_jitter: float
    saturation_jitter: float
    brightness_jitter: float
    contrast_jitter: float
    stretch_kx: float
    stretch_kz: float


def gen_augmentation_values() -> Augmentations:
    augms: Augmentations = {
        "flip": bool(torch.rand(1).item() > 0.5),
        "roll": float(torch.rand(1).item()),
        "hue_jitter": float((torch.rand(1).item() - 0.5) * 0.2),
        "saturation_jitter": float(1 + (torch.rand(1).item() - 0.5) * 0.2),
        "brightness_jitter": float(1 + (torch.rand(1).item() - 0.5) * 0.2),
        "contrast_jitter": float(1 + (torch.rand(1).item() - 0.5) * 0.2),
        "stretch_kx": 1,
        "stretch_kz": 1,
    }

    MAX_STRETCH = 1
    augms["stretch_kx"] += float((torch.rand(1) * MAX_STRETCH).item())
    augms["stretch_kz"] += float((torch.rand(1) * MAX_STRETCH).item())

    if torch.rand(1) < 0.5:
        augms["stretch_kx"] = 1 / augms["stretch_kx"]

    if torch.rand(1) < 0.5:
        augms["stretch_kz"] = 1 / augms["stretch_kz"]

    return augms


@functools.lru_cache()
def _gen_stretch_angles_caching(bs: int, width: int, height: int, device: torch.device):
    """
    Pregenerate trig values for the stretch function. The values are cached.

    Parameters
    ----------
    bs
        Batch size.
    width
        Width of the input image.
    height
        Height of the input image.

    Returns
    -------
    `(sin(longitude), cos(longitude), tan(latitude))`
    """

    # This step thing is what they do in the DOPNet code. It appears to help with
    # the boundary cols/rows of the image.
    half_w_step = torch.pi / width
    half_h_step = 0.5 * torch.pi / height
    lon_prime = torch.linspace(-torch.pi + half_w_step, torch.pi - half_w_step, width, device=device)
    lat_prime = torch.linspace(-torch.pi / 2 + half_h_step, torch.pi / 2 - half_h_step, height, device=device)

    sin_lon_prime = torch.sin(lon_prime)
    cos_lon_prime = torch.cos(lon_prime)
    tan_lat_prime = torch.tan(lat_prime)

    # The values need to be tiled to compute a batch of images
    sin_lon_prime = torch.tile(sin_lon_prime[None, :], (bs, 1))
    cos_lon_prime = torch.tile(cos_lon_prime[None, :], (bs, 1))
    tan_lat_prime = torch.tile(tan_lat_prime[None, :], (bs, 1))

    return sin_lon_prime, cos_lon_prime, tan_lat_prime


def gen_pano_stretch_map(
    width: int,
    height: int,
    kx: Float[Tensor, " B"],
    kz: Float[Tensor, " B"],
    device: torch.device,
) -> Float[Tensor, "B H W 2"]:
    """
    Generates the flow fields for use with `torch.nn.functional.grid_sample()` to
    stretch panoramas in a batch along the X and Z axes. Based on code from the
    DOPNet model repo. The augmentation idea itself is proposed in the HorizonNet
    paper.

    Parameters
    ----------
    width
        Width of the input image.
    height
        Height of the input image.
    kx
        Stretch factor along the X axis.
    kz
        Stretch factor along the Z axis.

    Returns
    -------
    The flow field for use with grid_sample.
    """

    B = kx.shape[0]
    sin_lon_prime, cos_lon_prime, tan_lat_prime = _gen_stretch_angles_caching(B, width, height, device)

    lon = torch.atan2(torch.einsum("bi, b -> bi", sin_lon_prime, kz / kx), cos_lon_prime)
    lat = torch.arctan(torch.einsum("bi, bj, b -> bij", tan_lat_prime, torch.sin(lon) / sin_lon_prime, kx))

    # Scale the maps to [-1; 1] for grid_sample
    lon = lon / torch.pi
    lat = lat / (torch.pi / 2)

    sample_map = torch.empty((B, height, width, 2), dtype=torch.float32, device=device)
    sample_map[..., 0] = lon.unsqueeze(1)
    sample_map[..., 1] = lat

    return sample_map


def stretch_pano(
    img: UInt8[Tensor, "B C H W"],
    kx: Float[Tensor, " B"],
    kz: Float[Tensor, " B"],
) -> UInt8[Tensor, "B C H W"]:
    """
    Stretches a batch of panoramas along the X and Z axes.

    Parameters
    ----------
    img
        The input image.
    kx
        Stretch factor along the X axis.
    kz
        Stretch factor along the Z axis.

    Returns
    -------
    The stretched image.
    """

    B, C, H, W = img.shape
    sample_map = gen_pano_stretch_map(W, H, kx, kz, img.device)
    stretched_img = F.grid_sample(
        img, sample_map, padding_mode="reflection", align_corners=False, mode="bilinear"
    )

    return stretched_img


def stretch_contour(points: Float32[Tensor, "N 2"], kx: float, kz: float) -> Float32[Tensor, "N 2"]:
    """
    Stretch the contour along the x and y axes according to the HorizonNet
    paper.
    """

    points_m = poly_2d_to_3d(points.numpy())
    points_m[:, 0] *= kx
    points_m[:, 1] *= kz  # Scale y with kz because for us Z is the vertical axis
    points_stretched = torch.from_numpy(poly_3d_to_2d(points_m))

    return points_stretched


def apply_pano_augmentations(
    image: Float32[Tensor, "C H W"], augmentations: Augmentations
) -> Float32[Tensor, "C H W"]:
    image = stretch_pano(
        image.unsqueeze(0),
        torch.tensor([augmentations["stretch_kx"]]),
        torch.tensor([augmentations["stretch_kz"]]),
    ).squeeze(0)

    if augmentations["flip"]:
        image = torch.flip(image, dims=(2,))

    image = torch.roll(image, shifts=int(augmentations["roll"] * image.shape[1]), dims=2)
    image = Fvision.adjust_hue(image, augmentations["hue_jitter"])
    image = Fvision.adjust_saturation(image, augmentations["saturation_jitter"])
    image = Fvision.adjust_brightness(image, augmentations["brightness_jitter"])
    image = Fvision.adjust_contrast(image, augmentations["contrast_jitter"])

    return image


def apply_layout_augmentations(
    layout: Float32[Tensor, "N 2"], augmentations: Augmentations
) -> Float32[Tensor, "N 2"]:
    layout = stretch_contour(layout, augmentations["stretch_kx"], augmentations["stretch_kz"])

    if augmentations["flip"]:
        layout = torch.flip(layout, dims=(0,))
        layout[:, 0] = 1 - layout[:, 0]

    layout[:, 0] = (layout[:, 0] + augmentations["roll"]) % 1

    return layout
