import logging

import shapely.geometry as sg
import torch
import torchmetrics as tm
import utils.pano as pano
from data import tokens
from jaxtyping import Shaped
from torch import Tensor

log = logging.getLogger(__name__)
camera_height = 1.6 #height in meters, an approximation

def topdown_iou(
    pred_object: tokens.TokenSequence, target_object: tokens.TokenSequence, device: torch.device | None = None
) -> torch.Tensor:
    pred_coords_uv = tokens.get_seq_coordinates(pred_object).cpu().numpy()
    target_coords_uv = tokens.get_seq_coordinates(target_object).cpu().numpy()
    pred_coords = pano.uv_to_topdown(camera_height, pred_coords_uv)
    target_coords = pano.uv_to_topdown(camera_height, target_coords_uv)

    if device is None:
        device = torch.get_default_device()

    if pred_coords.shape[0] < 3 or target_coords.shape[0] < 3:
        return torch.tensor(0.0, dtype=torch.float32, device=device)

    try:
        pred_poly = sg.Polygon(pred_coords)
        target_poly = sg.Polygon(target_coords)

        if not pred_poly.is_valid:
            pred_poly = pred_poly.buffer(0)

        if not target_poly.is_valid:
            target_poly = target_poly.buffer(0)
    except Exception as e:
        log.error(e)
        return torch.tensor(0.0, dtype=torch.float32, device=device)

    intersection = pred_poly.intersection(target_poly).area
    union = pred_poly.union(target_poly).area
    iou = intersection / union

    return torch.tensor(iou, dtype=torch.float32, device=device)


def ensure_tensor_state(
    state_val: list[Shaped[Tensor, "*Dims"]] | Shaped[Tensor, " N *Dims"],
) -> Shaped[Tensor, " N *Dims"]:
    """
    A TorchMetrics cat state will be a list when doing .compute() when not using DDP,
    but a tensor when using DDP. This function ensures that the state is a tensor.
    ffs torchmetrics git gud.

    Parameters
    ----------
    state_val
        The state value to ensure is a tensor. If it's a list, it will be stacked.

    Returns
    -------
    The state value as a tensor. Shape: `(N, *Dims)` if `state_val` was a list, otherwise `(*Dims)`.
    """

    if isinstance(state_val, list):
        return torch.stack(state_val)

    return state_val


class IoU(tm.Metric):
    def __init__(self, name: str, **kwargs):
        super().__init__(**kwargs)
        self.name = name

        self.add_state("iou", default=[], dist_reduce_fx="cat")
        self.iou: list[Tensor]

    def update(self, preds: tokens.TokenSequence, targets: tokens.TokenSequence):
        iou = topdown_iou(preds, targets, device=preds["cls"].device)

        self.iou.append(iou)

    def compute(self) -> dict[str, dict[str, Tensor]]:
        iou = ensure_tensor_state(self.iou)
        result = {"IoU": {"Topdown IoU": iou.mean()}}
        return result
