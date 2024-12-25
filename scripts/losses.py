from typing import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from data import tokens
from jaxtyping import Float32, Int32
from model import model
from torch import Tensor


class LossOutput(TypedDict):
    cls: Float32[Tensor, ""]
    coord: Float32[Tensor, ""]
    total: Float32[Tensor, ""]


class CoordLoss(nn.Module):
    def __init__(self):
        super(CoordLoss, self).__init__()

    def forward(
        self,
        pred: Float32[Tensor, "B N 2"],
        target: Float32[Tensor, "B N 2"],
        target_cls: Int32[Tensor, " B"],
    ) -> Float32[Tensor, ""]:
        coord_mask = target_cls == tokens.TokenCls.COO.value
        masked_pred = pred[coord_mask]
        masked_target = target[coord_mask]
        l1_loss = F.l1_loss(masked_pred, masked_target, reduction="mean")

        return l1_loss


class ClsLoss(nn.Module):
    def __init__(self):
        super(ClsLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=tokens.TokenCls.PAD.value)

    def forward(self, pred: Float32[Tensor, "B N C"], target: Int32[Tensor, "B N"]) -> Float32[Tensor, ""]:
        return self.loss.forward(pred.reshape((-1, pred.shape[-1])), target.reshape((-1,)).to(torch.int64))


class TotalLoss(nn.Module):
    def __init__(self, cls_coef: float, coord_coef: float):
        super(TotalLoss, self).__init__()
        self.cls_loss = ClsLoss()
        self.coord_loss = CoordLoss()
        self.cls_coef = cls_coef
        self.coord_coef = coord_coef

    def forward(self, pred: model.ModelOutput, target: tokens.TokenBatch) -> LossOutput:
        cls_loss = self.cls_loss.forward(pred["cls"], target["cls"])
        coord_loss = self.coord_loss.forward(pred["coord"], target["coord"], target["cls"])

        cls_scaled = cls_loss * self.cls_coef
        coord_scaled = coord_loss * self.coord_coef
        total_loss = cls_scaled + coord_scaled
        losses: LossOutput = {
            "cls": cls_scaled,
            "coord": coord_scaled,
            "total": total_loss,
        }

        return losses


# TODO: Differentiable IoU loss
