from typing import TypedDict

import torch
from data import batch_types
from jaxtyping import Float32
from lightning import LightningModule
from losses import LossOutput, TotalLoss
from model.model import Model
from torch import Tensor


class LSConfig(TypedDict):
    cls_coef: float
    coord_coef: float
    initial_lr: float
    compile_model: bool


class LSOutput(TypedDict):
    losses: LossOutput
    loss: Float32[Tensor, ""]


class LightningSystem(LightningModule):
    def __init__(self, config: LSConfig):
        super(LightningSystem, self).__init__()

        self.loss = TotalLoss(config["cls_coef"], config["coord_coef"])
        self.model = Model(compile_layers=config["compile_model"])
        self.config = config

    def configure_optimizers(self):
        # TODO: Cosine or poly learning rate decay?
        return torch.optim.AdamW(self.model.parameters(), lr=self.config["initial_lr"])

    def _shared_step(self, batch: batch_types.Batch) -> LossOutput:
        pred = self.model.forward(batch["model_input"]["coords"], batch["model_input"]["images"])
        loss = self.loss.forward(pred, batch["target"])

        return loss

    def training_step(self, batch: batch_types.Batch, batch_idx: int) -> LSOutput:
        loss = self._shared_step(batch)
        out: LSOutput = {"losses": loss, "loss": loss["total"]}

        return out

    def validation_step(self, batch: batch_types.Batch, batch_idx: int) -> LSOutput:
        loss = self._shared_step(batch)
        out: LSOutput = {"losses": loss, "loss": loss["total"]}
        self.log("valid_loss", loss["total"])  # To use with the checkpointing callback

        return out

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)
