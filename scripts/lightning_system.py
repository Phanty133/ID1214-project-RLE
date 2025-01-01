from typing import NotRequired, TypedDict

import torch
from data import batch_types, tokens
from jaxtyping import Float32
from lightning import LightningModule
from losses import LossOutput, TotalLoss
from model.heads import ClassHead
from model.model import Model, ModelOutput
from torch import Tensor


class LSConfig(TypedDict):
    cls_coef: float
    coord_coef: float
    initial_lr: float
    decay_epochs: NotRequired[int]
    lr_min: NotRequired[float]
    cos_annealing: bool
    compile_model: bool


class LSOutput(TypedDict):
    losses: LossOutput
    loss: Float32[Tensor, ""]
    pred: tokens.TokenBatch


class LightningSystem(LightningModule):
    def __init__(self, config: LSConfig):
        super(LightningSystem, self).__init__()

        self.loss = TotalLoss(config["cls_coef"], config["coord_coef"])
        self.model = Model(compile_layers=config["compile_model"])
        self.config = config

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["initial_lr"])

        if not self.config.get("cos_annealing", False):
            return optimizer

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.get("decay_epochs", 10), eta_min=self.config.get("lr_min", 0)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                scheduler,
                torch.optim.lr_scheduler.ConstantLR(
                    optimizer,
                    factor=self.config.get("lr_min", 0) / self.config["initial_lr"],
                    total_iters=10000,
                ),
            ],
            milestones=[self.config.get("decay_epochs", 10)],
        )
        return [optimizer], [scheduler]

    def _shared_step(self, batch: batch_types.Batch) -> tuple[LossOutput, ModelOutput]:
        pred = self.model.forward(batch["model_input"]["coords"], batch["model_input"]["images"])
        loss = self.loss.forward(pred, batch["target"])
        return loss, pred

    def training_step(self, batch: batch_types.Batch, batch_idx: int) -> LSOutput:
        loss, model_output = self._shared_step(batch)
        pred_cls = ClassHead.get_classes(model_output["cls"])
        pred: tokens.TokenBatch = {
            "cls": pred_cls,
            "coord": model_output["coord"],
            "padding_mask": batch["model_input"]["coords"]["padding_mask"],
        }

        out: LSOutput = {"losses": loss, "loss": loss["total"], "pred": pred}
        self.log("train_loss", loss["total"], batch_size=len(batch["idx"]), prog_bar=True)

        return out

    def validation_step(self, batch: batch_types.Batch, batch_idx: int) -> LSOutput:
        loss, _ = self._shared_step(batch)
        inf_output = self.model.inference(batch["model_input"]["images"])
        out: LSOutput = {"losses": loss, "loss": loss["total"], "pred": inf_output}
        self.log("valid_loss", loss["total"], batch_size=len(batch["idx"]), prog_bar=True, sync_dist=True)

        return out

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)
