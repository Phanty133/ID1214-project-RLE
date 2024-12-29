import logging

import clearml
import lightning_system as ls
import numpy as np
import torch
import torchmetrics as tm
from data import batch_types, tokens
from lightning import Callback, LightningModule, Trainer
from metrics import iou

log = logging.getLogger(__name__)


class MetricReporter(Callback):
    def __init__(self):
        super().__init__()
        self.metrics = [iou.IoU("iou")]
        self.task: clearml.Task | None = clearml.Task.current_task()

        if self.task is None:
            log.warning("No ClearML task found")

    def skip_step(self, trainer: Trainer, outputs: ls.LSOutput | None, epoch_end=False):
        return (self.task is None) or (outputs is None and not epoch_end)

    def update_metrics(self, batch: batch_types.Batch, outputs: ls.LSOutput):
        samples = batch_types.split_batch(batch)
        preds = tokens.split_token_batch(outputs["pred"])

        for sample, pred in zip(samples, preds, strict=True):
            for metric in self.metrics:
                metric.update(pred, sample["target"])

    def on_validation_batch_end(  # type: ignore
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: ls.LSOutput,
        batch: batch_types.Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        if self.skip_step(trainer, outputs):
            return

        self.update_metrics(batch, outputs)

    def on_test_batch_end(  # type: ignore
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: ls.LSOutput,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def log_metric_scalars(self, metric: tm.Metric, index: int, trainer: Trainer) -> None:
        result = metric.compute()

        if self.task is None or trainer.state.stage == "sanity_check":
            return

        for t, subresult in result.items():
            for s, val in subresult.items():
                if isinstance(val, torch.Tensor):
                    val = val.item()

                if np.isnan(val) or np.isinf(val):
                    return

                self.task.get_logger().report_scalar(
                    title=t,
                    series=s,
                    value=val,
                    iteration=index,
                )

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.skip_step(trainer, None, epoch_end=True):
            return

        for metric in self.metrics:
            self.log_metric_scalars(metric, pl_module.current_epoch, trainer)
            metric.reset()

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.on_validation_epoch_end(trainer, pl_module)
