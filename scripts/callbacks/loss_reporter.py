from typing import cast

import clearml
from data import batch_types
from lightning import Callback, LightningModule, Trainer
from lightning_system import LSOutput
from torch import Tensor


class LossReporter(Callback):
    def __init__(self) -> None:
        super().__init__()

        self._task: clearml.Task | None = clearml.Task.current_task()

    def _on_batch_end(self, trainer: Trainer, outputs: LSOutput, title: str) -> None:
        if self._task is None:
            return

        for k, v in outputs["losses"].items():
            self._task.logger.report_scalar(
                title=title, series=k, value=cast(Tensor, v).item(), iteration=trainer.global_step
            )

    def on_train_batch_end(  # type: ignore
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: LSOutput,
        batch: batch_types.Batch,
        batch_idx: int,
    ) -> None:
        self._on_batch_end(trainer, outputs, "train")

    def on_validation_batch_end(  # type: ignore
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: LSOutput,
        batch: batch_types.Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._on_batch_end(trainer, outputs, "valid")

    def on_test_batch_end(self, *args, **kwargs) -> None:
        self.on_validation_epoch_end(*args, **kwargs)
