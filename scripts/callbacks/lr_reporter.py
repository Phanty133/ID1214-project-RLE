import clearml
from lightning import Callback, LightningModule, Trainer


class LRReporter(Callback):
    def __init__(self) -> None:
        super().__init__()

        self._task: clearml.Task | None = clearml.Task.current_task()

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._task is None:
            return

        lr = pl_module.trainer.optimizers[0].param_groups[0]["lr"]
        self._task.logger.report_scalar("LR", "LR", lr, iteration=trainer.current_epoch)
