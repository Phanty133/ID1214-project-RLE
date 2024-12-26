import clearml
import config
import lightning
import torch
from data.data_module import DataModule
from lightning_system import LightningSystem

from scripts.callbacks import image_reporter, loss_reporter, lr_reporter


def train():
    clearml.Task.init(
        project_name=config.CLEARML_PROJECT_NAME,
        task_name=config.CLEARML_TASK_NAME,
        reuse_last_task_id=False,
        auto_connect_frameworks=False,
    )

    torch.set_float32_matmul_precision("medium")
    ls = LightningSystem(config.LS_CONFIG)
    dm = DataModule()

    trainer = lightning.Trainer(
        max_epochs=config.MAX_EPOCHS,
        devices=config.NUM_GPUS,
        precision="bf16-mixed",
        log_every_n_steps=1,
        accumulate_grad_batches=config.GRAD_ACC,
        callbacks=[
            image_reporter.ImageReporter(max_samples=50),
            loss_reporter.LossReporter(),
            lr_reporter.LRReporter(),
        ],
    )
    trainer.fit(ls, dm)


if __name__ == "__main__":
    train()
