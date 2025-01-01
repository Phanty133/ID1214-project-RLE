import clearml
import config
import lightning
import torch
from callbacks import image_reporter, loss_reporter, lr_reporter, metric_reporter
from data.data_module import DataModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from lightning_system import LightningSystem
from utils import clearml_checkpoint_io


def train():
    clearml.Task.init(
        project_name=config.CLEARML_PROJECT_NAME,
        task_name=config.CLEARML_TASK_NAME,
        reuse_last_task_id=False,
        auto_connect_frameworks=False,
        output_uri=True,
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
            metric_reporter.MetricReporter(),
            ModelCheckpoint(monitor="valid_loss", mode="min", save_top_k=1, save_last=True, filename="best"),
        ],
        strategy=DDPStrategy(checkpoint_io=clearml_checkpoint_io.ClearMLCheckpointIO()),
        # overfit_batches=0.01,
    )
    trainer.fit(ls, dm)


if __name__ == "__main__":
    train()
