import config
import lightning
import torch
from data.data_module import DataModule
from lightning_system import LightningSystem


def train():
    torch.set_float32_matmul_precision("medium")
    ls = LightningSystem(config.LS_CONFIG)
    dm = DataModule()
    trainer = lightning.Trainer(
        max_epochs=config.MAX_EPOCHS,
        devices=config.NUM_GPUS,
        precision="bf16-mixed",
        log_every_n_steps=25,
        accumulate_grad_batches=config.GRAD_ACC,
        num_nodes=1,
    )
    trainer.fit(ls, dm)


if __name__ == "__main__":
    train()
