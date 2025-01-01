import config
import rich
from data import batch_types
from data.datasets.dataset_base import DatasetBase
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from scripts.data.datasets.panocontext.dataset import PanoContextDataset


class DataModule(LightningDataModule):
    def __init__(self) -> None:
        super(DataModule, self).__init__()

        self.train_dataset: DatasetBase | None = None
        self.valid_dataset: DatasetBase | None = None

    def prepare_data(self) -> None:
        if isinstance(self.train_dataset, PanoContextDataset):
            PanoContextDataset(config.DATASET_DIR, config.IMAGE_HW, split="train", apply_augms=True)

        if isinstance(self.valid_dataset, PanoContextDataset):
            PanoContextDataset(config.DATASET_DIR, config.IMAGE_HW, split="val", apply_augms=False)

    def setup(self, stage: str) -> None:
        self.train_dataset = config.TRAIN_DATASET(
            config.DATASET_DIR, config.IMAGE_HW, apply_augms=True, split="train"
        )
        self.valid_dataset = config.VALID_DATASET(
            config.DATASET_DIR, config.IMAGE_HW, apply_augms=False, split="val"
        )

        rich.print("[bold green]Train dataset size[/bold green]:", len(self.train_dataset))
        rich.print("[bold green]Valid dataset size[/bold green]:", len(self.valid_dataset))

    def train_dataloader(self):
        assert self.train_dataset is not None
        persistent_workers = config.NUM_DATA_WORKERS > 0

        return DataLoader[batch_types.Sample](
            self.train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            collate_fn=DatasetBase.collate_fn,
            persistent_workers=persistent_workers,
            num_workers=config.NUM_DATA_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        assert self.valid_dataset is not None
        persistent_workers = config.NUM_DATA_WORKERS > 0

        return DataLoader[batch_types.Sample](
            self.valid_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=DatasetBase.collate_fn,
            persistent_workers=persistent_workers,
            num_workers=config.NUM_DATA_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        return self.val_dataloader()
