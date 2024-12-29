from pathlib import Path
from typing import Literal

import cv2
import orjson
from data import batch_types
from data.datasets.dataset_base import DatasetBase
from data.datasets.zind import processing


class ZindDataset(DatasetBase):
    def __init__(
        self,
        dataset_dir: Path,
        image_size_wh: tuple[int, int],
        apply_augms: bool,
        split: Literal["train", "val"],
    ):
        super().__init__(dataset_dir, image_size_wh, apply_augms, split)

        zind_dir = processing.get_zind_dir(dataset_dir)

        if not zind_dir.exists():
            raise NotImplementedError("Download Zind dataset")

        self.dataset_dir = dataset_dir
        self.data = list(processing.iter_dataset(dataset_dir))

        partitions = orjson.loads((zind_dir / "zind_partition.json").read_bytes())
        self.data = [sample for sample in self.data if sample["floorplan_id"] in partitions[self.split]]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> batch_types.Sample:
        sample = self.data[index]
        image = cv2.imread(str(self.dataset_dir / sample["image_path"]))
        sample = self._process_sample(sample["id"], sample["layout_2d"], image)

        return sample
