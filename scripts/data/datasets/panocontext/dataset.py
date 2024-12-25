from pathlib import Path

import cv2
from data import batch_types
from data.datasets.dataset_base import DatasetBase
from data.datasets.panocontext import processing


class PanoContextDataset(DatasetBase):
    def __init__(self, dataset_dir: Path, image_size_wh: tuple[int, int]):
        super().__init__(dataset_dir, image_size_wh)

        if not dataset_dir.exists():
            processing.download_panocontext(dataset_dir)

        self.dataset_dir = dataset_dir
        self.data = list(processing.iter_dataset(dataset_dir))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> batch_types.Sample:
        sample = self.data[index]
        gt_layout = processing.read_room_layout_2d(sample)
        image = cv2.imread(str(self.dataset_dir / sample["image_path"]))
        assert gt_layout is not None

        sample = self._process_sample(sample["id"], gt_layout, image)
        return sample
