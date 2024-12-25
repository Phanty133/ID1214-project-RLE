from pathlib import Path

import cv2
import torch
from data import batch_types, tokens
from data.datasets.dataset_base import DatasetBase
from data.datasets.panocontext import processing


class PanoContextDataset(DatasetBase):
    def __init__(self, dataset_dir: Path, image_size_wh: tuple[int, int]):
        super().__init__(dataset_dir)
        self.image_size_wh = image_size_wh

        if not dataset_dir.exists():
            processing.download_panocontext(dataset_dir)

        self.dataset_dir = dataset_dir
        self.data = list(processing.iter_dataset(dataset_dir))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> batch_types.Sample:
        sample = self.data[index]
        gt_layout = processing.read_room_layout_2d(sample)
        assert gt_layout is not None

        lowest_x_idx = torch.argmin(gt_layout[:, 0])
        gt_layout = torch.cat([gt_layout[lowest_x_idx + 1 :], gt_layout[: lowest_x_idx + 1]]).flip(dims=(0,))

        layout_tokens = [tokens.Token.coo(coord) for coord in gt_layout]
        input_seq = [tokens.Token.eos()] + layout_tokens
        gt_seq = layout_tokens + [tokens.Token.eos()]
        input_seq_torch = tokens.pack_tokens(input_seq)
        gt_seq_torch = tokens.pack_tokens(gt_seq)

        image = cv2.imread(str(self.dataset_dir / sample["image_path"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size_wh)
        torch_image = torch.from_numpy(image).clone().permute(2, 0, 1).float() / 255.0

        return {
            "idx": sample["id"],
            "model_input": {
                "image": torch_image,
                "coords": input_seq_torch,
            },
            "target": gt_seq_torch,
            "metadata": {},
            "image": image,
        }
