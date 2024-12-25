import itertools

import config
import cv2
import numpy as np
from data import batch_types, tokens
from jaxtyping import UInt8
from lightning import Callback, LightningModule, Trainer
from lightning_system import LSOutput
from utils import pano


def draw_tokens(
    img: UInt8[np.ndarray, "H W C"], layout: tokens.TokenSequence, color: tuple[int, int, int]
) -> UInt8[np.ndarray, "H W C"]:
    corners = layout["coord"][layout["cls"] == tokens.TokenCls.COO.value].cpu().detach().float().numpy()
    pano_polyline = pano.get_pano_poly_contour(corners)
    pano_polyline *= np.array([img.shape[1], img.shape[0]])
    pano_polyline = pano_polyline.astype(np.int32)

    for pt_pair in itertools.pairwise(pano_polyline):
        if abs(pt_pair[0][0] - pt_pair[1][0]) > 0.5 * img.shape[1]:
            continue

        img = cv2.line(img, tuple(pt_pair[0]), tuple(pt_pair[1]), color, 2)

    corners_2d = (corners * np.array([img.shape[1], img.shape[0]])).astype(np.int32)

    for idx, pt in enumerate(corners_2d):
        img = cv2.circle(img, tuple(pt), 6, color, 2)
        img = cv2.putText(
            img, str(idx), tuple(pt + np.array([0, -10])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2
        )

    return img


class ImageReporter(Callback):
    def __init__(self, max_samples: int) -> None:
        super().__init__()

        self.max_samples = max_samples
        self._cur_samples = 0
        self._task = None

    def _report_image(self, idx: str, image: np.ndarray, trainer: Trainer) -> None:
        if self._task is None:
            subdir = "train" if trainer.training else "valid"
            output_dir = config.TEMP_OUTPUT_DIR / "images" / subdir / str(trainer.current_epoch)
            output_dir.mkdir(parents=True, exist_ok=True)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_dir / f"{idx}.png"), image)
        else:
            raise NotImplementedError

    def _on_batch_end(self, trainer: Trainer, batch: batch_types.Batch, outputs: LSOutput) -> None:
        if self._cur_samples >= self.max_samples:
            return

        for idx, sample in enumerate(batch_types.split_batch(batch)):
            if self._cur_samples >= self.max_samples:
                break

            pred_seq: tokens.TokenSequence = {
                "cls": outputs["pred"]["cls"][idx],
                "coord": outputs["pred"]["coord"][idx],
            }
            image = sample["image"]
            image = draw_tokens(image, sample["target"], (0, 255, 0))
            image = draw_tokens(image, pred_seq, (255, 0, 0))

            self._report_image(batch["idx"][idx], image, trainer)
            self._cur_samples += 1

    def _on_epoch_end(self) -> None:
        self._cur_samples = 0

    def on_train_batch_end(  # type: ignore
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: LSOutput,
        batch: batch_types.Batch,
        batch_idx: int,
    ) -> None:
        self._on_batch_end(trainer, batch, outputs)

    def on_validation_batch_end(  # type: ignore
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: LSOutput,
        batch: batch_types.Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._on_batch_end(trainer, batch, outputs)

    def on_test_batch_end(self, *args, **kwargs) -> None:
        self.on_validation_epoch_end(*args, **kwargs)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._on_epoch_end()

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._on_epoch_end()

    def on_test_epoch_end(self, *args, **kwargs) -> None:
        self.on_validation_epoch_end(*args, **kwargs)
