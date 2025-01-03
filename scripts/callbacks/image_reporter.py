import itertools
from typing import Any, Mapping

import clearml
import config
import cv2
import numpy as np
from data import batch_types, tokens
from jaxtyping import Float, UInt8
from lightning import Callback, LightningModule, Trainer
from lightning_system import LSOutput
from utils import pano


def draw_token_ndarray(
    img: UInt8[np.ndarray, "H W C"],
    layout_uv: Float[np.ndarray, "N 2"],
    color: tuple[int, int, int],
    label_corners: bool = True,
) -> UInt8[np.ndarray, "H W C"]:
    pano_polyline = pano.get_pano_poly_contour(layout_uv)
    pano_polyline *= np.array([img.shape[1], img.shape[0]])
    pano_polyline = pano_polyline.astype(np.int32)

    for pt_pair in itertools.pairwise(pano_polyline):
        if abs(pt_pair[0][0] - pt_pair[1][0]) > 0.5 * img.shape[1]:
            continue

        img = cv2.line(img, tuple(pt_pair[0]), tuple(pt_pair[1]), color, 2)

    corners_2d = (layout_uv * np.array([img.shape[1], img.shape[0]])).astype(np.int32)

    for idx, pt in enumerate(corners_2d):
        img = cv2.circle(img, tuple(pt), 6, color, 2)

        if label_corners:
            img = cv2.putText(
                img, str(idx), tuple(pt + np.array([0, -10])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2
            )

    return img


def draw_token_sequence(
    img: UInt8[np.ndarray, "H W C"],
    layout: tokens.TokenSequence,
    color: tuple[int, int, int],
    label_corners: bool = True,
) -> UInt8[np.ndarray, "H W C"]:
    corners = tokens.get_seq_coordinates(layout).cpu().detach().float().numpy()
    img = draw_token_ndarray(img, corners, color, label_corners)
    return img


def overlay_metadata(img: UInt8[np.ndarray, "H W C"], metadata: Mapping[str, Any]):
    for idx, (k, v) in enumerate(metadata.items()):
        img = cv2.putText(
            img, f"{k}: {v:3f}", (5, 15 + 15 * idx), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    return img


class ImageReporter(Callback):
    def __init__(self, max_samples: int) -> None:
        super().__init__()

        self.max_samples = max_samples
        self._cur_samples = 0
        self._task: clearml.Task | None = clearml.Task.current_task()

    def _report_image(self, idx: str, image: np.ndarray, trainer: Trainer) -> None:
        series = "train" if trainer.training else "valid"

        if self._task is None:
            output_dir = config.TEMP_OUTPUT_DIR / "images" / series / str(trainer.current_epoch)
            output_dir.mkdir(parents=True, exist_ok=True)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_dir / f"{idx}.png"), image)
        else:
            self._task.logger.report_image(
                series=f"{idx}",
                title=f"images:{series}",
                iteration=trainer.current_epoch,
                image=image,
            )

    def _on_batch_end(self, trainer: Trainer, batch: batch_types.Batch, outputs: LSOutput) -> None:
        if self._cur_samples >= self.max_samples:
            return

        preds = tokens.split_token_batch(outputs["pred"])

        for idx, (sample, pred) in enumerate(zip(batch_types.split_batch(batch), preds, strict=True)):
            if self._cur_samples >= self.max_samples:
                break

            image = sample["image"]
            image = draw_token_sequence(image, sample["target"], (0, 255, 0))
            image = draw_token_sequence(image, pred, (255, 0, 0))
            image = overlay_metadata(image, sample["metadata"])

            self._report_image(batch["idx"][idx], image, trainer)
            self._cur_samples += 1

    def _clear_sample_counter(self) -> None:
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

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._clear_sample_counter()

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._clear_sample_counter()

    def on_test_epoch_start(self, *args, **kwargs) -> None:
        self.on_validation_epoch_end(*args, **kwargs)
