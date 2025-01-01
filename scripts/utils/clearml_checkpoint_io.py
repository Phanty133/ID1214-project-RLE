import os
import tempfile
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, override

import clearml
import torch
from clearml.model import Framework
from lightning.fabric.plugins import CheckpointIO
from lightning.fabric.utilities.types import _PATH

log = getLogger(__name__)


class ClearMLCheckpointIO(CheckpointIO):
    def __init__(self) -> None:
        super().__init__()

        self._output_models: dict[str, clearml.OutputModel] = {}

    @override
    def save_checkpoint(
        self, checkpoint: dict[str, Any], path: _PATH, storage_options: Any | None = None
    ) -> None:
        if storage_options is not None:
            raise TypeError(
                "`Trainer.save_checkpoint(..., storage_options=...)` with `storage_options` arg"
                f" is not supported for `{self.__class__.__name__}`."
                " Please implement your custom `CheckpointIO` to define how you'd like to use"
                " `storage_options`."
            )

        task: clearml.Task | None = clearml.Task.current_task()

        if task is None:
            log.warning("ClearML Task not found. Skipping checkpoint save.")
            return

        path = Path(path)
        model_name = path.stem
        filename = path.name

        if model_name not in self._output_models:
            self._output_models[model_name] = clearml.OutputModel(
                task=task,
                name=model_name,
                label_enumeration=task.get_labels_enumeration(),
                framework=Framework.pytorch,
                # base_model_id=in_model_id, TODO:
            )

        fd, temp_file = tempfile.mkstemp()
        os.close(fd)
        torch.save(checkpoint, temp_file)
        self._output_models[model_name].update_weights(
            weights_filename=str(temp_file),
            auto_delete_file=True,
            update_comment=False,
            target_filename=filename,
        )

    @override
    def load_checkpoint(
        self, path: _PATH, map_location: Callable | None = lambda storage, loc: storage
    ) -> dict[str, Any]:
        task: clearml.Task | None = clearml.Task.current_task()

        if task is None:
            raise FileNotFoundError(f"ClearML Task not found. Checkpoint file not found: {path}")

        path = Path(path)
        model_name = path.stem

        input_model = clearml.InputModel(name=model_name)
        checkpoint = input_model.get_local_copy()

        return torch.load(checkpoint, map_location=map_location)

    @override
    def remove_checkpoint(self, path: _PATH) -> None:
        log.error("ClearMLCheckpointIO.remove_checkpoint is not implemented.")
