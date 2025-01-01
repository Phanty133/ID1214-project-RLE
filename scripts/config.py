from pathlib import Path

from data.datasets.panocontext.dataset import PanoContextDataset 
from lightning_system import LSConfig

DATASET_DIR = Path("./workspace/datasets")
TEMP_OUTPUT_DIR = Path("./workspace/outputs")

TRAIN_DATASET = PanoContextDataset
VALID_DATASET = PanoContextDataset
IMAGE_HW: tuple[int, int] = (384, 384)
BATCH_SIZE: int = 8
GRAD_ACC = 6
NUM_DATA_WORKERS: int = 4

LS_CONFIG: LSConfig = {
    "cls_coef": 0.05,
    "coord_coef": 1.0,
    "initial_lr": 5e-5,
    "decay_epochs": 50,
    "lr_min": 1e-6,
    "cos_annealing": True,
    "compile_model": False,
}
MAX_EPOCHS: int | None = -1
NUM_GPUS: int = 1

CLEARML_PROJECT_NAME = "RLE"
CLEARML_TASK_NAME = "ZinD test"
