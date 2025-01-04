"""
Script for testing inference on a panorama image.
"""

import sys
from pathlib import Path


FILE_DIR = Path(__file__).parent
ROOT_DIR = FILE_DIR.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
DATA_DIR = FILE_DIR / "data"

sys.path.append(".")
sys.path.append(str(ROOT_DIR))
sys.path.append(str(SCRIPTS_DIR))

import argparse
import logging
from collections import OrderedDict
import scripts.utils.pdraw as draw
import scripts.utils.pano as pn

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from jaxtyping import Float32, UInt8
from rich.logging import RichHandler
from torch import Tensor

from scripts.callbacks import image_reporter
from scripts.data import tokens
from scripts.model.model import Model

log = logging.getLogger(__name__)
log.addHandler(RichHandler(rich_tracebacks=True))
log.setLevel(logging.INFO)


IMAGE_SIZE = [512, 512]


def _check_files(ckpt_path: Path, pano_path: Path) -> bool:
    files_ok = True

    if not ckpt_path.exists():
        log.error(f"Checkpoint file not found at {ckpt_path}")
        files_ok = False

    if not pano_path.exists():
        log.error(f"Panorama file not found at {pano_path}")
        files_ok = False

    return files_ok


def rename_compiled_layers(ckpt: OrderedDict[str, Tensor]) -> OrderedDict[str, Tensor]:
    new_ckpt = OrderedDict()

    for key, value in ckpt.items():
        new_key = key.replace("model.", "", 1)

        if "._orig_mod" in key:
            new_key = new_key.replace("._orig_mod", "")

        new_ckpt[new_key] = value

    return new_ckpt


def apply_pano_preprocessing(image: UInt8[np.ndarray, "H W 3"]) -> Float32[Tensor, "3 H W"]:
    torch_image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    torch_image = F.resize(torch_image, IMAGE_SIZE)
    torch_image = F.normalize(torch_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return torch_image


def main() -> int:
    parser = argparse.ArgumentParser(description="Test inference on a panorama image.")
    parser.add_argument("--ckpt", type=str, help="Path to the model checkpoint.")
    parser.add_argument("--pano", type=str, help="Path to the panorama image.")
    parser.add_argument(
        "--corner-labels", action="store_true", help="Draw corner labels on the output image."
    )

    args = parser.parse_args()
    ckpt_path = Path(args.ckpt)
    pano_path = Path(args.pano)

    if not _check_files(ckpt_path, pano_path):
        return 1

    log.info("Loading files")
    pano = cv2.cvtColor(cv2.imread(str(pano_path)), cv2.COLOR_BGR2RGB)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if pano is None:
        log.error("Failed to read panorama image")
        return 1

    log.info("Starting inference on panorama image")
    pano_torch = apply_pano_preprocessing(pano)
    model = Model()
    model_weights = ckpt.get("state_dict", ckpt)
    model_weights = rename_compiled_layers(model_weights)
    try:
        model.load_state_dict(model_weights)
    except Exception as e:
        log.error("Failed to load model state dict.", exc_info=e)
        return 1

    token_batch = model.inference(pano_torch.unsqueeze(0))
    token_seq = tokens.split_token_batch(token_batch)[0]
    contour_uv = tokens.get_seq_coordinates(token_seq).cpu().detach().numpy()
    output_img = image_reporter.draw_token_ndarray(
        pano, contour_uv, color=(255, 0, 0), label_corners=args.corner_labels
    )

    topdown_coordinates =  []
    for uv in contour_uv:
        uv_rotated =  [1-(uv[0] + 0.75) % 1, uv[1]]
        topdown_coordinates.append(pn.uv_to_topdown(1.6, uv_rotated))

    draw.create_image(topdown_coordinates, 1024, 1024, 100, Path("topdown.png").resolve())
    cv2.imwrite("output.png", cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
    log.info("Output image saved to output.png")
    return 0


if __name__ == "__main__":
    exit(main())
