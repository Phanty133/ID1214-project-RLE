import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import orjson
import torch
from data.datasets.zind.typings import ZindData, ZindSample
from utils.pano import poly_3d_to_2d

log = logging.getLogger(__name__)


def get_zind_dir(dataset_root_dir: Path) -> Path:
    return dataset_root_dir / "zind"


def read_zind_data(json_path: Path | str) -> ZindData | None:
    if isinstance(json_path, str):
        json_path = Path(json_path)

    try:
        return orjson.loads(json_path.read_bytes())
    except FileNotFoundError:
        return None


def iter_dataset(dataset_root_dir: Path) -> Iterator[ZindSample]:
    ds_dir = get_zind_dir(dataset_root_dir)

    for subdir in sorted(ds_dir.iterdir()):
        if subdir.is_file():
            continue

        zind_data = read_zind_data(subdir / "zind_data.json")

        if zind_data is None:
            log.warning(f"Skipping {subdir} because zind_data.json is missing")
            continue

        for floor in zind_data["merger"].values():
            for complete_room in floor.values():
                for partial_room in complete_room.values():
                    for pano in partial_room.values():
                        layout_topdown = np.array(pano["layout_raw"]["vertices"])
                        z = np.ones_like(layout_topdown[:, 0])[:, None] * -pano["camera_height"]
                        layout_3d = np.concatenate([layout_topdown, z], axis=-1)
                        layout_3d[..., 0] *= -1
                        layout_2d = torch.from_numpy(poly_3d_to_2d(layout_3d))
                        img_id = pano["image_path"].split("/")[-1].split(".")[0]
                        sample: ZindSample = {
                            "id": f"{subdir.name}/{img_id}",
                            "floorplan_id": subdir.name,
                            "image_path": subdir / pano["image_path"],
                            "layout_2d": layout_2d,
                        }

                        yield sample
