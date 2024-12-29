import shutil
import zipfile
from pathlib import Path
from typing import Iterator

import httpx
import orjson
import rich
import torch
from data.datasets.panocontext.typings import PanoContextAnnotation, PanoContextSample, Tag
from jaxtyping import Float32
from rich.progress import Progress
from torch import Tensor

PANOCONTEXT_DATA_URL = "https://panocontext.cs.princeton.edu/panoContext_data.zip"


def _get_panocontext_dir(dataset_root_dir: Path) -> Path:
    return dataset_root_dir / "panocontext"


def download_panocontext(dataset_root_dir: Path):
    dataset_dir = _get_panocontext_dir(dataset_root_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dataset_dir / "panocontext.zip"

    with Progress() as progress:
        task = progress.add_task("Downloading PanoContext...", total=None)

        with httpx.stream("GET", PANOCONTEXT_DATA_URL) as response:
            total = int(response.headers.get("content-length", 0))
            progress.update(task, total=total)

            with open(zip_path, "wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_dir)

    shutil.rmtree(dataset_dir / "bedroom_exp")
    shutil.rmtree(dataset_dir / "living_room_exp")
    zip_path.unlink()


def iter_dataset(dataset_root_dir: Path) -> Iterator[PanoContextSample]:
    ds_dir = _get_panocontext_dir(dataset_root_dir)
    subdirs: list[Tag] = ["bedroom", "living_room"]

    for subdir in subdirs:
        for sample_dir in (ds_dir / subdir).iterdir():
            if sample_dir.is_file():
                continue

            id = sample_dir.name.split("_")[1]
            image_path = sample_dir / f"{sample_dir.name}.jpg"
            annotation_path = sample_dir / f"{sample_dir.name}.json"
            sample: PanoContextSample = {
                "id": id,
                "tag": subdir,
                "image_path": image_path,
                "annotation_path": annotation_path,
            }
            layout = read_room_layout_2d(sample)

            if layout is None:
                rich.print(f"[red]Skipping sample because layout is missing:[/red] {sample['id']}")
                continue

            yield sample


def read_room_layout_2d(sample: PanoContextSample) -> Float32[Tensor, "N 2"] | None:
    """
    Returns the room layout in the image plane, normalized.
    """

    if not sample["annotation_path"].exists():
        rich.print(f"[red]Annotation file not found:[/red] {sample['annotation_path']}")
        return None

    annotation: PanoContextAnnotation = orjson.loads(sample["annotation_path"].read_bytes())
    img_res = annotation["cameras"][0]["images"][0]["resolution"]
    img_wh = torch.tensor([img_res["width"], img_res["height"]], dtype=torch.float32)

    for obj in annotation["objects"]:
        if obj["type"] != 3:  # Skip if it's not a layout object
            continue

        layout_raw_dicts = [p["projection"][0]["position2D"] for p in obj["points"]]
        layout_raw = [[p["x"], p["y"]] for p in layout_raw_dicts]
        layout = torch.tensor(layout_raw, dtype=torch.float32)
        layout = layout / img_wh
        layout = layout[layout[:, 1] > 0.5]
        layout = layout[layout[:, 0].argsort()]

        return layout

    return None
