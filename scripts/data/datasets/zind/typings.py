from pathlib import Path
from typing import TypedDict

from jaxtyping import Float32
from torch import Tensor

Vector2D = tuple[float, float]


class FloorPlanTransformation(TypedDict):
    rotation: float
    translation: Vector2D
    scale: float


class FloorplanToRedrawTransformation(TypedDict):
    image_path: str
    rotation: float
    translation: Vector2D
    checksum: str
    scale: float


class LayoutRaw(TypedDict):
    doors: list[Vector2D]
    vertices: list[Vector2D]
    windows: list[Vector2D]
    openings: list[Vector2D]


class LayoutComplete(TypedDict):
    doors: list[Vector2D]
    internal: list[Vector2D]
    vertices: list[Vector2D]
    windows: list[Vector2D]
    openings: list[Vector2D]


class Pano(TypedDict):
    layout_raw: LayoutRaw
    is_ceiling_flat: bool
    is_primary: bool
    image_path: str
    is_inside: bool
    checksum: str
    layout_complete: LayoutComplete
    camera_height: float
    floor_number: int
    label: str
    floor_plan_transformation: FloorPlanTransformation
    ceiling_height: float


PartialRoom = dict[str, Pano]
CompleteRoom = dict[str, PartialRoom]
MergedFloor = dict[str, CompleteRoom]


class Pin(TypedDict):
    position: Vector2D
    label: str


class RedrawPano(TypedDict):
    doors: list[tuple[Vector2D, Vector2D]]
    vertices: list[Vector2D]
    pins: list[Pin]
    windows: list[tuple[Vector2D, Vector2D]]


Room = dict[str, RedrawPano]
Floor = dict[str, Room]


class ZindData(TypedDict):
    redraw: dict[str, Floor]
    floorplan_to_redraw_transformation: dict[str, FloorplanToRedrawTransformation]
    scale_meters_per_coordinate: dict[str, float]
    merger: dict[str, MergedFloor]


class ZindSample(TypedDict):
    id: str
    floorplan_id: str
    image_path: Path
    layout_2d: Float32[Tensor, "N 2"]
