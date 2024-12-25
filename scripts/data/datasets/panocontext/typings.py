from pathlib import Path
from typing import Any, Literal, TypedDict

Tag = Literal["bedroom", "living_room"]


class PanoContextSample(TypedDict):
    id: str
    tag: Tag
    image_path: Path
    annotation_path: Path


class Position2D(TypedDict):
    x: float
    y: float


class Projection(TypedDict):
    camera: int
    image: int
    position2D: Position2D


class Point(TypedDict):
    position3D: dict[str, Any]
    projection: list[Projection]


class Object(TypedDict):
    type: int
    name: str
    time: str
    creator: str
    points: list[Point]


class Principle(TypedDict):
    x: int
    y: int


class Resolution(TypedDict):
    width: int
    height: int


class Film(TypedDict):
    width: float
    height: float


class Image(TypedDict):
    folder: str
    file: str
    frame: int
    principle: Principle
    resolution: Resolution
    film: Film
    type: str
    plane: str


class Rotation(TypedDict):
    pitch: int
    yaw: int
    roll: int


class Camera(TypedDict):
    modelview: list[int]
    rotation: Rotation
    focal_length: int
    images: list[Image]


class PointCloud(TypedDict):
    number: int
    vertex: str
    color: str


class PanoContextAnnotation(TypedDict):
    cameras: list[Camera]
    objects: list[Object]
    PointCloud: PointCloud
