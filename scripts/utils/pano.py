from typing import Literal

import numpy as np
from jaxtyping import Float32

MeridianPlane = Literal["xz", "yz", "-xz", "-yz"]


def spherical_to_cartesian(
    spherical: Float32[np.ndarray, "N 3"], meridian: MeridianPlane = "yz"
) -> Float32[np.ndarray, "N 3"]:
    r = spherical[..., 0]
    azimuthal_angle = spherical[..., 1]
    polar_angle = spherical[..., 2]

    sin_polar = np.sin(polar_angle)

    if meridian == "-xz" or meridian == "-yz":
        azimuthal_angle = np.pi - azimuthal_angle

    if meridian == "xz" or meridian == "-xz":
        x = r * sin_polar * np.cos(azimuthal_angle)
        y = r * sin_polar * np.sin(azimuthal_angle)
    elif meridian == "yz" or meridian == "-yz":
        x = r * sin_polar * np.sin(azimuthal_angle)
        y = r * sin_polar * np.cos(azimuthal_angle)

    z = r * np.cos(polar_angle)

    return np.stack([x, y, z], axis=-1)


def cartesian_to_spherical(
    xyz: Float32[np.ndarray, "N 3"], meridian: MeridianPlane = "yz"
) -> Float32[np.ndarray, "N 3"]:
    # Polar axis=z
    r = np.linalg.norm(xyz, axis=-1)

    polar_angle = np.arccos(np.divide(xyz[..., 2], r, where=r != 0))

    if meridian == "xz" or meridian == "-xz":
        azimuthal_angle = np.arctan2(xyz[..., 1], xyz[..., 0])
    elif meridian == "yz" or meridian == "-yz":
        azimuthal_angle = np.arctan2(xyz[..., 0], xyz[..., 1])

    if meridian == "-xz" or meridian == "-yz":
        azimuthal_angle = np.pi - azimuthal_angle

    out = np.stack([r, azimuthal_angle, polar_angle], axis=-1)
    out = np.where(np.isnan(out), np.zeros_like(out), out)

    return out


def poly_2d_to_3d(poly_2d: Float32[np.ndarray, "N 2"]) -> Float32[np.ndarray, "N 3"]:
    coords = poly_2d.astype(np.float32)
    angles = np.zeros_like(poly_2d)
    angles[..., 0] = np.pi * (2 * coords[..., 0] - 1)
    angles[..., 1] = np.pi * (coords[..., 1])

    lengths = np.ones((*poly_2d.shape[:-1], 1), dtype=np.float32)
    spherical = np.concatenate((lengths, angles), axis=-1)
    rays = spherical_to_cartesian(spherical)

    return rays


def poly_3d_to_2d(poly_3d: Float32[np.ndarray, "N 3"]) -> Float32[np.ndarray, "N 2"]:
    spherical = cartesian_to_spherical(poly_3d)
    angles = spherical[..., [1, 2]]
    coords = np.zeros_like(angles, dtype=np.float32)
    coords[..., 0] = (angles[..., 0] / np.pi + 1) / 2
    coords[..., 1] = angles[..., 1] / np.pi

    return coords


def get_pano_poly_contour(
    poly_2d: Float32[np.ndarray, "N 2"], points_per_segment: int = 60
) -> Float32[np.ndarray, "N_interp 2"]:
    poly_3d = poly_2d_to_3d(poly_2d)
    points = np.stack(
        [poly_3d, np.roll(poly_3d, -1, axis=0)],
        axis=1,
        dtype=np.float32,
    )
    point_deltas = points[:, 1] - points[:, 0]
    interps = np.linspace(start=0, stop=1 - 1 / points_per_segment, num=points_per_segment).reshape(-1, 1, 1)
    contour_3d = point_deltas * interps + points[:, 0]
    contour_3d = contour_3d.transpose(1, 0, 2).reshape(-1, 3)
    contour = poly_3d_to_2d(contour_3d)
    contour = np.mod(contour, [1, 1])

    return contour
