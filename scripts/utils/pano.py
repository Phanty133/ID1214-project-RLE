from typing import Literal

import numpy as np
from jaxtyping import Float32
import shapely.geometry as sg

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

def get_topdown_coords( xyz: Float32[np.ndarray, "N 3"], meridian: MeridianPlane = "yz"
) -> Float32[np.ndarray, "N 2"]:
    return np.stack([x, z], axis=-1)
    


def uv_to_spherical( uv: Float32[np.ndarray, "N 2"], meridian: MeridianPlane = "yz"
) -> Float32[np.ndarray, "N 3"]:
    u = uv[0]
    v = uv[1]
    phi = np.pi * (v - 0.5)
    theta = 2 * np.pi *(u-0.5)
    r = 1
    return np.stack([r, phi, theta], axis=-1)

def uv_to_topdown(uv: Float32[np.ndarray, "N 2"], meridian: MeridianPlane = "yz"
) -> Float32[np.ndarray, "N 3"]:
    spherical = uv_to_spherical(uv)
    cartesian = spherical_to_cartesian(spherical)
    x = cartesian[0]
    y = cartesian[1]
    z = cartesian[2]

    return [y, x]

if __name__ == "__main__":

    #Test for single point
    uv = [0.75, 0.75]
    sample_corner_uv =  np.stack(uv, axis=-1)
    spherical = uv_to_spherical(sample_corner_uv)
    cartesian = spherical_to_cartesian(spherical)

    x = cartesian[0]
    y = cartesian[1]
    z = cartesian[2]
    topdown = uv_to_topdown(uv)
    print("UV: ", uv)
    print("Spherical coord: ", spherical)
    print("Cartesian coord: ", [x, y, z])
    print("Topdown :", topdown)

    #Test for multiple points
    uv_arr = [[0.32, 0.41], [0.22, 0.73], [0.74, 0.25]]
    uv_target_arr = [[0.34, 0.42], [0.25, 0.71], [0.75, 0.21]]

    coord_arr = [] 
    target_arr = [] 

    for element in uv_arr:
        td = uv_to_topdown(element)
        coord_arr.append(td)

    for element in uv_target_arr:
        td = uv_to_topdown(element)
        target_arr.append(td)

    print("ORIGINAL POSITIONS", coord_arr)
    print("TARGET POSITIONS", target_arr)

    poly = sg.Polygon(coord_arr)
    target_poly = sg.Polygon(target_arr)
    intersection = poly.intersection(target_poly).area
    union = poly.union(target_poly).area
    iou = intersection / union
    print("IOU", iou)
