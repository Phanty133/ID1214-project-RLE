from typing import Literal

import numpy as np
from jaxtyping import Float32
import shapely.geometry as sg

MeridianPlane = Literal["xz", "yz", "-xz", "-yz"]


def spherical_to_cartesian(
    spherical: Float32[np.ndarray, "N 3"], meridian: MeridianPlane = "yz"
) -> Float32[np.ndarray, "N 3"]:
    r = spherical[..., 0]
    azimuthal_angle = spherical[1]
    polar_angle = spherical[2]

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

    polar_angle = np.arccos(np.divide(xyz[2], r, where=r != 0))

    if meridian == "xz" or meridian == "-xz":
        azimuthal_angle = np.arctan2(xyz[1], xyz[0])
    elif meridian == "yz" or meridian == "-yz":
        azimuthal_angle = np.arctan2(xyz[0], xyz[1])

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
    
def uv_to_spherical(cameraHeight, uv: Float32[np.ndarray, "N 2"], meridian: MeridianPlane = "yz"
) -> Float32[np.ndarray, "N 3"]:
    u = uv[0]
    v = uv[1]
    phi = np.pi * (v)
    theta = 2 * np.pi *(u-0.5)
    r = -cameraHeight/np.cos(phi)
    #r = 1
    return np.stack([r, theta, phi], axis=-1)

def spherical_to_uv( spherical: Float32[np.ndarray, "N 3"], meridian: MeridianPlane = "yz"
) -> Float32[np.ndarray, "N 2"]:
    r = spherical[0]
    phi = spherical[2] 
    theta = spherical[1] 
    u = theta/(2*np.pi) + 0.5
    v = (phi/np.pi) #v- axis might be inverted?
    return np.stack([u,v], axis=-1)


def uv_to_topdown(cameraheight, uv: Float32[np.ndarray, "N 2"], meridian: MeridianPlane = "yz"
) -> Float32[np.ndarray, "N 3"]:
    spherical = uv_to_spherical( cameraheight, uv)
    cartesian = spherical_to_cartesian(spherical)
    x = cartesian[0]
    y = cartesian[1]
    z = cartesian[2]

    return [y, x]

if __name__ == "__main__":

    #Test for single UV point
    uv = [0.375     , 0.80408672]
    sample_corner_uv =  np.stack(uv, axis=-1)
    cameraheight = 2
    spherical = uv_to_spherical(cameraheight, sample_corner_uv)
    cartesian = spherical_to_cartesian(spherical)

    x = cartesian[0]
    y = cartesian[1]
    z = cartesian[2]
    topdown = uv_to_topdown(cameraheight, uv)
    print("Original UV: ", uv)
    print("Spherical coord: ", spherical)
    print("Cartesian coord: ", [x, y, z])

    #Test for single cartesian coordinate
    c = [-1, 1 ,0-cameraheight]
    print("Original Cartesian: ", cartesian)
    c_sphere = cartesian_to_spherical(cartesian)
    print("Spherical coord: ", c_sphere)

    c_uv = spherical_to_uv(c_sphere)
    print("Final UV: ", c_uv)

    #Test for multiple UV points
    uv_arr = [[0.375     , 0.80408672],[0.625     , 0.80408672],[0.875     , 0.80408672],[0.125     , 0.80408672]]
    #uv_arr = [[0.32, 0.41], [0.22, 0.73], [0.74, 0.25]]
    uv_target_arr = [[0.34, 0.42], [0.25, 0.71], [0.75, 0.21]]

    coord_arr = [] 
    target_arr = [] 

    for element in uv_arr:
        td = uv_to_topdown(cameraheight, element)
        coord_arr.append(td)

    for element in uv_target_arr:
        td = uv_to_topdown(cameraheight, element)
        target_arr.append(td)

    print("ORIGINAL UV POSITIONS", coord_arr)
    print("TARGET POSITIONS", target_arr)

    poly = sg.Polygon(coord_arr)
    target_poly = sg.Polygon(target_arr)
    intersection = poly.intersection(target_poly).area
    union = poly.union(target_poly).area
    iou = intersection / union
    print("IOU", iou)

    
    #Test for multiple cartesian points
    cart_arr = [[-1, 1, -2], [1, 1, -2], [1, -1, -2], [-1, -1, -2]]
    uvs_arr = []
    for element in cart_arr:
        print(element)
        c_sphere = cartesian_to_spherical(element)
        c_uv = spherical_to_uv(c_sphere)
        uvs_arr.append(c_uv)

    print("ORIGINAL CARTESIAN POSITIONS", cart_arr)
    print("FINAL UV POSITIONS", uvs_arr)

