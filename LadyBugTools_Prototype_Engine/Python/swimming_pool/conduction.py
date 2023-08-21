from typing import Tuple

import numpy as np


def edge_length_from_aspect_ratio(
    surface_area: float, aspect_ratio: float = 1.61803399
) -> Tuple[float]:
    """Estimate pool width and length from surface area.

    Args:
        surface_area (float): Surface area of the pool in m2.
        aspect_ratio (float, optional): Ratio of length to width. Defaults to
            1.61803399, or "The Golden Ratio".

    Returns:
        (float, float,):
            - pool_width: Width of the pool in m.
            - pool_length: Length of the pool in m.
    """

    width = (surface_area * (1 / aspect_ratio)) ** 0.5
    length = width * aspect_ratio

    return width, length


def ground_interface_area_prism(surface_area: float, average_depth: float) -> float:
    """Estimate pool ground interface area from surface area and depth.
        This method assumes that the pool may be described as a rectangular prism.

    Args:
        surface_area (float): Surface area of the pool in m2.
        average_depth (float): Depth of the pool in m.

    Returns:
        float: Ground interface area of the pool in m2.
    """

    width, length = edge_length_from_aspect_ratio(surface_area)
    return surface_area + (((2 * length) + (2 * width)) * average_depth)


def ground_interface_area_cylinder(surface_area: float, average_depth: float) -> float:
    """Estimate pool ground interface area from surface area and depth.
        This method assumes that the pool may be described as a cylinder.

    Args:
        surface_area (float): Surface area of the pool in m2.
        average_depth (float): Depth of the pool in m.

    Returns:
        float: Ground interface area of the pool in m2.
    """

    radius = (surface_area / np.pi) ** 0.5

    return (2 * np.pi * radius * average_depth) + surface_area


def conduction_gain_interface_area(
    surface_area: float,
    average_depth: float,
    interface_u_value: float,
    soil_temperature: float,
    water_temperature: float,
) -> float:
    """The total conduction through the ground interface of the pool, from the
    pool to the soil (-Ve, heat lost) or from the soil to the pool (+Ve, heat
    gained).

    Source:
        Eq 12 from "Nouaneque, H.V., et al, (2011), Energy model validation of
        heated outdoor swimming pools in cold weather. SimBuild 2011.

    Args:
        surface_area (float): Surface area of the pool in m2.
        average_depth (float): Depth of the pool in m.
        interface_u_value (float): U-value of the pool-ground interface in W/m2K.
        soil_temperature (float): Temperature of the soil in C.
        water_temperature (float): Temperature of the water in C.

    Returns:
        float: Total conduction through the ground interface of the pool in W.

    """

    interface_area = ground_interface_area_prism(surface_area, average_depth)
    gain = interface_u_value * interface_area * (water_temperature - soil_temperature)
    return -gain
