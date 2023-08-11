from typing import Tuple
from warnings import warn

import numpy as np
from honeybee_energy.construction.opaque import OpaqueConstruction
from scipy.interpolate import interp1d


def pool_width_length(
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

    width, length = pool_width_length(surface_area)
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


def pool_shape_factor(surface_area: float, average_depth: float) -> float:
    """Estimate pool shape factor from surface area and depth.
    This method assumes that the pool may be described as a rectangular prism.

    Equations 4 and 5 from Woolley J, et al., Swimming pools as heat sinks for air
    conditioners: Model design and experimental..., Building and Environment
    (2010), doi:10.1016/j.buildenv.2010.07.014

    Args:
        surface_area (float): Surface area of the pool in m2.
        average_depth (float): Depth of the pool in m.

    Returns:
        float: Shape factor of the pool.
    """

    interface_area = ground_interface_area_prism(surface_area, average_depth)
    d = 2 * average_depth
    D = ((interface_area + (d**2)) ** 0.5) - d

    return d / D


def pool_characteristic_length(surface_area: float) -> float:
    """The L_c component of the shape factor equation.

    Equation 5 from Woolley J, et al., Swimming pools as heat sinks for air
    conditioners: Model design and experimental..., Building and Environment
    (2010), doi:10.1016/j.buildenv.2010.07.014

    Args:
        surface_area (float): Surface area of the pool in m2.

    Returns:
        float: Characteristic length of the pool in m.
    """

    return (surface_area / (4 * np.pi)) ** 0.5


def pool_dimensionless_conduction_heat_rate(shape_factor: float) -> float:
    """The q_ss component of the shape factor equation.

    From Woolley J, et al., Swimming pools as heat sinks for air
    conditioners: Model design and experimental..., Building and Environment
    (2010), doi:10.1016/j.buildenv.2010.07.014

    Args:
        shape_factor (float): Shape factor of the pool.

    Returns:
        float: Dimensionless conduction heat rate of the pool.
    """

    d_over_D = [0.1, 1.0, 2.0]
    q_ss = [0.943, 0.956, 0.961]
    q_ss_interpolant = interp1d(d_over_D, q_ss, kind="quadratic", bounds_error=True)

    try:
        return q_ss_interpolant(shape_factor)
    except ValueError as exc:
        if shape_factor < min(d_over_D):
            warn(
                f"{exc}\nSetting conduction_heat_rate to {min(q_ss)} as shape_factor < {min(d_over_D)}"
            )
            return 0.943
        else:
            warn(
                f"{exc}\nSetting conduction_heat_rate to {max(q_ss)} as shape_factor > {max(d_over_D)}"
            )
            return 0.961


def conduction_gain_interface_area(
    surface_area: float,
    average_depth: float,
    interface_u_value: float,
    soil_temperature: float,
    water_temperature: float,
) -> float:
    """The total conduction through the ground interface of the pool.

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

    return interface_u_value * interface_area * (water_temperature - soil_temperature)


def conduction_gain_shape_factor(
    surface_area: float,
    average_depth: float,
    soil_temperature: float,
    water_temperature: float,
    soil_conductivity: float = 0.33,
) -> float:
    """The total conduction through the ground interface of the pool.

    Source:
        Equation 2 from Woolley J, et al., Swimming pools as heat sinks for air
        conditioners: Model design and experimental..., Building and Environment
        (2010), doi:10.1016/j.buildenv.2010.07.014

    Args:
        surface_area (float): Surface area of the pool in m2.
        average_depth (float): Depth of the pool in m.
        soil_temperature (float): Temperature of the soil in C.
        water_temperature (float): Temperature of the water in C.
        soil_conductivity (float): Thermal conductivity of the soil in W/mK.
            Defaults to 0.33W/mK for dry sand.

    Returns:
        float: Total conduction through the ground interface of the pool in W.

    """

    interface_area = ground_interface_area_prism(surface_area, average_depth)
    d = 2 * average_depth
    D = ((interface_area + (d**2)) ** 0.5) - d
    shape_factor = pool_shape_factor(surface_area, average_depth)
    dimensionless_conduction_heat_rate = pool_dimensionless_conduction_heat_rate(
        shape_factor
    )
    characteristic_length = pool_characteristic_length(surface_area)
    shape_factor_surface_area = (2 * (D**2)) + (4 * D * d)

    return (
        (1 / 2 * characteristic_length)
        * dimensionless_conduction_heat_rate
        * soil_conductivity
        * (shape_factor_surface_area / interface_area)
        * (soil_temperature - water_temperature)
    )  # W
