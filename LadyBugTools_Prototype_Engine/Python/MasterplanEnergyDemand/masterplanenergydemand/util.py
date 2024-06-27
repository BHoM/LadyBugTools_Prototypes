# region: IMPORTS
# pylint: disable=E0401


from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from honeybee_energy.construction.opaque import OpaqueConstruction
from honeybee_energy.material.opaque import EnergyMaterial
from ladybug.datacollection import BaseCollection
from ladybug.wea import Wea
from ladybug_geometry.geometry2d import Vector2D
from ladybug_geometry.geometry3d import Face3D, Vector3D
from matplotlib.colors import colorConverter
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression

from .config import DATA_PATH, SRI_DATA, logger

# pylint: enable=E0401
# endregion: IMPORTS


def cardinality(direction_angle: float, directions: int = 16):
    """Returns the cardinal orientation of a given angle, where that angle is
    related to north at 0 degrees.

    Args:
        direction_angle (float):
            The angle to north in degrees (+Ve is interpreted as clockwise
            from north at 0.0 degrees).
        directions (int):
            The number of cardinal directions into which angles shall be
            binned (This value should be one of 4, 8, 16 or 32, and is centred
            about "north").

    Returns:
        int:
            The cardinal direction the angle represents.
    """

    if direction_angle > 360 or direction_angle < 0:
        raise ValueError(
            f"The angle entered ({direction_angle}) is beyond the normally expected range for an orientation in degrees."
        )

    cardinal_directions = {
        4: ["N", "E", "S", "W"],
        8: ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
        16: [
            "N",
            "NNE",
            "NE",
            "ENE",
            "E",
            "ESE",
            "SE",
            "SSE",
            "S",
            "SSW",
            "SW",
            "WSW",
            "W",
            "WNW",
            "NW",
            "NNW",
        ],
        32: [
            "N",
            "NbE",
            "NNE",
            "NEbN",
            "NE",
            "NEbE",
            "ENE",
            "EbN",
            "E",
            "EbS",
            "ESE",
            "SEbE",
            "SE",
            "SEbS",
            "SSE",
            "SbE",
            "S",
            "SbW",
            "SSW",
            "SWbS",
            "SW",
            "SWbW",
            "WSW",
            "WbS",
            "W",
            "WbN",
            "WNW",
            "NWbW",
            "NW",
            "NWbN",
            "NNW",
            "NbW",
        ],
    }

    if directions not in cardinal_directions:
        raise ValueError(
            f'The input "directions" must be one of {list(cardinal_directions.keys())}.'
        )

    val = int((direction_angle / (360 / directions)) + 0.5)

    arr = cardinal_directions[directions]

    return arr[(val % directions)]


def angle_from_north(vector: Vector3D) -> float:
    """For a 3D vector, determine the clockwise angle to north at [0, 1, 0].

    Args:
        vector (Vector3D):
            A ladybug_geometry Vector3D object.

    Returns:
        float:
            The angle between vector and north in degrees clockwise from [0, 1].
    """

    north = Vector2D(0, 1)
    vec2d = Vector2D(vector.x, vector.y)

    try:
        return np.rad2deg(north.angle_clockwise(vec2d))
    except ZeroDivisionError as exc:
        raise ValueError("The vector provided is vertical and does not allow for an angle to north to be calculated.") from exc


def face_orientation(face: Face3D) -> str:
    """Determine the cardinal orientation of a face.

    Args:
        face (Face3D):
            A ladybug_geometry Face3D object.

    Returns:
        str:
            The cardinal orientation of the face.
    """

    normal = face.normal
    angle = angle_from_north(normal)

    return cardinality(angle, 8)


def _log_message(property_name: str, value: Any, unit: str = "") -> None:
    logger.info(
        '> no "%s" provided, using default value of %s%s', property_name, value, unit
    )


def number_validator(value: float | int, prop_name: str, gt: float = None, ge: float = None, lt: float = None, le: float = None) -> None:
    """Validate a number against a set of constraints.

    Args:
        value (float | int): A number to validate.
        prop_name (str): The name of the property being validated.
        gt (float, optional): Value the number should be greater than. Defaults to None.
        ge (float, optional): Value the number should be greater than or equal to. Defaults to None.
        lt (float, optional): Value the number should be less than. Defaults to None.
        le (float, optional): Value the number should be less than or equal to. Defaults to None.
    """

    if not isinstance(value, (float, int)):
        raise ValueError(f"{prop_name} must be a number")
    if gt is not None and ge is not None:
        raise ValueError("Both 'gt' and 'ge' cannot be provided.")
    if lt is not None and le is not None:
        raise ValueError("Both 'lt' and 'le' cannot be provided.")
    if gt is not None and value <= gt:
        raise ValueError(f"{prop_name} must be greater than {gt}")
    if ge is not None and value < ge:
        raise ValueError(f"{prop_name} must be greater than or equal to {ge}")
    if lt is not None and value >= lt:
        raise ValueError(f"{prop_name} must be less than {lt}")
    if le is not None and value > le:
        raise ValueError(f"{prop_name} must be less than or equal to {le}")


def list_of_nums_validator(value: Any, prop_name: str, length: int, ge: float = None, le: float = None, gt: float = None, lt: float = None) -> None:
    """Validate a list of numbers against a set of constraints.

    Args:
        value (Any): The list of numbers to validate.
        prop_name (str): The name of the property being validated.
        length (int): The number of items the list should have.
        ge (float, optional): Value each number in the list should be greater than or equal to. Defaults to None.
        le (float, optional): Value each number in the list should be less than or equal to. Defaults to None.
        gt (float, optional): Value each number in the list should be greater than. Defaults to None.
        lt (float, optional): Value each number in the list should be less than. Defaults to None.

    """
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{prop_name} must be a list or tuple")
    if len(value) != length:
        raise ValueError(f"{prop_name} must have {length} items")
    for i in value:
        number_validator(i, prop_name, gt, ge, lt, le)

  
def estimate_sri_properties(
    target_sri: float, target_emittance: float = 0.85, tolerance: float = 5
) -> tuple[float, float]:
    """Estimate the solar absorptance and thermal emittance for a target SRI.

    This method uses a linear regression model to estimate the solar absorptance and
    thermal emittance of a material to achieve a target Solar Reflective Index (SRI).
    The model is trained on a dataset of these properties and the resultant SRI using
    fixed values for insolation, air_temperature, sky_temperature, and wind_speed.

    Args:
        target_sri: The target Solar Reflective Index (SRI) for the material.
        target_emittance: The target Thermal Emittance for the material. Default is 0.85.
        tolerance: The acceptable tolerance between resultant SRI and target SRI. Default is 5.

    Returns:
        tuple[float, float]: The estimated solar absorptance and thermal emittance.
    """

    if target_sri < 0 or target_sri > 122:
        raise ValueError("Target SRI must be between 0 and 122.")

    if target_emittance <= 0 or target_emittance >= 1:
        raise ValueError(
            "Thermal absorptivity estimation is beyond allowable limits for the target SRI."
        )

    model = LinearRegression()
    model.fit(
        SRI_DATA[["solar_absorptivity", "thermal_absorptivity"]].values, SRI_DATA["sri"].values
    )

    possible_combinations = []
    sris = []
    for sa in np.linspace(0, 1, 101):
        sri = model.predict([[sa, target_emittance]])[0]
        if np.isclose(sri, target_sri, atol=tolerance):
            possible_combinations.append(sa)
            sris.append(sri)
    sri_ = np.mean(sris)
    sa_ = np.mean(possible_combinations)
    ta_ = target_emittance

    logger.debug(
        (
            "Target SRI of %f±%f achieved (%.1f), using solar absorptance "
            "of %.3f and thermal emittance of %.3f"
        ),
        target_sri,
        tolerance,
        sri_,
        sa_,
        ta_,
    )

    if sa_ <= 0 or sa_ >= 1 or np.isnan(sa_):
        raise ValueError(
            (
                "Solar absorptivity estimation is beyond allowable limits "
                "for the target SRI."
            )
        )

    return sa_, ta_


def random_id(seed: int = None, length: int = 8) -> str:
    """Generate a random identifier.

    Args:
        seed (int): The seed for the random number generator.

    Returns:
        str: A random identifier.
    """

    alpha_num = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    
    np.random.seed(seed)

    return "".join(np.random.choice(alpha_num, size=length))

def calculate_sri(
    solar_reflectance: float,
    thermal_emittance: float,
    insolation: float = 1000,
    air_temperature: float = 36.85,
    sky_temperature: float = 26.85,
    wind_speed: float = 4,
) -> float:
    """Calculate the SRI of a material from its reflectance and emittance.
    Note, this method assumes a horizontal material (facing the sky).

    This method is based on the tool created by Ronnen Levinson, Heat Island
    Group, LBNL. It uses the method from ASTM Standard E1980-11 to calculate
    the SRI of a material, given its solar reflectance and thermal emittance.

    Wind speed is used inestead of wind convection coeffeicnt, based on the
    suggested values in ASTM E1980-11.

    Args:
        solar_reflectance (float): Solar reflectance of the material. Unitless,
            between 0 (black body) and 1 (white-body).
        thermal_emittance (float): Thermal emittance of the material. Unitless,
            between 0 (white-body) and 1 (black-body).
        insolation (float, optional): Insolation incident on the material.
            Defaults to 1000W/m2.
        air_temperature (float, optional): Air temperature. Defaults to 36.85C.
        sky_temperature (float, optional): Sky temperature. Defaults to 26.85C.
        wind_speed (float, optional): Speed of wind. Defaults to 4m/s.

    Returns:
        float: SRI of the material. Unitless.
    """

    if not 0 <= solar_reflectance <= 1:
        raise ValueError(
            "Solar reflectance must be between 0 (black body) and 1 (white-body)."
        )
    if not 0 <= thermal_emittance <= 1:
        raise ValueError(
            "Thermal emittance must be between 0 (white-body) and 1 (black-body)."
        )
    if wind_speed < 0:
        raise ValueError("Wind speed must be greater than 0.")

    # convert wind speed to wind convection coeffecient
    speeds = [1, 4, 8]
    coeffs = [5, 12, 30]
    wind_convection_coefficient = np.interp(wind_speed, speeds, coeffs)

    # set the sensitivity threshold for the iterative calculation. Lower is more accurate but slower.
    threshold = 0.5  # W
    increment = 0.01  # K
    iterations = 100000

    air_temperature = air_temperature + 273.15  # K
    sky_temperature = sky_temperature + 273.15  # K

    sigma = 5.67e-8  # W m-2 K-4 Stefan-Boltzmann constant
    blackbody_solar_reflectance = 0.05  #  0 - 1
    whitebody_solar_reflectance = 0.8  #  0 - 1
    blackbody_thermal_emittance = 0.9  #  0 - 1
    whitebody_thermal_emittance = 0.9  #  0 - 1

    surface_temperature = 200.0  # K
    n = 0
    while not np.isclose(
        (1 - solar_reflectance) * insolation
        - (
            thermal_emittance
            * sigma
            * (surface_temperature**4 - sky_temperature**4)
            + wind_convection_coefficient * (surface_temperature - air_temperature)
        ),
        0,
        atol=threshold,
    ):
        n += 1
        if n > iterations:
            raise ValueError(
                f"SRI calculation did not converge. Surface temperature of {surface_temperature - 273.15}C."
            )
        surface_temperature += increment

    blackbody_surface_temperature = 200.0  # K
    n = 0
    while not np.isclose(
        (1 - blackbody_solar_reflectance) * insolation
        - (
            blackbody_thermal_emittance
            * sigma
            * (blackbody_surface_temperature**4 - sky_temperature**4)
            + wind_convection_coefficient
            * (blackbody_surface_temperature - air_temperature)
        ),
        0,
        atol=threshold,
    ):
        n += 1
        if n > iterations:
            raise ValueError("SRI calculation did not converge.")
        blackbody_surface_temperature += increment

    whitebody_surface_temperature = 200.0  # K
    n = 0
    while not np.isclose(
        (1 - whitebody_solar_reflectance) * insolation
        - (
            whitebody_thermal_emittance
            * sigma
            * (whitebody_surface_temperature**4 - sky_temperature**4)
            + wind_convection_coefficient
            * (whitebody_surface_temperature - air_temperature)
        ),
        0,
        atol=threshold,
    ):
        n += 1
        if n > iterations:
            raise ValueError("SRI calculation did not converge.")
        whitebody_surface_temperature += increment

    solar_reflective_index = (
        100
        * (blackbody_surface_temperature - surface_temperature)
        / (blackbody_surface_temperature - whitebody_surface_temperature)
    )

    if solar_reflective_index < 0:
        return 0

    return solar_reflective_index


def material_sri(
    material: EnergyMaterial,
    insolation: float = 1000,
    air_temperature: float = 36.85,
    sky_temperature: float = 26.85,
    wind_speed: float = 4,
) -> float:
    """Calculate the SRI of a Honeybee material.
    Note, this method assumes a horizontal material (facing the sky).

    Args:
        material (_EnergyMaterialOpaqueBase): A Honeybee opaque material.
        insolation (float, optional): Insolation incident on the material.
            Defaults to 1000W/m2.
        air_temperature (float, optional): Air temperature. Defaults to 36.85C.
        sky_temperature (float, optional): Sky temperature. Defaults to 26.85C.
        wind_speed (float, optional): Speed of wind. Defaults to 4m/s.

    Returns:
        float: SRI of the material. Unitless.
    """

    return calculate_sri(
        solar_reflectance=material.solar_reflectance,
        thermal_emittance=material.thermal_absorptance,
        insolation=insolation,
        air_temperature=air_temperature,
        sky_temperature=sky_temperature,
        wind_speed=wind_speed,
    )


def construction_sri(
    construction: OpaqueConstruction,
    insolation: float = 1000,
    air_temperature: float = 36.85,
    sky_temperature: float = 26.85,
    wind_speed: float = 4,
) -> float:
    """Calculate the SRI of a Honeybee construction.
    Note, this method assumes a horizontal construction (facing the sky).

    Args:
        construction (Opaqueconstruction): A Honeybee construction material.
        insolation (float, optional): Insolation incident on the material.
            Defaults to 1000W/m2.
        air_temperature (float, optional): Air temperature. Defaults to 36.85C.
        sky_temperature (float, optional): Sky temperature. Defaults to 26.85C.
        wind_speed (float, optional): Speed of wind. Defaults to 4m/s.

    Returns:
        float: SRI of the construction. Unitless.
    """

    return calculate_sri(
        solar_reflectance=construction.outside_solar_reflectance,
        thermal_emittance=construction.outside_emissivity,
        insolation=insolation,
        air_temperature=air_temperature,
        sky_temperature=sky_temperature,
        wind_speed=wind_speed,
    )