# region: IMPORTS
# pylint: disable=E0401


import calendar
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from honeybee.model import Model, Room
from honeybee_energy.construction.opaque import OpaqueConstruction
from honeybee_energy.hvac.idealair import IdealAirSystem
from honeybee_energy.material.opaque import EnergyMaterial
from honeybee_energy.programtype import ProgramType
from honeybee_energy.result.eui import eui_from_sql
from honeybee_energy.result.loadbalance import LoadBalance
from ladybug.datacollection import (AnalysisPeriod, BaseCollection, Header,
                                    HourlyContinuousCollection,
                                    MonthlyCollection)
from ladybug.datatype import TYPESDICT
from ladybug.datatype.fraction import Fraction
from ladybug.datatype.generic import GenericType
from ladybug.sql import SQLiteResult
from ladybug_geometry.geometry2d import Vector2D
from ladybug_geometry.geometry3d import Face3D, Vector3D
from matplotlib.colors import colorConverter
from sklearn.linear_model import LinearRegression

from .config import INDEX, SRI_DATA, colour_defaults, logger

# pylint: enable=E0401
# endregion: IMPORTS

# CUSTOM DATA TYPES #
Occupants = GenericType(name="Occupants", unit="people", min=0, abbreviation="Occ")
OccupantDensity = GenericType(
    name="Occupant Density", unit="people/m2", min=0, abbreviation="Occ/m2"
)


def get_color(variable: str) -> str:
    """Return the color hex-code for a given variable. This assumes that the vairable is in the format Variable (Unit)."""

    default_color = "magenta"
    try:
        return colour_defaults[variable.split(" (")[0]]
    except KeyError:
        try:
            return colour_defaults[variable]
        except KeyError:
            logger.warning(
                'No default color available for "%s". Using "%s".',
                variable,
                default_color,
            )
    return default_color


def get_unit(variable: str) -> str | None:
    """Return the unit for a given variable. This assumes that the vairable is in the format Variable (Unit)."""

    default_unit = None
    try:
        return variable.split(" (")[1].split(")")[0]
    except IndexError:
        logger.warning(
            'No unit available for "%s". Using "%s".', variable, default_unit
        )
        return default_unit


def consistent_units(variables: list[str]) -> None:
    """Determine whether all variables share the same units for plotting."""

    try:
        units = set(var.split(" (")[1].split(")")[0] for var in variables)
    except IndexError:
        return None

    if len(units) > 1:
        raise ValueError(
            f"All variables must have the same units for plotting. Units found: {units}"
        )


def relative_luminance(color: Any):
    """Calculate the relative luminance of a color according to W3C standards

    Args:
        color (Any):
            matplotlib color or sequence of matplotlib colors - Hex code,
            rgb-tuple, or html color name.

    Returns:
        float:
            Luminance value between 0 and 1.
    """
    rgb = colorConverter.to_rgba_array(color)[:, :3]
    rgb = np.where(rgb <= 0.03928, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    lum = rgb.dot([0.2126, 0.7152, 0.0722])
    try:
        return lum.item()
    except ValueError:
        return lum


def contrasting_color(color: Any):
    """Calculate the contrasting color for a given color.

    Args:
        color (Any):
            matplotlib color or sequence of matplotlib colors - Hex code,
            rgb-tuple, or html color name.

    Returns:
        str:
            String code of the contrasting color.
    """
    return ".15" if relative_luminance(color) > 0.408 else "w"


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
        raise ValueError(
            "The vector provided is vertical and does not allow for an angle to north to be calculated."
        ) from exc


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


def number_validator(
    value: float | int,
    prop_name: str,
    gt: float = None,
    ge: float = None,
    lt: float = None,
    le: float = None,
) -> None:
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


def list_of_nums_validator(
    value: Any,
    prop_name: str,
    length: int,
    ge: float = None,
    le: float = None,
    gt: float = None,
    lt: float = None,
) -> None:
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
        SRI_DATA[["solar_absorptivity", "thermal_absorptivity"]].values,
        SRI_DATA["sri"].values,
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

    alpha_num = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

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
            thermal_emittance * sigma * (surface_temperature**4 - sky_temperature**4)
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


def describe_analysis_period(
    analysis_period: list[AnalysisPeriod],
) -> str:
    """Create a description of the given analysis period.

    Args:
        analysis_period (AnalysisPeriod):
            A Ladybug analysis period.

    Returns:
        str:
            A description of the analysis period.
    """

    if not isinstance(analysis_period, AnalysisPeriod):
        raise ValueError("Analysis period must be a Ladybug AnalysisPeriod object.")

    base_str = (
        f"{calendar.month_abbr[analysis_period.st_month]} {analysis_period.st_day:02} to "
        f"{calendar.month_abbr[analysis_period.end_month]} {analysis_period.end_day:02} between "
        f"{analysis_period.st_hour:02}:00 and {analysis_period.end_hour:02}:59"
    )
    base_str = "".join(base_str)

    return base_str


def header_to_string(header: Header) -> str:
    """Convert a Ladybug header object into a string.

    Args:
        header (Header):
            A Ladybug header object.

    Returns:
        str:
            A Ladybug header string."""

    return f"{header.data_type} ({header.unit})"


def collection_to_series(
    collection: HourlyContinuousCollection, name: str = None
) -> pd.Series:
    """Convert a Ladybug hourlyContinuousCollection object into a Pandas Series object.

    Args:
        collection (BaseCollection):
            Ladybug data collection object.
        name (str, optional):
            The name of the resulting Pandas Series object. Defaults to None,
            which uses the collection datatype.

    Returns:
        pd.Series:
            A Pandas Series object.
    """

    index = pd.to_datetime(collection.header.analysis_period.datetimes)
    if len(collection.values) == 12:
        index = pd.date_range(f"{index[0].year}-01-01", periods=12, freq="MS")

    return pd.Series(
        data=collection.values,
        index=index,
        name=header_to_string(collection.header) if not name else name,
    )


def header_from_string(string: str, is_leap_year: bool = False) -> Header:
    """Convert a string into a Ladybug header object.

    Args:
        string (str):
            A Ladybug header string.
        is_leap_year (bool, optional):
            A boolean to indicate whether the header is for a leap year. Default is False.

    Returns:
        Header:
            A Ladybug header object."""

    str_elements = string.split(" ")

    if (len(str_elements) < 2) or ("(" not in string) or (")" not in string):
        raise ValueError(
            "The string to be converted into a LB Header must be in the format 'variable (unit)'"
        )

    str_elements = string.split(" ")
    unit = str_elements[-1].replace("(", "").replace(")", "")
    data_type = " ".join(str_elements[:-1])

    try:
        data_type = TYPESDICT[data_type.replace(" ", "")]()
    except KeyError:
        data_type = GenericType(name=data_type, unit=unit)

    return Header(
        data_type=data_type,
        unit=unit,
        analysis_period=AnalysisPeriod(is_leap_year=is_leap_year),
    )


def collection_from_series(series: pd.Series) -> BaseCollection:
    """Convert a Pandas Series object into a Ladybug BaseCollection-like object.

    Args:
        series (pd.Series): A Pandas Series object.

    Returns:
        BaseCollection: A Ladybug BaseCollection-like object.
    """

    header = header_from_string(
        series.name, is_leap_year=series.index.is_leap_year.any()
    )
    header.metadata["source"] = "From custom pd.Series"

    freq = pd.infer_freq(series.index)
    if freq in ["H", "h"]:
        if series.index.is_leap_year.any():
            if len(series.index) != 8784:
                raise ValueError(
                    "The number of values in the series must be 8784 for leap years."
                )
        else:
            if len(series.index) != 8760:
                raise ValueError("The series must have 8760 rows for non-leap years.")

        return HourlyContinuousCollection(
            header=header,
            values=series.values,
        )

    if freq in ["M", "MS"]:
        if len(series.index) != 12:
            raise ValueError("The series must have 12 rows for months.")

        return MonthlyCollection(
            header=header,
            values=series.values.tolist(),
            datetimes=range(1, 13),
        )

    raise ValueError("The series must be hourly or monthly.")


def simulate_model(
    model: Model, epw_file: Path, simulation_directory: Path = None
) -> Path:
    """Given a Honeybee model and an EPW file, simulate the model and return the simulation results.

    Args:
        model (Model): A Honeybee model.
        epw_file (Path): Path to an EPW file.
        simulation_directory (Path, optional): Directory to save the simulation results. Defaults to None.

    Returns:
        Path: The sql file resulting from the simulation.
    """

    raise NotImplementedError("Not yet implemented")


def aggregate_collection(
    collections: list[HourlyContinuousCollection], agg: str
) -> HourlyContinuousCollection:
    """Apply an aggregation to a set of hourly continuous collections.

    Args:
        collections (list[HourlyContinuousCollection]): A list of hourly continuous collections.
        agg (str): The aggregation method to apply. One of ["mean", "sum", "max", "min", "median"].

    Returns:
        HourlyContinuousCollection: The aggregated collection.
    """

    if not isinstance(collections, (list, tuple)):
        raise ValueError(
            "Collections must be an enumerable of HourlyContinuousCollection objects."
        )

    if len(collections) == 0:
        raise ValueError("No collections provided for aggregation.")

    if len(collections) == 1:
        return collections[0]

    if not len(set(col.header.unit for col in collections)) == 1:
        raise ValueError("All collections must have the same datatype for aggregation.")

    if not len(set(len(col) for col in collections)) == 1:
        raise ValueError("All collections must have the same length for aggregation.")

    # check collections are aligned
    df = pd.concat([collection_to_series(col) for col in collections], axis=1)

    return collection_from_series(df.agg(agg, axis=1).rename(df.columns[0]))


def subtract_loss_from_gain(gain_load, loss_load):
    """Create a single DataCollection from gains and losses."""
    total_loads = []
    for gain, loss in zip(gain_load, loss_load):
        total_load = gain - loss
        total_load.header.metadata["type"] = total_load.header.metadata["type"].replace(
            "Gain ", ""
        )
        total_loads.append(total_load)
    return total_loads


def room_energy_result(_sql: Path) -> list[list[HourlyContinuousCollection]]:
    """A duplicate of the method used in Grasshopper to get the data collections necessary for a Load Balance calculation.

    Args:
        _sql (Path): The path to the SQL file containing Honeybee Energy simulation results.

    Returns:
        list[list[HourlyContinuousCollections]]: A set of results.

    Reference:
        https://github.com/ladybug-tools/honeybee-grasshopper-energy/blob/master/honeybee_grasshopper_energy/src/HB%20Read%20Room%20Energy%20Result.py

    Notes:
        Returned collections include:
        cooling: DataCollections for the cooling energy in kWh. For Ideal Air
            loads, this output is the sum of sensible and latent heat that must
            be removed from each room.  For detailed HVAC systems, this output
            will be electric energy needed to power each chiller/cooling coil.
        heating: DataCollections for the heating energy needed in kWh. For Ideal
            Air loads, this is the heat that must be added to each room.  For
            detailed HVAC systems, this will be fuel energy or electric energy
            needed for each boiler/heating element.
        lighting: DataCollections for the electric lighting energy used for
            each room in kWh.
        electric_equip: DataCollections for the electric equipment energy used
            for each room in kWh.
        gas_equip: DataCollections for the gas equipment energy used for each
            room in kWh.
        process: DataCollections for the process load energy used for each
            room in kWh.
        hot_water: DataCollections for the service hote water energy used for each
            room in kWh.
        fan_electric: DataCollections for the fan electric energy in kWh for
            either a ventilation fan or a HVAC system fan.
        pump_electric: DataCollections for the water pump electric energy in kWh
            for a heating/cooling system.
        people_gain: DataCollections for the internal heat gains in each room
            resulting from people (kWh).
        solar_gain: DataCollections for the total solar gain in each room (kWh).
        infiltration_load: DataCollections for the heat loss (negative) or heat
            gain (positive) in each room resulting from infiltration (kWh).
        mech_vent_load: DataCollections for the heat loss (negative) or heat gain
            (positive) in each room resulting from the outdoor air coming through
            the HVAC System (kWh).
        nat_vent_load: DataCollections for the heat loss (negative) or heat gain
            (positive) in each room resulting from natural ventilation (kWh).
    """

    _sql = Path(_sql).absolute().as_posix()

    # List of all the output strings that will be requested
    cooling_outputs = LoadBalance.COOLING + (
        "Cooling Coil Electricity Energy",
        "Chiller Electricity Energy",
        "Zone VRF Air Terminal Cooling Electricity Energy",
        "VRF Heat Pump Cooling Electricity Energy",
        "Chiller Heater System Cooling Electricity Energy",
        "District Cooling Water Energy",
        "Evaporative Cooler Electricity Energy",
    )
    heating_outputs = LoadBalance.HEATING + (
        "Boiler NaturalGas Energy",
        "Heating Coil Total Heating Energy",
        "Heating Coil NaturalGas Energy",
        "Heating Coil Electricity Energy",
        "Humidifier Electricity Energy",
        "Zone VRF Air Terminal Heating Electricity Energy",
        "VRF Heat Pump Heating Electricity Energy",
        "VRF Heat Pump Defrost Electricity Energy",
        "VRF Heat Pump Crankcase Heater Electricity Energy",
        "Chiller Heater System Heating Electricity Energy",
        "District Heating Water Energy",
        "Baseboard Electricity Energy",
        "Hot_Water_Loop_Central_Air_Source_Heat_Pump Electricity Consumption",
        "Boiler Electricity Energy",
        "Water Heater NaturalGas Energy",
        "Water Heater Electricity Energy",
        "Cooling Coil Water Heating Electricity Energy",
    )
    lighting_outputs = LoadBalance.LIGHTING
    electric_equip_outputs = LoadBalance.ELECTRIC_EQUIP
    gas_equip_outputs = LoadBalance.GAS_EQUIP
    process_outputs = LoadBalance.PROCESS
    shw_outputs = ("Water Use Equipment Heating Energy",) + LoadBalance.HOT_WATER
    fan_electric_outputs = (
        "Zone Ventilation Fan Electricity Energy",
        "Fan Electricity Energy",
        "Cooling Tower Fan Electricity Energy",
    )
    pump_electric_outputs = "Pump Electricity Energy"
    people_gain_outputs = LoadBalance.PEOPLE_GAIN
    solar_gain_outputs = LoadBalance.SOLAR_GAIN
    infil_gain_outputs = LoadBalance.INFIL_GAIN
    infil_loss_outputs = LoadBalance.INFIL_LOSS
    vent_loss_outputs = LoadBalance.VENT_LOSS
    vent_gain_outputs = LoadBalance.VENT_GAIN
    nat_vent_gain_outputs = LoadBalance.NAT_VENT_GAIN
    nat_vent_loss_outputs = LoadBalance.NAT_VENT_LOSS
    all_output = [
        cooling_outputs,
        heating_outputs,
        lighting_outputs,
        electric_equip_outputs,
        gas_equip_outputs,
        process_outputs,
        shw_outputs,
        fan_electric_outputs,
        pump_electric_outputs,
        people_gain_outputs,
        solar_gain_outputs,
        infil_gain_outputs,
        infil_loss_outputs,
        vent_loss_outputs,
        vent_gain_outputs,
        nat_vent_gain_outputs,
        nat_vent_loss_outputs,
    ]

    sql_obj = SQLiteResult(_sql)

    # get all of the results relevant for energy use
    cooling = sql_obj.data_collections_by_output_name(cooling_outputs)
    heating = sql_obj.data_collections_by_output_name(heating_outputs)
    lighting = sql_obj.data_collections_by_output_name(lighting_outputs)
    electric_equip = sql_obj.data_collections_by_output_name(electric_equip_outputs)
    hot_water = sql_obj.data_collections_by_output_name(shw_outputs)
    gas_equip = sql_obj.data_collections_by_output_name(gas_equip_outputs)
    process = sql_obj.data_collections_by_output_name(process_outputs)
    fan_electric = sql_obj.data_collections_by_output_name(fan_electric_outputs)
    pump_electric = sql_obj.data_collections_by_output_name(pump_electric_outputs)

    # get all of the results relevant for gains and losses
    people_gain = sql_obj.data_collections_by_output_name(people_gain_outputs)
    solar_gain = sql_obj.data_collections_by_output_name(solar_gain_outputs)
    infil_gain = sql_obj.data_collections_by_output_name(infil_gain_outputs)
    infil_loss = sql_obj.data_collections_by_output_name(infil_loss_outputs)
    vent_loss = sql_obj.data_collections_by_output_name(vent_loss_outputs)
    vent_gain = sql_obj.data_collections_by_output_name(vent_gain_outputs)
    nat_vent_gain = sql_obj.data_collections_by_output_name(nat_vent_gain_outputs)
    nat_vent_loss = sql_obj.data_collections_by_output_name(nat_vent_loss_outputs)

    # do arithmetic with any of the gain/loss data collections
    infiltration_load = []
    if len(infil_gain) == len(infil_loss):
        infiltration_load = subtract_loss_from_gain(infil_gain, infil_loss)
    mech_vent_load = []
    if len(vent_gain) == len(vent_loss) == len(cooling) == len(heating):
        mech_vent_loss = subtract_loss_from_gain(heating, vent_loss)
        mech_vent_gain = subtract_loss_from_gain(cooling, vent_gain)
        mech_vent_load = [
            data.duplicate()
            for data in subtract_loss_from_gain(mech_vent_gain, mech_vent_loss)
        ]
        for load in mech_vent_load:
            load.header.metadata["type"] = "Zone Ideal Loads Ventilation Heat Energy"
    nat_vent_load = []
    if len(nat_vent_gain) == len(nat_vent_loss):
        nat_vent_load = subtract_loss_from_gain(nat_vent_gain, nat_vent_loss)

    # remove the district hot water system used for service hot water from space heating
    shw_equip, distr_i = [], None
    for i, heat in enumerate(heating):
        if not isinstance(heat, float):
            try:
                heat_equip = heat.header.metadata["System"]
                if heat_equip.startswith("SHW"):
                    shw_equip.append(i)
                elif heat_equip == "SERVICE HOT WATER DISTRICT HEAT":
                    distr_i = i
            except KeyError:
                pass
    if len(shw_equip) != 0 and distr_i is None:
        hot_water = [heating.pop(i) for i in reversed(shw_equip)]
    elif distr_i is not None:
        for i in reversed(shw_equip + [distr_i]):
            heating.pop(i)

    return [
        cooling,
        heating,
        lighting,
        electric_equip,
        gas_equip,
        process,
        hot_water,
        fan_electric,
        pump_electric,
        people_gain,
        solar_gain,
        infiltration_load,
        mech_vent_load,
        nat_vent_load,
    ]


def face_result(_sql: Path) -> list[HourlyContinuousCollection]:
    """A duplicate of the method used in Grasshopper to get the data collections necessary for a Load Balance calculation.

    Args:
        _sql (Path): The path to the SQL file containing Honeybee Energy simulation results.

    Returns:
        list[HourlyContinuousCollection]: A set of results.

    Reference:
        https://github.com/ladybug-tools/honeybee-grasshopper-energy/blob/master/honeybee_grasshopper_energy/src/HB%20Read%20Face%20Result.py

    Notes:
        Returned collections include:
        face_indoor_temp: DataCollections for the indoor face temperature of each face.
        face_outdoor_temp: DataCollections for the outdoor face temperature of each face.
        face_energy_flow: DataCollections for the energy flow through each face.
    """

    _sql = Path(_sql).absolute().as_posix()

    # List of all the output strings that will be requested
    face_indoor_temp_output = "Surface Inside Face Temperature"
    face_outdoor_temp_output = "Surface Outside Face Temperature"
    opaque_energy_flow_output = "Surface Inside Face Conduction Heat Transfer Energy"
    window_loss_output = "Surface Window Heat Loss Energy"
    window_gain_output = "Surface Window Heat Gain Energy"
    all_output = [
        face_indoor_temp_output,
        face_outdoor_temp_output,
        opaque_energy_flow_output,
        window_loss_output,
        window_gain_output,
    ]

    def ironpython_results(sql_file):
        sql_obj = SQLiteResult(sql_file)  # create the SQL result parsing object
        # get all of the results
        face_indoor_temp = sql_obj.data_collections_by_output_name(
            face_indoor_temp_output
        )
        face_outdoor_temp = sql_obj.data_collections_by_output_name(
            face_outdoor_temp_output
        )
        opaque_energy_flow = sql_obj.data_collections_by_output_name(
            opaque_energy_flow_output
        )
        window_loss = sql_obj.data_collections_by_output_name(window_loss_output)
        window_gain = sql_obj.data_collections_by_output_name(window_gain_output)
        return (
            face_indoor_temp,
            face_outdoor_temp,
            opaque_energy_flow,
            window_loss,
            window_gain,
        )

    (
        face_indoor_temp,
        face_outdoor_temp,
        opaque_energy_flow,
        window_loss,
        window_gain,
    ) = ironpython_results(_sql)

    # do arithmetic with any of the gain/loss data collections
    window_energy_flow = []
    if len(window_gain) == len(window_loss):
        window_energy_flow = subtract_loss_from_gain(window_gain, window_loss)
    face_energy_flow = opaque_energy_flow + window_energy_flow

    return face_indoor_temp, face_outdoor_temp, face_energy_flow


def room_comfort_result(_sql: Path) -> list[HourlyContinuousCollection]:
    """A duplicate of the method used in Grasshopper to get the data collections necessary for thermal comfort assessment.

    Args:
        _sql (Path): The path to the SQL file containing Honeybee Energy simulation results.

    Returns:
        list[HourlyContinuousCollection]: A set of results.

    Reference:
        https://github.com/ladybug-tools/honeybee-grasshopper-energy/blob/master/honeybee_grasshopper_energy/src/HB%20Read%20Room%20Comfort%20Result.py

    Notes:
        oper_temp: DataCollections for the operative temperature of each zone.
        air_temp: DataCollections for the air temperature of each zone.
        rad_temp: DataCollections for the radiant temperature of each zone.
        rel_humidity: DataCollections for the relative humidity of each zone.
        unmet_heat: DataCollections for the unmet heating setpoint time of each zone.
        unmet_cool: DataCollections for the unmet cooling setpoint time of each zone.
    """

    _sql = Path(_sql).absolute().as_posix()
    sql_obj = SQLiteResult(_sql)

    # List of all the output strings that will be requested
    oper_temp_output = "Zone Operative Temperature"
    air_temp_output = "Zone Mean Air Temperature"
    rad_temp_output = "Zone Mean Radiant Temperature"
    rel_humidity_output = "Zone Air Relative Humidity"
    heat_setpt_output = "Zone Heating Setpoint Not Met Time"
    cool_setpt_output = "Zone Cooling Setpoint Not Met Time"
    all_output = [
        oper_temp_output,
        air_temp_output,
        rad_temp_output,
        rel_humidity_output,
        heat_setpt_output,
        cool_setpt_output,
    ]

    # get all of the results
    oper_temp = sql_obj.data_collections_by_output_name(oper_temp_output)
    air_temp = sql_obj.data_collections_by_output_name(air_temp_output)
    rad_temp = sql_obj.data_collections_by_output_name(rad_temp_output)
    rel_humidity = sql_obj.data_collections_by_output_name(rel_humidity_output)
    unmet_heat = sql_obj.data_collections_by_output_name(heat_setpt_output)
    unmet_cool = sql_obj.data_collections_by_output_name(cool_setpt_output)

    return oper_temp, air_temp, rad_temp, rel_humidity, unmet_heat, unmet_cool


def load_balance(
    _rooms_model: list[Room] | Model,
    cooling_: list[HourlyContinuousCollection],
    heating_: list[HourlyContinuousCollection],
    lighting_: list[HourlyContinuousCollection],
    electric_equip_: list[HourlyContinuousCollection],
    gas_equip_: list[HourlyContinuousCollection],
    process_: list[HourlyContinuousCollection],
    hot_water_: list[HourlyContinuousCollection],
    people_gain_: list[HourlyContinuousCollection],
    solar_gain_: list[HourlyContinuousCollection],
    infiltration_load_: list[HourlyContinuousCollection],
    mech_vent_load_: list[HourlyContinuousCollection],
    nat_vent_load_: list[HourlyContinuousCollection],
    face_energy_flow_: list[HourlyContinuousCollection],
) -> list[list[HourlyContinuousCollection]]:
    """A method to calculate the load balance of a building model, replicating the code used in the Grasshopper component.

    Args:
        _rooms_model (list[Room] | Model):
            Either a list of Rooms or a Model object containing the rooms.
        cooling_ (list[HourlyContinuousCollection]):
            A set of data collections for cooling energy.
        heating_ (list[HourlyContinuousCollection]):
            A set of data collections for heating energy.
        lighting_ (list[HourlyContinuousCollection]):
            A set of data collections for lighting energy.
        electric_equip_ (list[HourlyContinuousCollection]):
            A set of data collections for electric equipment energy.
        gas_equip_ (list[HourlyContinuousCollection]):
            A set of data collections for gas equipment energy.
        process_ (list[HourlyContinuousCollection]):
            A set of data collections for process energy.
        hot_water_ (list[HourlyContinuousCollection]):
            A set of data collections for hot water energy.
        people_gain_ (list[HourlyContinuousCollection]):
            A set of data collections for people gain energy.
        solar_gain_ (list[HourlyContinuousCollection]):
            A set of data collections for solar gain energy.
        infiltration_load_ (list[HourlyContinuousCollection]):
            A set of data collections for infiltration load energy.
        mech_vent_load_ (list[HourlyContinuousCollection]):
            A set of data collections for mechanical ventilation load energy.
        nat_vent_load_ (list[HourlyContinuousCollection]):
            A set of data collections for natural ventilation load energy.
        face_energy_flow_ (list[HourlyContinuousCollection]):
            A set of data collections for face energy flow.

    Returns:
        balance (list[HourlyContinuousCollection]): A set of data collections containing the load balance results.
        balance_stor (list[HourlyContinuousCollection]): A set of data collections containing the load balance results, including thermal mass.
        norm_bal (list[HourlyContinuousCollection]): A set of data collections containing the normalized load balance results.
        norm_bal_stor (list[HourlyContinuousCollection]): A set of data collections containing the normalized load balance results, including thermal mass.
    """

    def check_input(input_list):
        """Check that an input isn't a zero-length list or None."""
        return None if len(input_list) == 0 or input_list[0] is None else input_list

    # extract any rooms from input Models
    is_model, floor_area = False, 0
    rooms = []
    for hb_obj in _rooms_model:
        if isinstance(hb_obj, Model):
            rooms.extend(hb_obj.rooms)
            is_model = True
            floor_area += hb_obj.floor_area
        else:
            rooms.append(hb_obj)

    # if a detailed HVAC system is assigned to the rooms, give a warning
    bad_rooms = []
    for room in rooms:
        hvac = room.properties.energy.hvac
        if hvac is not None and not isinstance(hvac, (IdealAirSystem)):
            bad_rooms.append(room.display_name)
    if len(bad_rooms) != 0:
        if len(bad_rooms) > 20:
            bad_rooms = bad_rooms[:20] + ["..."]
        msg = (
            "The following Rooms use HVAC systems other than Ideal Air.\n"
            "The cooling and heating results for detailed HVAC are electricity and\n"
            "fuel, which cannot be used in load balances of thermal energy.\n"
            "Either replace these detailed HVAC systems with Ideal Air Systems or \n"
            'use the "HB Annual Loads" component to get a load balance these\n'
            "roomswith detailed HVAC:\n\n{}".format("\n".join(bad_rooms))
        )
        logger.warning(msg)

    # if the input is for individual rooms, check the solar to ensure no groued zones
    if not is_model and len(solar_gain_) != 0:
        msg = (
            "Air boundaries with grouped zones detected in solar data but individual "
            "rooms were input.\nIt is recommended that the full model be input for "
            "_rooms_model to ensure correct representaiton of solar."
        )
        for coll in solar_gain_:
            if "Solar Enclosure" in coll.header.metadata["Zone"]:
                logger.warning(msg)

    # process all of the inputs
    cooling_ = check_input(cooling_)
    heating_ = check_input(heating_)
    lighting_ = check_input(lighting_)
    electric_equip_ = check_input(electric_equip_)
    gas_equip_ = check_input(gas_equip_)
    process_ = check_input(process_)
    hot_water_ = check_input(hot_water_)
    people_gain_ = check_input(people_gain_)
    solar_gain_ = check_input(solar_gain_)
    infiltration_load_ = check_input(infiltration_load_)
    mech_vent_load_ = check_input(mech_vent_load_)
    nat_vent_load_ = check_input(nat_vent_load_)
    face_energy_flow_ = check_input(face_energy_flow_)

    # process hot water to ensure it's the correct type
    hw_type = "Water Use Equipment Heating Energy"
    if hot_water_ is not None and hot_water_[0].header.metadata["type"] == hw_type:
        hot_water_ = [hw.duplicate() * 0.25 for hw in hot_water_]
        for hw in hot_water_:
            hw.header.metadata = {
                "type": "Water Use Equipment Zone Sensible Heat Gain Energy",
                "System": hw.header.metadata["System"],
            }

    # construct the load balance object and output the results
    load_bal_obj = LoadBalance(
        rooms,
        cooling_,
        heating_,
        lighting_,
        electric_equip_,
        gas_equip_,
        process_,
        hot_water_,
        people_gain_,
        solar_gain_,
        infiltration_load_,
        mech_vent_load_,
        nat_vent_load_,
        face_energy_flow_,
        "Meters",
        use_all_solar=is_model,
    )
    if is_model:
        load_bal_obj.floor_area = floor_area

    balance = load_bal_obj.load_balance_terms(False, False)
    balance_stor = []
    norm_bal = []
    norm_bal_stor = []
    if len(balance) != 0:
        balance_stor = balance + [load_bal_obj.storage]
        norm_bal = load_bal_obj.load_balance_terms(True, False)
        norm_bal_stor = load_bal_obj.load_balance_terms(True, True)

    return balance, balance_stor, norm_bal, norm_bal_stor


def load_balance_from_sql(_rooms_model: Model, _sql: Path, as_dataframe: bool = True) -> list[HourlyContinuousCollection] | pd.DataFrame:
    
    # get the rooms results
    (
        cooling, 
        heating, 
        lighting, 
        electric_equip, 
        gas_equip, 
        process, 
        hot_water, 
        fan_electric, 
        pump_electric, 
        people_gain, 
        solar_gain, 
        infiltration_load, 
        mech_vent_load, 
        nat_vent_load,
    ) = room_energy_result(_sql)

    # get face results
    (
        face_indoor_temp, 
        face_outdoor_temp, 
        face_energy_flow, 
    ) = face_result(_sql=_sql)
    
    # run the load balance
    (
        balance, 
        balance_stor, 
        norm_bal, 
        norm_bal_stor, 
    ) = load_balance(
        _rooms_model=_rooms_model,
        cooling_=cooling,
        heating_=heating,
        lighting_=lighting,
        electric_equip_=electric_equip,
        gas_equip_=gas_equip,
        process_=process,
        hot_water_=hot_water,
        people_gain_=people_gain,
        solar_gain_=solar_gain,
        infiltration_load_=infiltration_load,
        mech_vent_load_=mech_vent_load,
        nat_vent_load_=nat_vent_load,
        face_energy_flow_=face_energy_flow,
    )

    if as_dataframe:
        return pd.concat([collection_to_series(col) for col in balance], axis=1)
    
    return (
        balance, 
        balance_stor, 
        norm_bal, 
        norm_bal_stor, 
    )


def annual_eui(_sql) -> pd.Series:
    """Get the EUI from a SQL file."""
    results = eui_from_sql(_sql)
    _, _, end_use_pairs = (
        results["eui"],
        results["total_floor_area"],
        results["end_uses"],
    )
    eui_end_use = end_use_pairs.values()
    end_uses = [use.replace("_", " ").title() for use in end_use_pairs.keys()]
    return pd.Series(data=eui_end_use, index=end_uses, name="EUI (kWh/m2)")


def suspendlogging(func):
    @wraps(func)
    def inner(*args, **kwargs):
        previousloglevel = logger.getEffectiveLevel()
        try:
            return func(*args, **kwargs)
        finally:
            logger.setLevel(previousloglevel)

    return inner


def occupancy_schedule_from_program(program: ProgramType) -> HourlyContinuousCollection:
    """Return the occupancy schedule from a program, giving a collection of
    zeros if no occupancy is found.

    Args:
        program (ProgramType): A honeybee_energy program type.

    Returns:
        HourlyContinuousCollection: The occupancy schedule.
    """
    if program.people is not None:
        return program.people.occupancy_schedule.data_collection()

    return HourlyContinuousCollection(
        header=Header(
            data_type=Fraction(),
            unit="fraction",
            analysis_period=AnalysisPeriod(),
            metadata={"schedule": "Building_People_Occ Schedule"},
        ),
        values=list(np.zeros(8760)),
    )


def occupancy_from_program(
    program: ProgramType, area: float = None
) -> HourlyContinuousCollection:
    """Return the occupancy from a program, giving a value of 0 if no occupancy is found.

    Args:
        program (ProgramType): A honeybee_energy program type.
        area (float, optional): The area of the program. Defaults to None which returns people/m2.

    Returns:
        HourlyContinuousCollection: The occupancy schedule, either in people or people/m2.
    """

    if area is not None:
        if area <= 0:
            raise ValueError("area must be greater than 0.")

    # get the occupancy schedule
    schedule = occupancy_schedule_from_program(program=program)

    if (schedule.total > 0) and (program.people.people_per_area > 0):
        # calculate the occupancy from the schedule and the people per area
        values = np.array((schedule * program.people.people_per_area).values)
    else:
        values = np.array(schedule.values)

    if area is not None:
        # calculate the occupancy from the schedule and the area
        values = values * area
        header = Header(
            data_type=Occupants,
            unit="people",
            analysis_period=AnalysisPeriod(),
        )
    else:
        header = Header(
            data_type=OccupantDensity,
            unit="people/m2",
            analysis_period=AnalysisPeriod(),
        )

    return HourlyContinuousCollection(header=header, values=list(values))
