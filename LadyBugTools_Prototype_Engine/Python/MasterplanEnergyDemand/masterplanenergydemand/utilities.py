"""Helper methods for various functions around this tool."""

# pylint: disable=E0401
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ladybug.datacollection import BaseCollection
from ladybug.wea import Wea
from ladybug_geometry.geometry2d import Vector2D
from ladybug_geometry.geometry3d import Vector3D
from matplotlib.colors import colorConverter
from sklearn.linear_model import LinearRegression

from . import DATA_PATH
from .enum import BuildingType

# pylint: enable=E0401


logger = logging.getLogger(__name__.split(".", maxsplit=1)[0])

def collection_to_series(collection: BaseCollection) -> pd.Series:
    """Helper method to convert a ladybug datacollection to a pandas Series for
     easier handling/visualisation.

    Args:
        collection (BaseCollection):
            The ladybug collection to convert.

    Returns:
        pd.Series:
            The converted pandas Series.
    """

    if not isinstance(collection, BaseCollection):
        raise TypeError(f"Input is wrong type!: {type(collection)}")

    index = pd.DatetimeIndex(collection.header.analysis_period.datetimes)
    if len(collection.values) == 12:
        index = pd.date_range(
            f"{collection.header.analysis_period.datetimes[0].year}-01-01",
            periods=12,
            freq="MS",
        )

    return pd.Series(
        collection.values,
        index=index,
        name=f"{collection.header.data_type.name} ({collection.header.unit})",
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

    data = pd.read_csv(DATA_PATH / "sri_data.csv", header=0)

    model = LinearRegression()
    model.fit(
        data[["solar_absorptivity", "thermal_absorptivity"]].values, data["sri"].values
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

    if sa_ <= 0 or sa_ >= 1:
        logger.error(
            (
                "Solar absorptivity estimation is beyond allowable limits "
                "for the target SRI."
            )
        )

    return sa_, ta_


def typical_lift_energy(
    building_type: BuildingType,
    occupancy_schedule: pd.Series,
    target_n_floors: int,
    target_building_height: float,
) -> pd.Series:
    """Estimate the annual energy consumption of a lift system in Wh,
    for a building of the given height.

    Source:
    For energy demand per year:
        Ang, Jia Hui, et al. 'Comprehensive Energy Consumption of Elevator
        Systems Based on Hybrid Approach of Measurement and Calculation in Low-
        and High-Rise Buildings of Tropical Climate towards Energy Efficiency'.
        Sustainability, vol. 14, no. 8, Apr. 2022, p. 4779. DOI.org (Crossref),
        https://doi.org/10.3390/su14084779.
    For lift usage profile during day:
        Tukia, Toni, et al. 'Modeling the Aggregated Power Consumption of
        Elevators – the New York City Case Study'. Applied Energy, vol. 251,
        Oct. 2019, p. 113356. DOI.org (Crossref),
        https://doi.org/10.1016/j.apenergy.2019.113356.

    Args:
        building_type (BuildingType):
            The type of building to calculate for.
        occupancy_schedule (pd.Series):
            The occupancy schedule for the building.
        target_n_floors (int):
            The number of floors in the building.
        target_building_height (float):
            The height of the building in meters.

    Returns:
        pd.Series:
            The estimated hourly energy consumption of the lift system in Wh.
    """

    if not isinstance(occupancy_schedule, pd.Series):
        raise TypeError("Occupancy schedule must be a pandas Series.")

    if not isinstance(target_n_floors, int):
        raise TypeError("Number of floors must be an integer.")

    if not isinstance(target_building_height, (int, float)):
        raise TypeError("Building height must be a number.")

    if target_n_floors < 2:
        return pd.Series(
            np.zeros(8760), index=occupancy_schedule.index, name="Lifts (Wh)"
        )

    if len(occupancy_schedule) != 8760:
        raise ValueError("Occupancy schedule must have 8760 values.")

    if not isinstance(occupancy_schedule.index, pd.DatetimeIndex):
        raise ValueError("Occupancy schedule must have a datetime index.")

    # load the datasets
    usage_profile = pd.read_csv(
        Path(__file__).parent / "data" / "lift_profile.csv",
        header=0,
        index_col=0,
    )
    annual_energy = pd.read_csv(
        Path(__file__).parent / "data" / "lift_energy.csv",
        header=0,
    )

    usage = []
    for wkday in occupancy_schedule.resample("D").mean().index.weekday:
        if wkday in [5, 6]:
            usage += usage_profile.weekend.values.tolist()
        else:
            usage += usage_profile.weekday.values.tolist()
    usage_profile = pd.Series(usage, index=occupancy_schedule.index)

    # get "Office" or "Residential" based on building type
    lift_bdg_type = "Residential" if "APARTMENT" in building_type.name else "Office"
    data = annual_energy[annual_energy.BuildingUse == lift_bdg_type][
        ["BuildingHeight_m", "Floors", "AnnualEnergyConsumption_Wh"]
    ]
    model = LinearRegression()
    model.fit(
        data[["BuildingHeight_m", "Floors"]].values,
        data["AnnualEnergyConsumption_Wh"].values,
    )
    # get annual total energy demand value
    annual_energy_wh = model.predict([[target_building_height, target_n_floors]])[0]

    # apportion energy across year, based on occupancy level and daily usage profile
    temp = (usage_profile / usage_profile.sum()) * (
        occupancy_schedule / occupancy_schedule.sum()
    )
    temp = temp / temp.sum()
    temp.name = "Lifts (Wh)"
    return temp * annual_energy_wh


def insolation(
    epw_file: str | Path, azimuth: float = 180, altitude: float = 90
) -> float:
    """Calculate the insolation for a given EPW file, azimuth, and altitude.

    Args:
        epw_file (str | Path):
            The path to the EPW file or the name of the EPW file in the data folder.
        azimuth (float):
            The azimuth angle in degrees.
        altitude (float):
            The altitude angle in degrees.

    Returns:
        float:
            The insolation in Wh/m2.
    """

    if not isinstance(epw_file, (str, Path)):
        raise TypeError("EPW file must be a string or a Path.")

    wea = Wea.from_epw_file(epw_file)
    total, _, _, _ = wea.directional_irradiance(altitude=altitude, azimuth=azimuth)

    return collection_to_series(total)


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

    return np.rad2deg(north.angle_clockwise(vec2d))


def convert_dataframe(
    dataframe: pd.DataFrame,
    source_unit: str,
    target_unit: str,
    remove_unit: bool = False,
) -> pd.DataFrame:
    """Convert a dataframe from one unit to another.

    Args:
        dataframe (pd.DataFrame):
            The pandas dataframe to convert.
        source_unit (str):
            The unit of the source object.
        target_unit (str):
            The unit to convert to.
        remove_unit (bool, optional):
            Whether to remove the unit from the objects name/s. Default is False.

    Returns:
        pd.DataFrame:
            The converted dataframe.
    """

    logger.warning("MAKE DATA COLLECTIONS INSTEAD TO ALLOW FOR UNIT CONVERSION EASILY!")
    
    lookup = {
        ("Wh", "Wh"): 1,
        ("kWh", "kWh"): 1,
        ("MWh", "MWh"): 1,
        ("Wh", "kWh"): 1 / 1000,
        ("kWh", "Wh"): 1000,
        ("Wh", "MWh"): 1 / 1000000,
        ("MWh", "Wh"): 1000000,
        ("kWh", "MWh"): 1 / 1000,
        ("MWh", "kWh"): 1000,
        ("Wh/m2", "Wh/m2"): 1,
        ("kWh/m2", "kWh/m2"): 1,
        ("MWh/m2", "MWh/m2"): 1,
        ("Wh/m2", "kWh/m2"): 1 / 1000,
        ("kWh/m2", "Wh/m2"): 1000,
        ("Wh/m2", "MWh/m2"): 1 / 1000000,
        ("MWh/m2", "Wh/m2"): 1000000,
        ("kWh/m2", "MWh/m2"): 1 / 1000,
        ("MWh/m2", "kWh/m2"): 1000,
    }

    for col in dataframe.columns:
        if col.split(" (")[1][:-1] != source_unit:
            raise ValueError(
                f'Column "{col}" does not indicate the expected unit of "{source_unit}".'
            )

    if (source_unit, target_unit) in lookup:
        dataframe *= lookup[(source_unit, target_unit)]
    else:
        raise ValueError(
            f"Conversion from {source_unit} to {target_unit} is not supported."
        )

    if remove_unit:
        dataframe.columns = [i.split(" (")[0] for i in dataframe.columns]
    else:
        dataframe.columns = [
            i.replace(source_unit, target_unit) for i in dataframe.columns
        ]

    return dataframe
