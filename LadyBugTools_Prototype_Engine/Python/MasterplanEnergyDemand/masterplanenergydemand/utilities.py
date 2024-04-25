"""..."""

# pylint: disable=too-many-lines, import-error, logging-fstring-interpolation, unused-import, too-many-locals, too-many-statements, too-many-branches, no-name-in-module

from pathlib import Path
from typing import Any
import calendar
import warnings
import logging

import numpy as np
import pandas as pd

from ladybug.wea import Wea
from matplotlib.colors import colorConverter
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from ladybug_geometry.geometry3d import Vector3D
from ladybug_geometry.geometry2d import Vector2D

from .enum import (
    BuildingType,
)
from . import FORMATTING

logging.basicConfig(
    level=logging.INFO,  # change to CRITICAL to silence logging (mostly)
    format="%(levelname)s - %(message)s",
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

    data = pd.read_csv(
        Path(__file__).parent / "data" / "sri_data.csv",
        header=0,
    )

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

    logging.debug(
        "Target SRI of %f±%f achieved (%.1f), using solar absorptance of %.3f and thermal emittance of %.3f",
        target_sri,
        tolerance,
        sri_,
        sa_,
        ta_,
    )

    if sa_ <= 0 or sa_ >= 1:
        logging.error(
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

    return pd.Series(
        total.values,
        index=pd.to_datetime(total.header.analysis_period.datetimes),
        name="Insolation (Wh/m2)",
    )


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
            "The angle entered is beyond the normally expected range for an orientation in degrees."
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


def plot_monthly_stacked_bar(df: pd.DataFrame, ax: plt.Axes = None) -> plt.Axes:
    """_"""

    # remove units from dataframe columns (if they exist)
    df = df.rename(columns=lambda x: x.split(" (")[0])

    # resample to monthly
    data = df.resample("MS").sum()

    if ax is None:
        ax = plt.gca()

    # get colors
    colors = [FORMATTING.color[i] for i in data.columns]

    # plot
    data.plot(
        ax=ax,
        kind="bar",
        stacked=True,
        color=colors,
        legend=False,
        zorder=3,
        width=0.8,
    )

    # add labels to each bar segment
    ylims = ax.get_ylim()
    for rect in ax.patches:
        y_value = rect.get_y() + rect.get_height() / 2
        x_value = rect.get_x() + rect.get_width() / 2
        fc = rect.get_facecolor()
        actual_value = rect.get_height()

        if abs(actual_value) / (max(ylims) - min(ylims)) > 0.025:
            if abs(actual_value) < 100:
                val = f"{actual_value:,.1f}"
            else:
                val = f"{actual_value:,.0f}"
            # Create annotation
            ax.text(
                x_value,
                y_value,
                val,
                ha="center",
                va="center",
                fontsize="xx-small",
                color=contrasting_color(fc),
                alpha=0.75,
                zorder=8,
            )

    # add axes formatting
    ax.set_xticklabels([f"{i:%b}" for i in data.index], rotation=0)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
    )

    return ax


def plot_pie(series: pd.Series, ax: plt.Axes = None) -> plt.Axes:
    """_"""

    if ax is None:
        ax = plt.gca()

    colors = [FORMATTING.color[i] for i in series.index]

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            if pct / 100 > 0.075:
                return f"{pct/100:.2%}\n({val:,})"
            return ""

        return my_autopct

    wedges, _, autotexts = ax.pie(
        series,
        startangle=90,
        counterclock=False,
        colors=colors,
        autopct=make_autopct(series.values),
        textprops={"color": "w", "fontsize": "x-small"},
    )
    ax.legend(wedges, series.index, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.tight_layout()
    for txt, wdg in zip(autotexts, wedges):
        txt.set_color(contrasting_color(wdg.get_facecolor()))

    return ax


def plot_diurnal(data: pd.DataFrame, ax: plt.Axes = None) -> plt.Axes:
    """_"""

    # remove units from dataframe columns (if they exist)
    data = data.rename(columns=lambda x: x.split(" (")[0])

    colors = [FORMATTING.color[i] for i in data.columns]

    grp_mean = data.groupby([data.index.month, data.index.hour]).mean()

    target_idx = pd.MultiIndex.from_product([range(1, 13, 1), range(24)])
    major_ticks = range(len(target_idx))[::12]
    minor_ticks = range(len(target_idx))[::6]
    major_ticklabels = []
    for i in target_idx:
        if i[1] == 0:
            major_ticklabels.append(f"{calendar.month_abbr[i[0]]}")
        elif i[1] == 12:
            major_ticklabels.append("")

    if ax is None:
        ax = plt.gca()

    for n, (var, vals) in enumerate(grp_mean.items()):
        for month in range(1, 13, 1):
            x_vals = np.arange((month - 1) * 24, (month * 24))
            y_vals = vals.loc[month].tolist()
            ax.plot(
                x_vals,
                y_vals,
                c=colors[n],
                label=var if month == 1 else "_nolegend_",
            )
            ax.axvline(x=x_vals[0], c="k", lw=0.6, ls="-", alpha=0.25)

    ax.xaxis.set_major_locator(mticker.FixedLocator(major_ticks))
    ax.xaxis.set_minor_locator(mticker.FixedLocator(minor_ticks))
    ax.set_xticklabels(
        major_ticklabels,
        minor=False,
        ha="left",
    )
    ax.set_xlim(0, 288)
    ax.set_ylim(grp_mean.min().min(), grp_mean.max().max())
    ax.legend(loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.grid(which="minor", alpha=0.25)

    return ax


def plot_duration_curve(
    data: pd.Series | pd.DataFrame,
    ax: plt.Axes = None,
    remove_zero: bool = True,
    **kwargs,
) -> plt.Axes:
    """Plot a duration curve for a given pandas Series of data."""

    if ax is None:
        ax = plt.gca()

    if isinstance(data, pd.Series):
        data = data.to_frame()

    # sort by most to least
    data = data[data.sum(axis=0).sort_values(ascending=False).index]

    density = kwargs.pop("density", False)
    if density:
        raise NotImplementedError("Density is not yet implemented for duration curves.")

    orientation = kwargs.pop("orientation", "horizontal")

    # get the defaults
    color = [FORMATTING.color[i] for i in data]
    label = [kwargs.pop("label", i) for i in data]
    # cumulative = kwargs.pop("cumulative", -1)

    if remove_zero:
        data[data == 0] = np.nan

    ax.hist(
        data,
        color=color,
        density=kwargs.pop("density", False),
        bins=kwargs.pop("bins", 100),
        label=label,
        histtype=kwargs.pop("histtype", "step"),
        orientation=orientation,
        cumulative=kwargs.pop("cumulative", -1),
        **kwargs,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if orientation == "horizontal":
            ax.set_xlabel("Annual hours")
            ax.set_ylim(0, np.nanmax(data.values) * 1.05)
        else:
            ax.set_ylabel("Annual hours")
            ax.set_xlim(0, np.nanmax(data.values) * 1.05)

    ax.legend(loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.grid(which="minor", alpha=0.25)

    plt.tight_layout()

    return ax


def plot_heating_cooling_series(
    heating: pd.Series, cooling: pd.Series, ax: plt.Axes = None, kind: str = "line"
) -> plt.Axes:
    """Plot heating and cooling series on the same axis."""

    if ax is None:
        ax = plt.gca()

    if kind == "line":
        ax.plot(heating, label="Heating", c=FORMATTING.color["Heating"], lw=0.5)
        ax.plot(cooling, label="Cooling", c=FORMATTING.color["Cooling"], lw=0.5)
    elif kind == "stacked":
        ax.stackplot(
            heating.index,
            heating,
            cooling,
            labels=["Heating", "Cooling"],
            colors=[FORMATTING.color["Heating"], FORMATTING.color["Cooling"]],
        )

    ax.legend(loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    ax.set_xlim(
        min(heating.index.min(), cooling.index.min()),
        max(heating.index.max(), cooling.index.max()),
    )
    ax.set_ylim(0, ax.get_ylim()[-1])

    return ax


def create_legend(ax: plt.Axes = None, legend_type: str = "energy") -> plt.Axes:
    """Create a legend from the FORMATTING color dictionary."""

    if legend_type not in ["energy", "thermal"]:
        raise ValueError('The "type" argument must be one of "energy" or "thermal".')

    if legend_type == "energy":
        variables = [
            "Heating",
            "Cooling",
            "Service Hot Water",
            "Lighting",
            "Mechanical Ventilation",
            "Electric Equipment",
            "Gas Equipment",
            "Fans",
            "Pumps",
        ]
    elif legend_type == "thermal":
        variables = [
            "Heating",
            "Cooling",
            "Solar",
            "Service Hot Water",
            "Lighting",
            "Infiltration",
            "Mechanical Ventilation",
            "Window Conduction",
            "Opaque Conduction",
            "Electric Equipment",
            "People",
            "Gas Equipment",
            "Storage",
        ]

    if ax is None:
        ax = plt.gca()

    for k, v in FORMATTING.color.items():
        if k in variables:
            ax.bar(0, 0, color=v, label=k)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
    )
    ax.axis("off")

    return ax


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
