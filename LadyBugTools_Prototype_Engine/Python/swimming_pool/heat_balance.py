from typing import Union
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from ladybugtools_toolkit.ladybug_extension.epw import EPW, epw_to_dataframe
from .evaporation import evaporation_rate_penman, evaporation_gain_mancic
from .conduction import conduction_gain_interface_area
from .convection import convection_gain_bowen_ratio
from .shortwave import shortwave_gain
from .occupants import occupant_gain
from .longwave import longwave_gain
from .helpers import supply_water_heating


def heat_balance(
    epw: EPW,
    water_surface_area: Union[float, pd.Series, np.ndarray],
    average_depth: float,
    target_water_temperature: Union[float, pd.Series, np.ndarray],
    people_density: Union[float, pd.Series, np.ndarray] = 0,
    ground_interface_u_value: float = 0.25,
    water_supply_temperature: Union[float, pd.Series, np.ndarray] = None,
    include_epw: bool = False,
    target_temperature_band: float = 1,
) -> pd.DataFrame:
    """
    Calculates the heat balance of a water body using the given parameters.

    Parameters:
    -----------
    epw : EPW
        An instance of the EPW class containing the weather data.
    water_surface_area : Union[float, pd.Series, np.ndarray]
        The surface area of the water body in square meters. A profile can be
        applied here instead of a static value, which varies surface area
        based on the proportion of pool covered. NOTE - Coverage does not impact
        convective heat loss/gain.
    average_depth : float
        The average depth of the water body in meters.
    target_water_temperature : Union[float, pd.Series, np.ndarray]
        The temperature of the water body in Celsius. A profile can be applied
        here instead of a static value, which varies temperature for each timestep.
    people_density : Union[float, pd.Series, np.ndarray], optional
        The density of people in the water body in m2/person. Default is 0, but a
        profile can be applied here instead of a static value, which varies
        occupancy level based on the occupancy level of the pool.
    ground_interface_u_value : float, optional
        The U-value of the ground interface in W/m2K. Default is 0.25.
    water_supply_temperature : Union[float, pd.Series, np.ndarray], pd.Series, np.ndarray, optional
        The temperature of the water supplied to the water body. If not provided, the
        ground temperature at 1m beneath surface using the EPW file is used.
    include_epw : bool, optional
        Set to True to include the EPW data in the output DataFrame. Default is False.
    target_temperature_band : float, optional
        The temperature band around the target water temperature in which the
        water body is considered to be at the target temperature. Default is 1.

    Returns:
    --------
    pd.DataFrame
        A pandas DataFrame containing the heat balance results.
    """

    epw_df = epw_to_dataframe(
        epw, include_additional=True, ground_temperature_depth=1
    ).droplevel([0, 1], axis=1)

    if water_supply_temperature is None:
        water_supply_temperature = epw_df["Ground Temperature (C)"]

    if isinstance(water_surface_area, (pd.Series, np.ndarray)):
        if len(water_surface_area) != len(epw_df):
            raise ValueError(
                "The length of the water surface area profile must match the length of the EPW file."
            )
    else:
        water_surface_area = np.ones(len(epw_df)) * water_surface_area

    evaporation_rate = evaporation_rate_penman(epw)  # l/m2/hour
    water_loss_m3_hour = evaporation_rate * water_surface_area / 1000  # m3/hour

    heat_balance_df = pd.DataFrame()
    heat_balance_df["Q_solar (W)"] = shortwave_gain(
        surface_area=water_surface_area,
        insolation=epw_df["Global Horizontal Radiation (Wh/m2)"],
    )
    heat_balance_df["Q_occupants (W)"] = occupant_gain(
        m2_per_person=people_density, surface_area=water_surface_area
    )
    heat_balance_df["Q_longwave (W)"] = longwave_gain(
        surface_area=water_surface_area,
        sky_temperature=epw_df["Sky Temperature (C)"],
        water_temperature=target_water_temperature,
    )
    heat_balance_df["Q_conduction (W)"] = conduction_gain_interface_area(
        surface_area=water_surface_area,
        average_depth=average_depth,
        interface_u_value=ground_interface_u_value,
        soil_temperature=epw_df["Ground Temperature (C)"],
        water_temperature=target_water_temperature,
    )
    heat_balance_df["Q_evaporation (W)"] = evaporation_gain_mancic(
        surface_area=water_surface_area, evaporation_rate=evaporation_rate
    )
    heat_balance_df["Q_convection (W)"] = convection_gain_bowen_ratio(
        air_temperature=epw_df["Dry Bulb Temperature (C)"],
        water_temperature=target_water_temperature,
        atmospheric_pressure=epw_df["Atmospheric Station Pressure (Pa)"],
        evaporation_gain=heat_balance_df["Q_evaporation (W)"],
    )
    heat_balance_df["Q_feedwater (W)"] = supply_water_heating(
        water_volume_m3_hour=water_loss_m3_hour,
        supply_temperature=water_supply_temperature,
        target_temperature=target_water_temperature,
    )

    # add remaining energy required to achieve target water temperature
    heat_balance_df["Q_remaining (W)"] = -heat_balance_df.sum(axis=1)

    if include_epw:
        heat_balance_df = pd.concat([epw_df, heat_balance_df, evaporation_rate], axis=1)

    return heat_balance_df


def plot_timeseries(heat_balance_df: pd.DataFrame, ax: plt.Axes = None) -> plt.Axes:
    """Create a timeseries plot.

    Args:
        heat_balance_df (pd.DataFrame): DataFrame containing heat balance results.

    Returns:
        plt.Axes: A matplotlib Axes object.
    """

    if ax is None:
        ax = plt.gca()

    heat_balance_df.plot(ax=ax, ylabel="W")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1)

    return ax


def plot_monthly_balance(
    heat_balance_df: pd.DataFrame, ax: plt.Axes = None
) -> plt.Axes:
    """Create a monthly heat balance plot.

    Args:
        heat_balance_df (pd.DataFrame): DataFrame containing heat balance results.

    Returns:
        plt.Axes: A matplotlib Axes object.
    """

    if ax is None:
        ax = plt.gca()

    (heat_balance_df.resample("MS").sum() / 1000).plot(
        ax=ax, kind="bar", stacked=True, width=0.95, ylabel="kWh"
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1)

    return ax
