from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint
from iapws import SeaWater
from ladybugtools_toolkit.ladybug_extension.epw import (
    EPW,
    epw_to_dataframe,
    wind_speed_at_height,
)
from tqdm import tqdm

# from .conduction import conduction_gain
from evaporation_convection import (
    evaporation_rate,
    evaporation_gain,
    convection_gain,
    vapor_pressure_antoine,
    latent_heat_of_vaporisation,
)
from conduction import conduction_gain
from longwave import longwave_gain
from shortwave import shortwave_gain
from occupants import occupant_gain
from plot import plot_monthly_balance, plot_timeseries


def main(
    epw_file: str = r"C:\Users\tgerrish\Buro Happold\0053340 DDC - Project W Master - Climate\EPW_Modified\SAU_MD_Prince.Abdulmajeed.Bin.Abdulaziz.AP.404010_TMYx.2007-2021_FIXED_TG_300M.epw",
    surface_area: float = 1,
    avg_depth: float = 1,
    n_occupants: Union[int, Tuple[int]] = 0,
    target_water_temperature: Union[float, Tuple[float]] = 29,
    target_water_temperature_band: float = 1,
    supply_water_temperature: Union[float, Tuple[float]] = None,
    ground_interface_u_value: float = 0.25,
    water_salinity: float = 0.0038,
    wind_height_above_water: float = 1,
):
    """

    Assumptions:
    - The pool is a rectangular prism
    - The pool is filled with water of a fixed salinity (0.0038 kg/kg, for seawater typical of Persian Gulf, from which thermophysical properties are derived)
    - Water is supplied to the pool at the same rate as it is evaporated, so the water volume remains constant.

    Args:

    """

    # Load EPW data
    epw = EPW(epw_file)
    epw_df = epw_to_dataframe(
        epw, include_additional=True, ground_temperature_depth=avg_depth
    ).droplevel([0, 1], axis=1)

    air_temperature = epw_df["Dry Bulb Temperature (C)"]
    sky_temperature = epw_df["Sky Temperature (C)"]
    air_pressure = epw_df["Atmospheric Station Pressure (Pa)"]
    ground_temperature = epw_df["Ground Temperature (C)"]
    insolation = epw_df["Global Horizontal Radiation (Wh/m2)"]

    evap_rate = evaporation_rate(epw, wind_height_above_water=wind_height_above_water)

    # create supply water temperature profile
    if supply_water_temperature is None:
        supply_water_temperature = ground_temperature.rename(
            "Supply Water Temperature (C)"
        )
    elif isinstance(supply_water_temperature, (float, int)):
        supply_water_temperature = pd.Series(
            np.ones(len(ground_temperature)) * supply_water_temperature,
            name="Supply Water Temperature (C)",
            index=ground_temperature.index,
        )
    else:
        supply_water_temperature = pd.Series(
            supply_water_temperature,
            name="Supply Water Temperature (C)",
            index=ground_temperature.index,
        )
    supply_water_specific_heat = 4.186  # kJ/kg/K

    # create target water temperature profile
    if isinstance(target_water_temperature, (float, int)):
        target_water_temperature = pd.Series(
            np.ones(len(air_temperature)) * target_water_temperature,
            name="Target Water Temperature (C)",
            index=air_temperature.index,
        )
    else:
        target_water_temperature = pd.Series(
            target_water_temperature,
            name="Target Water Temperature (C)",
            index=air_temperature.index,
        )

    # create occupant gain
    if isinstance(n_occupants, (float, int)):
        n_occupants = np.ones(len(epw_df)) * n_occupants
    else:
        n_occupants = pd.Series(
            n_occupants,
            name="Number of Occupants (dimensionless)",
            index=air_temperature.index,
        )
    q_occupants = occupant_gain(n_occupants)

    # create solar gain
    q_solar = shortwave_gain(
        surface_area=surface_area,
        insolation=insolation,
    )

    # Derive properties from inputs
    container_volume = surface_area * avg_depth

    # set initial properties for water prior to looping through
    _current_water_temperature = ground_temperature.mean()
    water = SeaWater(
        T=_current_water_temperature + 273.15,
        P=air_pressure[0] * 1e-6,
        S=water_salinity,
        fast=True,
    )

    q_evaporation = []
    q_conduction = []
    q_convection = []
    q_longwave = []
    water_temperature = []
    pbar = tqdm(list(enumerate(epw_df.iterrows())))
    for n, (idx, vals) in pbar:
        if idx.hour == 0:
            pbar.set_description(f"Calculating {idx:%b %d}")

        # calculate heat balance for this point in time
        _evap_gain = evaporation_gain(
            evap_rate=evap_rate[n],
            surface_area=surface_area,
            latent_heat_of_evaporation=2256,  # latent_heat_of_vaporisation(air_temperature[n]),
            water_density=water.rho,
        )
        q_evaporation.append(_evap_gain)

        _conv_gain = convection_gain(
            evap_gain=_evap_gain,
            air_temperature=air_temperature[n],
            water_temperature=_current_water_temperature,
            atmospheric_pressure=air_pressure[n],
        )
        q_convection.append(_conv_gain)

        _lwav_gain = longwave_gain(
            surface_area=surface_area,
            sky_temperature=sky_temperature[n],
            water_temperature=_current_water_temperature,
        )
        q_longwave.append(_lwav_gain)

        _cond_gain = conduction_gain(
            surface_area=surface_area,
            average_depth=avg_depth,
            interface_u_value=ground_interface_u_value,
            soil_temperature=ground_temperature[n],
            water_temperature=_current_water_temperature,
        )
        q_conduction.append(_cond_gain)

        # calculate the resultant energy balance following these gains
        _sum_gain = (
            _evap_gain
            + _conv_gain
            + _lwav_gain
            + _cond_gain
            + q_solar[n]
            + q_occupants[n]
        )

        # TODO - calculate resultant temperature in remaining water
        volume_water_lost = (evap_rate[0] * surface_area) / 1000  # m3
        mass_water_lost = volume_water_lost * water.rho  # kg
        remaining_water_volume = container_volume - volume_water_lost  # m3
        remaining_water_specific_heat = water.cp  # kJ/kg
        remaining_water_density = water.rho  # kg/m3
        remaining_water_mass = remaining_water_volume * remaining_water_density  # kg
        remaining_water_temperature = (
            _sum_gain / (remaining_water_mass * remaining_water_specific_heat)
        ) + _current_water_temperature  # C
        _current_water_temperature = (
            (mass_water_lost * supply_water_specific_heat * supply_water_temperature[n])
            + (
                remaining_water_mass
                * supply_water_specific_heat
                * remaining_water_temperature
            )
        ) / (
            (mass_water_lost * supply_water_specific_heat)
            + (remaining_water_mass * supply_water_specific_heat)
        )

        # TODO - calculate temperatureb after topping up with supply water
        water = SeaWater(
            T=_current_water_temperature + 273.15,
            P=air_pressure[n] * 1e-6,
            S=water_salinity,
            fast=True,
        )
        water_temperature.append(_current_water_temperature)

        # _current_water_isobaric_heat_capacity = water.cp * (
        #     ureg.kilojoule / (ureg.kilogram * ureg.kelvin)
        # )
        # TODO - calculate the heat required to maintain the mixed water at a given temperature

    # combine heat gains into a single dataframe for plotting
    q_solar = pd.Series(q_solar, index=epw_df.index, name="Q_Solar (W)")
    q_occupants = pd.Series(q_occupants, index=epw_df.index, name="Q_Occupants (W)")
    q_longwave = pd.Series(q_longwave, index=epw_df.index, name="Q_Longwave (W)")
    q_conduction = pd.Series(q_conduction, index=epw_df.index, name="Q_Conduction (W)")
    q_evaporation = pd.Series(
        q_evaporation, index=epw_df.index, name="Q_Evaporation (W)"
    )
    q_convection = pd.Series(q_convection, index=epw_df.index, name="Q_Convection (W)")

    water_temperature = pd.Series(
        water_temperature, index=epw_df.index, name="Water Temperature (C)"
    )

    gains_df = pd.concat(
        [
            q_solar,
            q_occupants,
            q_longwave,
            q_conduction,
            q_evaporation,
            q_convection,
        ],
        axis=1,
    )

    fig, [ax0, ax1] = plt.subplots(2, 1, figsize=(8, 8))
    plot_timeseries(gains_df, ax=ax0)
    plot_monthly_balance(gains_df, ax=ax1)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(water_temperature)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
