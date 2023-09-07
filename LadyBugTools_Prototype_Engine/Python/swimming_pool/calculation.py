from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ladybugtools_toolkit.ladybug_extension.epw import (
    EPW,
    epw_to_dataframe,
)
from tqdm import tqdm

from .evaporation_convection import (
    evaporation_rate,
    evaporation_gain,
    convection_gain,
    density_water,
    specific_heat_water,
    latent_heat_of_vaporisation,
)
from .conduction import conduction_gain
from .longwave import longwave_gain
from .shortwave import shortwave_gain
from .occupants import occupant_gain
from .plot import plot_monthly_balance, plot_timeseries


def main(
    epw_file: str,
    surface_area: float = 1,
    average_depth: float = 1,
    n_occupants: Union[int, Tuple[int], pd.Series] = 0,
    target_water_temperature: Union[float, Tuple[float], pd.Series] = None,
    supply_water_temperature: Union[float, Tuple[float], pd.Series] = None,
    ground_interface_u_value: float = 0.25,
    wind_height_above_water: float = 1,
    coverage_schedule: Union[int, Tuple[int], pd.Series] = None,
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
        epw=epw, include_additional=True, ground_temperature_depth=average_depth
    ).droplevel([0, 1], axis=1)

    air_temperature = epw_df["Dry Bulb Temperature (C)"]
    sky_temperature = epw_df["Sky Temperature (C)"]
    air_pressure = epw_df["Atmospheric Station Pressure (Pa)"]
    ground_temperature = epw_df["Ground Temperature (C)"]
    insolation = epw_df["Global Horizontal Radiation (Wh/m2)"]

    evap_rate = evaporation_rate(
        epw=epw, wind_height_above_water=wind_height_above_water
    )

    # create coverage schedule, which is a multiplier on values which are exposed to air/sun
    # TODO- Implement and add effecrts to various heat gain mehcnaisms
    if coverage_schedule is None:
        coverage_schedule = pd.Series(
            np.zeros(len(epw_df)), name="Water Coverage (dimensionless)"
        )
    elif isinstance(coverage_schedule, (float, int)):
        if coverage_schedule > 1 or coverage_schedule < 0:
            raise ValueError("Coverage schedule must be between 0 and 1 (inclusive)")
        coverage_schedule = pd.Series(
            np.zeros(len(epw_df)) * coverage_schedule,
            name="Water Coverage (dimensionless)",
        )
    else:
        coverage_schedule = pd.Series(
            coverage_schedule, name="Water Coverage (dimensionless)"
        )

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

    # create target water temperature profile
    if target_water_temperature is None:
        target_water_temperature = ground_temperature.rename(
            "Target Water Temperature (C)"
        )
    elif isinstance(target_water_temperature, (float, int)):
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
        n_occupants = pd.Series(
            np.ones(len(epw_df)) * n_occupants,
            name="Number of Occupants (dimensionless)",
            index=air_temperature.index,
        )
    else:
        n_occupants = pd.Series(
            n_occupants,
            name="Number of Occupants (dimensionless)",
            index=air_temperature.index,
        )
    q_occupants = occupant_gain(n_people=n_occupants)

    # create solar gain
    q_solar = shortwave_gain(
        surface_area=surface_area,
        insolation=insolation,
    )

    # Derive properties from inputs
    container_volume = surface_area * average_depth

    # set initial properties for water prior to looping through
    _current_water_temperature = (
        ground_temperature.mean()
    )  # use avg ground temp for year as starting water temp
    _current_water_density = density_water(_current_water_temperature)  # kg/m3
    _current_water_specific_heat_capacity = (
        specific_heat_water(_current_water_temperature) * 1000
    )  # J/kg/K
    _current_water_latent_heat_of_vaporisation = latent_heat_of_vaporisation(
        _current_water_temperature, air_pressure[0]
    )  # kJ/kg

    q_evaporation = []
    q_conduction = []
    q_convection = []
    q_longwave = []
    q_htg_clg = []
    q_energy_balance = []
    water_temperature_without_htgclg = []
    remaining_water_temperature = []
    pbar = tqdm(list(enumerate(epw_df.iterrows())))
    for n, (idx, _) in pbar:
        pbar.set_description(f"Calculating {idx:%b %d %H:%M}")

        # calculate heat balance for this point in time
        _evap_gain = evaporation_gain(
            evap_rate=evap_rate[n],
            surface_area=surface_area * (1 - coverage_schedule[n]),
            latent_heat_of_evaporation=_current_water_latent_heat_of_vaporisation,  # TODO - check units here
            water_density=_current_water_density,
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
            average_depth=average_depth,
            interface_u_value=ground_interface_u_value,
            soil_temperature=ground_temperature[n],
            water_temperature=_current_water_temperature,
        )
        q_conduction.append(_cond_gain)

        # calculate the resultant energy balance following these gains
        _energy_balance = (
            _evap_gain
            + _conv_gain
            + _lwav_gain
            + _cond_gain
            + q_solar[n]
            + q_occupants[n]
        )
        q_energy_balance.append(_energy_balance)

        # calculate resultant temperature in remaining water
        volume_water_lost = (evap_rate[n] * surface_area) / 1000  # m3
        remaining_water_volume = container_volume - volume_water_lost  # m3
        remaining_water_mass = remaining_water_volume * _current_water_density  # kg
        remaining_water_temperature = (
            _energy_balance
            / (remaining_water_mass * _current_water_specific_heat_capacity)
        ) + _current_water_temperature  # C
        remaining_water_temperature.append(remaining_water_temperature)

        # TODO - add water to pool in terms of energy rather than temperature

        # add water back into the body of water at a given temperature
        _current_water_temperature = np.average(
            [supply_water_temperature[n], remaining_water_temperature],
            weights=[volume_water_lost, remaining_water_volume],
        )
        water_temperature_without_htgclg.append(_current_water_temperature)

        # recalculate properties for water at this temperature
        _current_water_density = density_water(_current_water_temperature)  # kg/m3
        _current_water_specific_heat_capacity = (
            specific_heat_water(_current_water_temperature) * 1000
        )  # J/kg/K
        _current_water_latent_heat_of_vaporisation = latent_heat_of_vaporisation(
            _current_water_temperature, air_pressure[n]
        )  # kJ/kg

        # TODO - calculate the heat required to reach the target water temperature
        q_htg_clg = (
            (_current_water_density * container_volume)
            * _current_water_specific_heat_capacity
            * (target_water_temperature[n] - _current_water_temperature)
        )

    # combine heat gains into a single dataframe
    q_solar = pd.Series(q_solar, index=epw_df.index, name="Q_Solar (W)")
    q_occupants = pd.Series(q_occupants, index=epw_df.index, name="Q_Occupants (W)")
    q_longwave = pd.Series(q_longwave, index=epw_df.index, name="Q_Longwave (W)")
    q_conduction = pd.Series(q_conduction, index=epw_df.index, name="Q_Conduction (W)")
    q_evaporation = pd.Series(
        q_evaporation, index=epw_df.index, name="Q_Evaporation (W)"
    )
    q_convection = pd.Series(q_convection, index=epw_df.index, name="Q_Convection (W)")
    q_conditioning_water_temp = pd.Series(
        q_htg_clg, index=epw_df.index, name="Q_Conditioning [WATERTEMP] (W)"
    )
    q_conditioning_heat_balance = pd.Series(
        q_energy_balance, index=epw_df.index, name="Q_Conditioning [HEATBALANCE] (W)"
    )

    water_temperature_without_htgclg = pd.Series(
        water_temperature_without_htgclg,
        index=epw_df.index,
        name="Water Temperature [without htg/clg] (C)",
    )
    remaining_water_temperature = pd.Series(
        remaining_water_temperature,
        index=epw_df.index,
        name="Water Temperature [prior to resupply] (C)",
    )

    gains_df = pd.concat(
        [
            q_solar,
            q_occupants,
            q_longwave,
            q_conduction,
            q_evaporation,
            q_convection,
            q_conditioning_water_temp,
            q_conditioning_heat_balance,
            water_temperature_without_htgclg,
            remaining_water_temperature,
            air_temperature,
            sky_temperature,
            air_pressure,
            ground_temperature,
            insolation,
            evap_rate,
            supply_water_temperature,
            target_water_temperature,
        ],
        axis=1,
    )

    return gains_df


if __name__ == "__main__":
    epw_file = r"C:\Users\tgerrish\Buro Happold\0053340 DDC - Project W Master - Climate\EPW_Modified\SAU_MD_Prince.Abdulmajeed.Bin.Abdulaziz.AP.404010_TMYx.2007-2021_FIXED_TG_300M.epw"
    surface_area = 1  # m2
    avg_depth = 1  # m

    # create occupant profile
    n_occupants = 0.1  # approx 625 max occupancy of an olympic pool
    datetimes = pd.date_range("2017-01-01 00:00:00", "2017-12-31 23:00:00", freq="H")
    n_occupants = np.where(
        datetimes.hour.isin([10, 11, 13, 14, 17]),
        n_occupants,
        np.where(datetimes.hour.isin([8, 9, 12, 15, 16]), n_occupants * 0.5, 0),
    )

    main(
        epw_file=epw_file,
        n_occupants=n_occupants,
    )
