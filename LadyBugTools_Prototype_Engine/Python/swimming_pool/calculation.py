from typing import Tuple, Union

import numpy as np
import pandas as pd
from ladybugtools_toolkit.ladybug_extension.epw import (
    EPW,
    epw_to_dataframe,
)
from tqdm import tqdm
from .etm import *

try:
    from .evaporation_convection import (
        evaporation_rate,
        evaporation_gain,
        convection_gain,
        density_water,
        specific_heat_water,
        latent_heat_of_vaporisation,
        wind_speed_at_height,
    )
    from .conduction import conduction_gain
    from .longwave import longwave_gain
    from .shortwave import shortwave_gain
    from .occupants import occupant_gain
    from .plot import plot_monthly_balance, plot_timeseries
    from ladybugtools_toolkit.prototypes.swimming_pool.etm import equibtemp
except:
    from evaporation_convection import (
        evaporation_rate,
        evaporation_gain,
        convection_gain,
        density_water,
        specific_heat_water,
        latent_heat_of_vaporisation,
        wind_speed_at_height,
    )
    from conduction import conduction_gain
    from longwave import longwave_gain
    from shortwave import shortwave_gain
    from occupants import occupant_gain
    from plot import plot_monthly_balance, plot_timeseries
    from etm import equibtemp


# TODO - implement target temperature schedule
# TODO - implement target temp band


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
    conditioning_schedule: Union[int, Tuple[int], pd.Series] = None,
    target_range: float = 1,
):
    """

    Assumptions:
    - The pool is a rectangular prism
    - Water is supplied to the pool at the same rate as it is evaporated, so the water volume remains constant.

    Args:
    target_range : the target temperature range is used for conditioning, used like target_water_temperature +/- target_range

    """

    # Load EPW data
    epw = EPW(epw_file)
    epw_df = epw_to_dataframe(
        epw=epw, include_additional=True, ground_temperature_depth=average_depth
    ).droplevel([0, 1], axis=1)

    air_temperature = epw_df["Dry Bulb Temperature (C)"].rename(
        ("Params", "Air Temperature (C)")
    )
    sky_temperature = epw_df["Sky Temperature (C)"].rename(
        ("Params", "Sky Temperature (C)")
    )
    air_pressure = epw_df["Atmospheric Station Pressure (Pa)"].rename(
        ("Params", "Air Pressure (Pa)")
    )
    ground_temperature = epw_df["Ground Temperature (C)"].rename(
        ("Params", "Ground Temperature (C)")
    )
    insolation = epw_df["Global Horizontal Radiation (Wh/m2)"].rename(
        ("Params", "Insolation (Wh/m2)")
    )

    # create coverage schedule, which is a multiplier on values which are exposed to air/sun
    coverage_name = ("Params", "Water Coverage (dimensionless)")
    if coverage_schedule is None:
        coverage_schedule = pd.Series(
            np.zeros(len(epw_df)),
            name=coverage_name,
            index=ground_temperature.index,
        )
    elif isinstance(coverage_schedule, (float, int)):
        if coverage_schedule > 1 or coverage_schedule < 0:
            raise ValueError("Coverage schedule must be between 0 and 1 (inclusive)")
        coverage_schedule = pd.Series(
            np.zeros(len(epw_df)) * coverage_schedule,
            name=coverage_name,
            index=ground_temperature.index,
        )
    else:
        coverage_schedule = pd.Series(
            coverage_schedule,
            name=coverage_name,
            index=ground_temperature.index,
        )

    # create supply water temperature profile
    supply_tmp_name = ("Params", "Supply Water Temperature (C)")
    if supply_water_temperature is None:
        supply_water_temperature = ground_temperature.rename(supply_tmp_name)
    elif isinstance(supply_water_temperature, (float, int)):
        supply_water_temperature = pd.Series(
            np.ones(len(ground_temperature)) * supply_water_temperature,
            name=supply_tmp_name,
            index=ground_temperature.index,
        )
    else:
        supply_water_temperature = pd.Series(
            supply_water_temperature,
            name=supply_tmp_name,
            index=ground_temperature.index,
        )
    supply_water_temperature.clip(lower=0, inplace=True)

    # create target water temperature profile
    setpt_name = ("Params", "Target Water Temperature (C)")
    if target_water_temperature is None:
        pass
    elif isinstance(target_water_temperature, (float, int)):
        target_water_temperature = pd.Series(
            np.ones(len(air_temperature)) * target_water_temperature,
            name=setpt_name,
            index=air_temperature.index,
        )
    else:
        target_water_temperature = pd.Series(
            target_water_temperature,
            name=setpt_name,
            index=air_temperature.index,
        )

    # create conditioning schedule, which is a multiplier on target_water_temperature
    if conditioning_schedule is None:
        conditioning_schedule = pd.Series(
            np.full(len(epw_df), True),
            name="Water Conditioning (dimensionless)",
            index=ground_temperature.index,
        )
    elif isinstance(conditioning_schedule, (bool)):
        conditioning_schedule = pd.Series(
            np.full(len(epw_df), conditioning_schedule),
            name="Water Conditioning (dimensionless)",
            index=ground_temperature.index,
        )
    else:
        conditioning_schedule = pd.Series(
            conditioning_schedule,
            name="Water Conditioning (dimensionless)",
            index=ground_temperature.index,
        )

    # create occupant gain
    occ_name = ("Params", "Number of Occupants (dimensionless)")
    if isinstance(n_occupants, (float, int)):
        n_occupants = pd.Series(
            np.ones(len(epw_df)) * n_occupants,
            name=occ_name,
            index=air_temperature.index,
        )
    else:
        n_occupants = pd.Series(
            n_occupants,
            name=occ_name,
            index=air_temperature.index,
        )
    q_occupants = occupant_gain(n_people=n_occupants)

    # create solar gain
    q_solar = shortwave_gain(
        surface_area=surface_area * (1 - coverage_schedule),
        insolation=insolation,
    )

    # Derive properties from inputs
    container_volume = surface_area * average_depth

    # set initial properties for water prior to looping through
    if target_water_temperature is None:
        _current_water_temperature = (
            supply_water_temperature.mean() 
        )  # use mean supply water temp as initial temperature
    else:
        _current_water_temperature = (
            target_water_temperature[0] - target_range
        )  # use initial target water temp low range as initial temperature
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
    evap_rate = []
    water_temperature_without_htgclg = []
    remaining_water_temperature = []
    pbar = tqdm(list(enumerate(epw_df.iterrows())))
    x = -1 # dev analysis statements
    for i, (n, (idx, _)) in enumerate(pbar):
        pbar.set_description(f"Calculating {idx:%b %d %H:%M}")

        # calculate heat balance for this point in time

        _lwav_gain = longwave_gain(
            surface_area=surface_area * (1 - coverage_schedule[n]),
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
        
        if i > x and x != -1:
            print("q solar:" + str(q_solar[n]))

        _evap_rate = equibtemp(
            sky_temperature[n],
            average_depth,
            wind_speed_at_height(epw.wind_speed[n], 10, wind_height_above_water),
            air_temperature[n],
            epw_df["Wet Bulb Temperature (C)"][n],
            epw.relative_humidity[n],
            (q_solar[n]/surface_area),
            _current_water_temperature,
            tstep=3600
        )
        
        if i > x and x != -1:
            print("evap rate:" + str(_evap_rate))
        evap_rate.append(_evap_rate)

        _evap_gain = evaporation_gain(
            evap_rate=_evap_rate,
            surface_area=surface_area * (1 - coverage_schedule[n]),
            latent_heat_of_evaporation=_current_water_latent_heat_of_vaporisation,
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
        
        # calculate the resultant energy balance following these gains (FUDGE FACTOR for evap gain as more research needs to be done into evaporative energy losses)
        _energy_balance = (
            _evap_gain*0.6
            + q_solar[n]
            + _conv_gain
            + _lwav_gain
            + _cond_gain
            + q_occupants[n]
        )
        q_energy_balance.append(_energy_balance)
        if i > x and x != -1:
            print("evap gain:" + str(_evap_gain))
        if i > x and x != -1:
            print("conv gain:" + str(_conv_gain))
        if i > x and x != -1:
            print("lwav gain:" + str(_lwav_gain))
        if i > x and x != -1:
            print("cond gain:" + str(_cond_gain))
        if i > x and x != -1:
            print("q solar:" + str(q_solar[n]))
        if i > x and x != -1:
            print("q occupants:" + str(q_occupants[n]))
        if i > x and x != -1:
            print("net energy loss:" + str(_energy_balance))
        if i > x and x != -1:
            print("temp before loss:" + str(_current_water_temperature))

        # calculate resultant temperature in remaining water
        volume_water_lost = (
            _evap_rate * (surface_area * (1 - coverage_schedule[n]))
        ) / 1000  # m3
        if i > x and x != -1:
            print("vol water lost:" + str(volume_water_lost))
        remaining_water_volume = container_volume - volume_water_lost  # m3
        if i > x and x != -1:
            print("remaining water vol:" + str(remaining_water_volume))
        remaining_water_mass = remaining_water_volume * _current_water_density  # kg
        if i > x and x != -1:
            print("remaining water mass:" + str(remaining_water_mass))
        _remaining_water_temperature = (
            (_energy_balance*3600)
            / (remaining_water_mass * _current_water_specific_heat_capacity)
        ) + _current_water_temperature  # C
        if i > x and x != -1:
            print("remaining water temp:" + str(_remaining_water_temperature))
        remaining_water_temperature.append(_remaining_water_temperature)
        if i > x and x != -1:
            print("woo!")

        # add water back into the body of water at a given temperature
        _current_water_temperature = np.average(
            [supply_water_temperature[n], _remaining_water_temperature],
            weights=[volume_water_lost, remaining_water_volume],
        )
        if i > x and x != -1:
            print("water temp after supply:" + str(_current_water_temperature))

        water_temperature_without_htgclg.append(_current_water_temperature)

        # recalculate properties for water at this temperature
        _current_water_density = density_water(_current_water_temperature)  # kg/m3
        _current_water_specific_heat_capacity = (
            specific_heat_water(_current_water_temperature) * 1000
        )  # J/kg.K
        _current_water_latent_heat_of_vaporisation = latent_heat_of_vaporisation(
            _current_water_temperature, air_pressure[n]
        )  # kJ/kg
        
        #_current_water_temperature = tw
        # calculate the heat required to reach the target water temperature
        if target_water_temperature is not None:
            if conditioning_schedule[n]:  #
                if _current_water_temperature < (
                    target_water_temperature[n] - target_range
                ):
                    q_htg_clg.append(
                        (
                            (_current_water_density * container_volume)
                            * _current_water_specific_heat_capacity
                            * (
                                (target_water_temperature[n] - target_range)
                                - _current_water_temperature
                            )
                        )/3600
                    )
                    _current_water_temperature = (
                        target_water_temperature[n] - target_range
                    )
                elif _current_water_temperature > (
                    target_water_temperature[n] + target_range
                ):
                    q_htg_clg.append(
                        (
                            (_current_water_density * container_volume)
                            * _current_water_specific_heat_capacity
                            * (
                                (target_water_temperature[n] + target_range)
                                - _current_water_temperature
                            )
                        )/3600
                    )
                    _current_water_temperature = (
                        target_water_temperature[n] + target_range
                    )
                else:
                    q_htg_clg.append(0)
            else:
                q_htg_clg.append(0)
        else:
            q_htg_clg.append(0)
        if i > x and x != -1:
            print("temp before step:" + str(_current_water_temperature))
        if i > x and x != -1:
            print("")
        if i > x + 12 and x != -1:
            return None

    # combine heat gains into a single dataframe
    q_solar = pd.Series(q_solar, index=epw_df.index, name=("Gains", "Q_Solar (W)"))
    q_occupants = pd.Series(
        q_occupants, index=epw_df.index, name=("Gains", "Q_Occupants (W)")
    )
    q_longwave = pd.Series(
        q_longwave, index=epw_df.index, name=("Gains", "Q_Longwave (W)")
    )
    q_conduction = pd.Series(
        q_conduction, index=epw_df.index, name=("Gains", "Q_Conduction (W)")
    )
    q_evaporation = pd.Series(
        q_evaporation, index=epw_df.index, name=("Gains", "Q_Evaporation (W)")
    )
    q_convection = pd.Series(
        q_convection, index=epw_df.index, name=("Gains", "Q_Convection (W)")
    )
    q_conditioning_water_temp = pd.Series(
        q_htg_clg, index=epw_df.index, name=("Other", "Q_Conditioning [WATERTEMP] (W)")
    )
    q_conditioning_heat_balance = pd.Series(
        q_energy_balance,
        index=epw_df.index,
        name=("Other", "Q_Conditioning [HEATBALANCE] (W)"),
    )

    water_temperature_without_htgclg = pd.Series(
        water_temperature_without_htgclg,
        index=epw_df.index,
        name=("Other", "Water Temperature [without htg/clg] (C)"),
    )
    remaining_water_temperature = pd.Series(
        remaining_water_temperature,
        index=epw_df.index,
        name=("Other", "Water Temperature [prior to resupply] (C)"),
    )
    
    evap_rate = pd.Series(
        evap_rate,
        index=epw_df.index,
        name=("Other", "Evaporation Rate (mm/h)")
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
    )
