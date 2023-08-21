from typing import List, Tuple, Union
from ladybugtools_toolkit.ladybug_extension.epw import (
    EPW,
    epw_to_dataframe,
    wind_speed_at_height,
)
import numpy as np
import pandas as pd
from tqdm import tqdm
from iapws import SeaWater
import pint

import matplotlib.pyplot as plt

from evaporation import evaporation_rate_penman
from conduction import ground_interface_area_prism
from ladybug.psychrometrics import saturated_vapor_pressure


ureg = pint.UnitRegistry()


def vapor_pressure_antoine(temperature: float) -> float:
    """Calculate the vapor pressure of water at a given temperature using the Antoine equation.

    Args:
        temperature (float):
            The temperature of water.

    Returns:
        float:
            The vapor pressure of water in Pascals, Pa.
    """
    try:
        temperature = temperature.to("degC").magnitude
    except:
        pass
    coefficient_a = 8.07131 if temperature < 100 else 8.14019
    coefficient_b = 1730.63 if temperature < 100 else 1810.94
    coefficient_c = 233.426 if temperature < 100 else 244.485

    pressure_mmhg = 10 ** (
        coefficient_a - coefficient_b / (temperature + coefficient_c)
    )

    return (pressure_mmhg * ureg.millimeter_Hg).to("pascal")


def main(
    epw_file: str = r"C:\Users\tgerrish\Buro Happold\0053340 DDC - Project W Master - Climate\EPW_Modified\SAU_MD_Prince.Abdulmajeed.Bin.Abdulaziz.AP.404010_TMYx.2007-2021_FIXED_TG_300M.epw",
    surface_area: float = 1,
    avg_depth: float = 1,
    target_water_temperature: Union[float, Tuple[float]] = 29,
    target_water_temperature_band: float = 1,
    supply_water_temperature: Union[float, Tuple[float]] = None,
    n_occupants: Union[int, Tuple[int]] = 0,
    ground_interface_u_value: float = 0.25,
):
    """

    Assumptions:
    - The pool is a rectangular prism
    - The pool is filled with water of a fixed salinity (0.0038 kg/kg, for seawater typical of Persian Gulf, from which thermophysical properties are derived)
    - Water is supplied to the pool at the same rate as it is evaporated, so the water level remains constant.

    Args:
        epw_file (_type_, optional): _description_.
        surface_area (float, optional): _description_. Defaults to 10.
        avg_depth (float, optional): _description_. Defaults to 1.
        target_water_temperature (Union[float, Tuple[float]], optional): _description_. Defaults to 29.
        target_water_temperature_band (float, optional): _description_. Defaults to 1.
        supply_water_temperature (Union[float, Tuple[float]], optional): _description_. Defaults to None.
        water_salinity (float, optional): _description_. Defaults to 0.0038.
        supply_water_salinity (float, optional): _description_. Defaults to None.

    """

    # Globals #
    water_salinity = 0.0038 * ureg.dimensionless
    occupant_gain = 300 * ureg.watt
    water_emissivity = 0.95 * ureg.dimensionless
    surface_area = surface_area * ureg.meter**2
    avg_depth = avg_depth * ureg.meter
    target_water_temperature_band = target_water_temperature_band * ureg.degC
    ground_interface_u_value = ground_interface_u_value * (
        ureg.watt / (ureg.meter**2 * ureg.kelvin)
    )

    # Load EPW data
    epw = EPW(epw_file)
    epw_df = epw_to_dataframe(epw, include_additional=True).droplevel([0, 1], axis=1)
    air_temperature = epw_df["Dry Bulb Temperature (C)"].values * ureg.degC
    sky_temperature = epw_df["Sky Temperature (C)"].values * ureg.degC
    air_pressure = epw_df["Atmospheric Station Pressure (Pa)"].values * ureg.pascal
    wind_speed = (
        wind_speed_at_height(
            reference_value=epw_df["Wind Speed (m/s)"],
            reference_height=10,
            target_height=1,
        ).values
        * ureg.meter
        / ureg.second
    )
    ground_temperature = epw_df["Ground Temperature (C)"].values * ureg.degC
    insolation = (
        epw_df["Global Horizontal Radiation (Wh/m2)"].values
        * ureg.watt
        * ureg.hour
        / ureg.meter**2
    )
    evaporation_rate = (
        (evaporation_rate_penman(epw=epw).values * ureg.litre / ureg.meter**2)
        * surface_area
    ).to("meter ** 3")

    if supply_water_temperature is None:
        supply_water_temperature = epw_df["Ground Temperature (C)"].values
    elif isinstance(supply_water_temperature, (float, int)):
        supply_water_temperature = np.ones(len(epw_df)) * supply_water_temperature
    else:
        if len(supply_water_temperature) != len(epw_df):
            raise ValueError(
                "The length of the supply water temperature profile must match the length of the EPW file."
            )
    supply_water_temperature = supply_water_temperature * ureg.degC

    if isinstance(target_water_temperature, (float, int)):
        target_water_temperature = np.ones(len(epw_df)) * target_water_temperature
    else:
        if len(target_water_temperature) != len(epw_df):
            raise ValueError(
                "The length of the supply water temperature profile must match the length of the EPW file."
            )
    target_water_temperature = target_water_temperature * ureg.degC

    if isinstance(n_occupants, (float, int)):
        n_occupants = np.ones(len(epw_df)) * n_occupants
    else:
        if len(n_occupants) != len(epw_df):
            raise ValueError(
                "The length of the n_occupants profile must match the length of the EPW file."
            )
    occupant_gain = occupant_gain * n_occupants

    # Derive properties from inputs
    container_volume = surface_area * avg_depth
    container_interface_area = ground_interface_area_prism(
        surface_area=surface_area, average_depth=avg_depth
    )

    # Initial conditions
    _current_water_temperature = target_water_temperature[0]
    _current_air_pressure = air_pressure[0]

    water = SeaWater(
        T=_current_water_temperature.to("kelvin").magnitude,
        P=_current_air_pressure.to("megapascal").magnitude,
        S=water_salinity.magnitude,
    )
    _current_water_density = water.rho * ureg.kilogram / ureg.meter**3
    _current_water_isobaric_heat_capacity = water.cp * (
        ureg.kilojoule / (ureg.kilogram * ureg.kelvin)
    )
    _current_water_specific_enthalpy = water.h * ureg.kilojoule / ureg.kilogram

    q_evaporation = []
    q_occupants = []
    q_conduction = []
    q_convection = []
    q_longwave = []
    q_solar = []
    pbar = tqdm(list(enumerate(epw_df.iterrows())))
    for n, (idx, vals) in pbar:
        if idx.hour == 0:
            pbar.set_description(f"Calculating {idx:%b %d}")

        # calculate heat balance or this point in time
        _evap_gain = (
            -evaporation_rate[n]
            * _current_water_density
            * _current_water_specific_enthalpy
        ).to(ureg.hour * ureg.watt)
        q_evaporation.append(_evap_gain)

        q_solar.append(insolation[n] * surface_area)
        q_occupants.append(occupant_gain[n] * ureg.hour)
        q_conduction.append(
            ground_interface_u_value
            * container_interface_area
            * (_current_water_temperature - ground_temperature[n]).to("kelvin")
            * ureg.hour
        )
        q_longwave.append(
            (
                (5.670374419e-8 * ureg.watt / (ureg.meter**2 * ureg.kelvin**4))
                * water_emissivity
                * (
                    (sky_temperature[n].to("kelvin") ** 4)
                    - (_current_water_temperature.to("kelvin") ** 4)
                )
            )
            * surface_area
            * ureg.hour
        )
        q_convection.append(
            (
                0.63
                * ureg.dimensionless
                * (
                    (_current_water_temperature - air_temperature[n]).to("kelvin")
                    / (
                        (
                            saturated_vapor_pressure(
                                air_temperature[n].to("kelvin").magnitude
                            )
                            * ureg.pascal
                        ).to("millibar")
                        - vapor_pressure_antoine(air_temperature[n]).to("millibar")
                    )
                )
                * air_pressure[n].to("millibar")
            )
            * _evap_gain
        )

    print(pint.Quantity.from_list(q_convection))
    # water = SeaWater(
    #     T=_current_water_temperature.to("kelvin").magnitude,
    #     P=_current_air_pressure.to("megapascal").magnitude,
    #     S=water_salinity.magnitude,
    # )

    #     SeaWater(
    #         T=273.15 + water_temperature,  # temperature, K
    #         P=air_pressure / 1000000,  # pressure, MPa
    #         S=_current_water_salinity,  # convert from x.x% to kg/kg
    #     )

    # df = pd.DataFrame(
    #     np.array(
    #         [
    #             water_volume_ts,
    #             water_salinity_ts,
    #         ],
    #         dtype=object,
    #     ).T,
    #     columns=[
    #         "Water Volume (m3)",
    #         "Water Salinity (kg/kg)",
    #     ],
    #     index=epw_df.index,
    # )
    # print(df)

    # df.plot()
    # plt.show()


if __name__ == "__main__":
    main()
