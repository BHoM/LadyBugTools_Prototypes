import numpy as np
import pandas as pd

# from ladybugtools_toolkit.ladybug_extension.epw import (
#     EPW,
#     collection_to_series,
#     wind_speed_at_height,
# )

from .evaporation import (
    #     evaporation_gain_mancic,
    #     # evaporation_gain_woolley,
    #     # evaporation_rate_penman,
    saturated_vapor_pressure,
    vapor_pressure_antoine,
    #     WIND_HEIGHT_ABOVE_WATER,
)


def convection_gain_czarnecki(
    surface_area: float,
    wind_speed_at_1m: float,
    air_temperature: float,
    water_temperature: float,
) -> pd.Series:
    """Calculate the convection gain from the air on a surface of water.

    Source:
        Swimming pool heating by solar energy, Czarnecki 1978

    Args:
        surface_area (float): Surface area of the pool in m2.
        wind_speed_at_1m (float): The wind speed at 1m above the water surface in m/s.
        air_temperature (float): Temperature of the air in C.
        water_temperature (float): Temperature of the pool in C.

    Returns:
        pd.Series: Convection gain from the air on a surface in W.
    """

    h_c = 3.1 + 4.1 * wind_speed_at_1m
    return h_c * (water_temperature - air_temperature) * surface_area  # W


def convection_gain_bowen_ratio(
    air_temperature: float,
    water_temperature: float,
    atmospheric_pressure: float,
    evaporation_gain: float,
) -> float:
    """Calculate the convection gain from the air on a surface of water based on the Bowen ratio and evaporative losses.

    Source:
        Eq 6 from "Nouaneque, H.V., et al, (2011), Energy model validation of
        heated outdoor swimming pools in cold weather. SimBuild 2011.

        Relationship between convective and evaporation heat losses from ice surfaces, Williams, G. P. (1959)

    Args:
        air_temperature (float): The air temperature in C.
        water_temperature (float): Temperature of the water in C.
        atmospheric_pressure (float): Atmospheric pressure in Pa.
        evaporation_gain (float): Evaporation gain in W.

    Returns:
        pd.Series: A pandas series of evaporation gain in W.

    """

    air_vapor_pressure_mb = vapor_pressure_antoine(air_temperature) * 0.01
    sat_vapor_pressure_mb = (
        np.vectorize(saturated_vapor_pressure)(air_temperature + 273.15) * 0.01
    )
    atmos_press_mb = atmospheric_pressure * 0.01

    C_v = (0.6 + 0.66) / 2  # average of 0.6 and 0.66

    R_b = (
        C_v
        * (
            (water_temperature - air_temperature)
            / (sat_vapor_pressure_mb - air_vapor_pressure_mb)
        )
        * atmos_press_mb
        / 1000
    )

    return R_b * evaporation_gain / 1000
