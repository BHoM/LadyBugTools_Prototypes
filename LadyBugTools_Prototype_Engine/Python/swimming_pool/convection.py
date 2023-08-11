import numpy as np
import pandas as pd
from ladybugtools_toolkit.ladybug_extension.epw import (
    EPW,
    collection_to_series,
    wind_speed_at_height,
)

from .evaporation import (
    evaporation_gain_mancic,
    evaporation_gain_woolley,
    evaporation_rate_penman,
    saturated_vapor_pressure,
    vapor_pressure_antoine,
    WIND_HEIGHT_ABOVE_WATER,
)


def convection_gain_czarnecki(
    surface_area: float, epw: EPW, water_temperature: float
) -> pd.Series:
    """Calculate the convection gain from the air on a surface of water.

    Source:
        Swimming pool heating by solar energy, Czarnecki 1978

    Args:
        surface_area (float): Surface area of the pool in m2.
        epw (EPW): EPW object.
        water_temperature (float): Temperature of the pool in C.

    Returns:
        pd.Series: Convection gain from the air on a surface in W.
    """

    wind_speed = wind_speed_at_height(
        collection_to_series(epw.wind_speed), 10, WIND_HEIGHT_ABOVE_WATER
    )

    air_temperature = collection_to_series(epw.dry_bulb_temperature)

    h_c = 3.1 + 4.1 * wind_speed

    return h_c * (water_temperature - air_temperature) * surface_area  # W


def convection_gain_bowen_ratio(
    surface_area: float, epw: EPW, water_temperature: float
) -> float:
    """Calculate the convection gain from the air on a surface of water.

    Source:
        Eq 6 from "Nouaneque, H.V., et al, (2011), Energy model validation of
        heated outdoor swimming pools in cold weather. SimBuild 2011.

        Relationship between convective and evaporation heat losses from ice surfaces, Williams, G. P. (1959)

    Args:
        surface_area (float): Surface area of the pool in m2.
        epw (EPW): An EPW object.
        water_temperature (float): Temperature of the water in C.

    Returns:
        pd.Series: A pandas series of evaporation gain in W.

    """

    air_temperature = collection_to_series(epw.dry_bulb_temperature)
    air_vapor_pressure_mb = vapor_pressure_antoine(air_temperature) * 0.01
    sat_vapor_pressure_mb = (
        pd.Series(
            np.vectorize(saturated_vapor_pressure)(air_temperature + 273.15),
            index=air_temperature.index,
        )
        * 0.01
    )
    atmos_press_mb = collection_to_series(epw.atmospheric_station_pressure) * 0.01

    evaporation_gain = evaporation_gain_mancic(surface_area, epw)

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
