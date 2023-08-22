import numpy as np
import pandas as pd
import pyet
from ladybugtools_toolkit.ladybug_extension.epw import (
    EPW,
    Sunpath,
    collection_to_series,
    wind_speed_at_height,
)
from ladybug.psychrometrics import saturated_vapor_pressure
import pint


def latent_heat_of_vaporisation(air_temperature: float):
    """Calculate the latent heat of vapourisation from air temperature.

    Args:
        airtemp: (float)
            Air temperaturem in degrees C.

    Returns:
        float:
            lambda [J kg-1 K-1].

    Source:
        J. Bringfelt. Test of a forest evapotranspiration model. Meteorology and
        Climatology Reports 52, SMHI, Norrköpping, Sweden, 1986.
    """

    return 4185.5 * (751.78 - 0.5655 * (air_temperature + 273.15))  # J/kg.K


def vapor_pressure_antoine(temperature_c: float) -> float:
    """Calculate the vapor pressure of water at a given temperature using the Antoine equation.

    Args:
        temperature (float):
            The temperature of water in degrees celsius.

    Returns:
        float:
            The vapor pressure of water in Pascals, Pa.
    """

    if isinstance(temperature_c, (pd.Series, np.ndarray)):
        coefficient_a = np.where(temperature_c < 100, 8.07131, 8.14019)
        coefficient_b = np.where(temperature_c < 100, 1730.63, 1810.94)
        coefficient_c = np.where(temperature_c < 100, 233.426, 244.485)
    elif isinstance(temperature_c, (float, int)):
        coefficient_a = 8.07131 if temperature_c < 100 else 8.14019
        coefficient_b = 1730.63 if temperature_c < 100 else 1810.94
        coefficient_c = 233.426 if temperature_c < 100 else 244.485

    pressure_mmhg = 10 ** (
        coefficient_a - coefficient_b / (temperature_c + coefficient_c)
    )
    pressure_pa = pressure_mmhg * 133.322

    if isinstance(temperature_c, pd.Series):
        return pd.Series(
            pressure_pa, index=temperature_c.index, name="Vapor Pressure (Pa)"
        )

    return pressure_pa


def evaporation_rate(epw: EPW, wind_height_above_water: float = 1) -> pd.Series:
    """Estimate the volume of water evaporated from a body of water of given surface area.

    Source:
        Using the Penman-Monteith equation to predict daily open-water evaporation across inland waters, McMahon et al. 2013
        https://github.com/pyet-org/pyet/blob/dev/examples/06_worked_examples_McMahon_etal_2013.ipynb

    Args:
        epw (EPW): EPW object.
        wind_height_above_water (float): The height of the wind speed measurement above the water surface in m.

    Returns:
        pd.Series: Evaporation rate in l/m2/hour.
    """

    # get select inputs from EPW file
    _dbt = collection_to_series(epw.dry_bulb_temperature)
    _rh = collection_to_series(epw.relative_humidity)

    # resample to daily values
    dbt_day = _dbt.resample("D")
    rh_day = _rh.resample("D")
    idx_day = dbt_day.mean().index

    # obtain sun-up time for each day
    sunpath = Sunpath.from_location(epw.location)
    sunshine_duration_hours = pd.Series(
        [
            (i["sunset"] - i["sunrise"]).total_seconds() / (60 * 60)
            for i in [
                sunpath.calculate_sunrise_sunset(month, day)
                for month, day in list(zip(*[idx_day.month, idx_day.day]))
            ]
        ],
        index=dbt_day.mean().index,
    )

    # Estimate daily open-water evaporation using Penman
    penman = pyet.penman(
        tmean=dbt_day.mean(),
        wind=wind_speed_at_height(
            collection_to_series(epw.wind_speed), 10, wind_height_above_water
        )
        .resample("D")
        .mean(),
        rs=collection_to_series(epw.global_horizontal_radiation).resample("D").sum()
        * 0.0036,  # [MJ m-2 d-1] from the original [Wh m-2 hour-1]
        tmax=dbt_day.max(),
        tmin=dbt_day.min(),
        rhmin=rh_day.min(),
        rhmax=rh_day.max(),
        rh=rh_day.mean(),
        pressure=collection_to_series(epw.atmospheric_station_pressure)
        .resample("D")
        .mean()
        / 1000,  # [kPa] from the original [Pa]
        n=sunshine_duration_hours,
        lat=np.deg2rad(epw.location.latitude),
        elevation=epw.location.elevation,
        aw=1.313,
        bw=1.381,
        albedo=0.08,
        clip_zero=True,
    )

    # We have to divide the coefficient from McMahon et al. 2013 as PyEts Penman uses an unit conversion factor Ku
    lambda0 = 2.45
    lambda1 = pyet.calc_lambda(dbt_day.mean())
    lambda_cor = lambda1 / lambda0
    penman_openwater_daily = (
        penman / lambda_cor
    )  # in units of mm/day, or l/m2/day, or kg/m2/day (if we assume 1l = 1kg)

    # Distribute values over course of day based on relative humidity (where highest RH equals lowest evaporation rate)
    rh_profile = (100 - _rh).groupby(
        [_rh.index.dayofyear, _rh.index.hour]
    ).sum().unstack().T / (100 - _rh).groupby([_rh.index.dayofyear]).sum()
    dbt_profile = (
        _dbt.groupby([_dbt.index.dayofyear, _dbt.index.hour]).sum().unstack().T
        / _dbt.groupby([_dbt.index.dayofyear]).sum()
    )

    res = pd.Series(
        (
            (
                pd.DataFrame(
                    np.average([rh_profile, dbt_profile], axis=0, weights=[1, 1])
                )
                * penman_openwater_daily.values
            )
            .unstack()
            .values
        ),
        index=_dbt.index,
        name="Evaporation Rate (l/m2/hour)",
    )

    return res


def evaporation_gain(
    evap_rate: float,
    surface_area: float,
    latent_heat_of_evaporation: float = 2260,
    water_density: float = 1000,
) -> float:
    """Calculate the evaporation gain from a body of water based on the volume
    of water being lost from it.

    Args:
        evaporation_rate (float): The rate of evaporation in l/m2/hour or mm/hour.
        surface_area (float): Surface area of the pool in m2.
        latent_heat_of_evaporation (float): The latent heat of evaporation in kJ/kg.
        water_density (float): The density of water in kg/m3.

    Source:
        Mančić Marko V., Živković Dragoljub S., Milosavljević Peđa M.,
        Todorović Milena N. (2014) Mathematical modelling and simulation of
        the thermal performance of a solar heated indoor swimming pool.
        Thermal Science 2014 Volume 18, Issue 3, Pages: 999-1010,
        https://doi.org/10.2298/TSCI1403999M

    Returns:
        float:
            Evaporation gain in Wh.
    """

    evaporation_rate_l_hour = evap_rate * surface_area  # l/hour
    water_loss_m3_hour = evaporation_rate_l_hour / 1000  # m3/hour

    water_mass = water_loss_m3_hour * water_density  # kg/hour
    water_loss_kg_s = water_mass / (60 * 60)  # kg/s
    res = -water_loss_kg_s * latent_heat_of_evaporation * 1000  # W
    if isinstance(evap_rate, pd.Series):
        return res.rename("Q_Evaporation (W)")
    return res


def convection_gain(
    evap_gain: float,
    air_temperature: float,
    water_temperature: float,
    atmospheric_pressure: float = 101325,
) -> float:
    """Calculate the convection gain from the air on a surface of water based on the Bowen ratio and evaporative losses.

    Source:
        Eq 6 from "Nouaneque, H.V., et al, (2011), Energy model validation of
        heated outdoor swimming pools in cold weather. SimBuild 2011.

        Relationship between convective and evaporation heat losses from ice surfaces, Williams, G. P. (1959)

    Args:
        evap_gain (float): Evaporation gain in W.
        air_temperature (float): The air temperature in C.
        water_temperature (float): Temperature of the water in C.
        atmospheric_pressure (float): Atmospheric pressure in Pa.

    Returns:
        pd.Series: A pandas series of evaporation gain in W.

    """

    air_vapor_pressure_mb = vapor_pressure_antoine(air_temperature) / 100
    if isinstance(air_temperature, (pd.Series, np.ndarray)):
        sat_vapor_pressure_mb = (
            np.vectorize(saturated_vapor_pressure)(air_temperature + 273.15) / 100
        )
    else:
        sat_vapor_pressure_mb = saturated_vapor_pressure(air_temperature + 273.15) / 100
    atmos_press_mb = atmospheric_pressure / 100

    C_v = 0.63  # average of 0.6 and 0.66

    R_b = (
        C_v
        * (
            (water_temperature - air_temperature)
            / (sat_vapor_pressure_mb - air_vapor_pressure_mb)
        )
        * atmos_press_mb
        / 1000
    )
    res = R_b * evap_gain / 1000
    if isinstance(res, pd.Series):
        return res.rename("Q_Convection (W)")
    return res
