import numpy as np
import pandas as pd
import pyet
from ladybug.psychrometrics import saturated_vapor_pressure
from ladybugtools_toolkit.ladybug_extension.epw import (
    EPW,
    Sunpath,
    collection_to_series,
    wind_speed_at_height,
    humid_ratio_from_db_rh,
)
from sklearn.linear_model import LinearRegression

WIND_HEIGHT_ABOVE_WATER = 1  # m


def vapor_pressure_antoine(temperature_c: float) -> float:
    """Calculate the vapor pressure of water at a given temperature using the Antoine equation.

    Args:
        temperature_c (float):
            The temperature in degrees Celsius.

    Returns:
        float:
            The vapor pressure of water in Pascals, Pa.
    """
    coefficient_a = np.where(temperature_c < 100, 8.07131, 8.14019)
    coefficient_b = np.where(temperature_c < 100, 1730.63, 1810.94)
    coefficient_c = np.where(temperature_c < 100, 233.426, 244.485)

    pressure_mmhg = 10 ** (
        coefficient_a - coefficient_b / (temperature_c + coefficient_c)
    )

    return pressure_mmhg * 133.322  # Pa


def evaporation_rate_jamesramsden(epw: EPW) -> pd.Series:
    """Calculate the evaporation rate of water from a body of water of given temperature.

    Source:
        https://www.engineeringtoolbox.com/evaporation-water-surface-d_690.html

    Args:
        epw (EPW): EPW object.

    Returns:
        pd.Series: Evaporation rate in l/m2/hour.
    """

    air_temperature = collection_to_series(epw.dry_bulb_temperature)
    wind_speed = wind_speed_at_height(
        collection_to_series(epw.wind_speed), 10, WIND_HEIGHT_ABOVE_WATER
    )
    relative_humidity = collection_to_series(epw.relative_humidity)
    air_pressure = collection_to_series(epw.atmospheric_station_pressure)

    hr_func = np.vectorize(humid_ratio_from_db_rh)

    humidity_ratio = hr_func(air_temperature, relative_humidity, air_pressure)
    max_humidity_ratio = hr_func(
        air_temperature, 100, air_pressure
    )  # NOTE - water_tempertrue replaced with air temperature heere to agree with other methods. Assuming that evaporation is drvien bnot by th e"boiling" oif water and its temrpeatures, but by the difference between the humidity ratio of the air and the maximum humidity ratio of the air.
    evaporation_coefficient = 25 + 19 * wind_speed

    return (evaporation_coefficient * (max_humidity_ratio - humidity_ratio)).rename(
        "Evaporation Rate (l/m2/hour)"
    )


def evaporation_rate_bensmallwood(epw: EPW) -> pd.Series:
    """Calculate the evaporation rate of water from a body of water of given temperature.

    Source:
        Old Excel sheet from Ben Smallwood.

    Args:
        epw (EPW): EPW object.

    Returns:
        pd.Series: Evaporation rate in l/m2/hour.
    """

    air_temperature = collection_to_series(epw.dry_bulb_temperature)  # C
    atmospheric_pressure = collection_to_series(epw.atmospheric_station_pressure)  # Pa
    rh = collection_to_series(epw.relative_humidity)  # %
    saturation_pressure_water_vapor = (
        np.exp(
            77.345
            + (0.0057 * (273.15 + air_temperature))
            - (7235 / (273.15 + air_temperature))
        )
    ) / (
        273.15 + air_temperature
    ) ** 8.2  # Pa
    partial_pressure_water_vapor = (rh / 100) * saturation_pressure_water_vapor  # Pa

    humidity_ratio_vapor_pressure = (0.62198 * partial_pressure_water_vapor) / (
        atmospheric_pressure - partial_pressure_water_vapor
    )  # kg/kg
    humidity_ratio_saturation_vapor_pressure = (
        0.62198
        * (saturation_pressure_water_vapor)
        / (atmospheric_pressure - saturation_pressure_water_vapor)
    )  # kg/kg

    evaporation_coefficient = 25 + 19 * wind_speed_at_height(
        collection_to_series(epw.wind_speed), 10, WIND_HEIGHT_ABOVE_WATER
    )

    water_loss_from_surface = evaporation_coefficient * (
        humidity_ratio_saturation_vapor_pressure - humidity_ratio_vapor_pressure
    )  # mm/hour, or l/m2/hour

    return water_loss_from_surface.rename("Evaporation Rate (l/m2/hour)")


def evaporation_rate_penman(epw: EPW) -> pd.Series:
    """Estimate the volume of water evaporated from a body of water of given surface area.

    Source:
        https://github.com/pyet-org/pyet/blob/dev/examples/06_worked_examples_McMahon_etal_2013.ipynb

    Args:
        epw (EPW): EPW object.

    Returns:
        pd.Series: Evaporation rate in l/m2/hour.
    """

    _dbt = collection_to_series(epw.dry_bulb_temperature)
    tmean_day = _dbt.resample("D").mean()
    tmax_day = _dbt.resample("D").max()
    tmin_day = _dbt.resample("D").min()

    _ws = wind_speed_at_height(
        collection_to_series(epw.wind_speed), 10, WIND_HEIGHT_ABOVE_WATER
    )
    wind_day = _ws.resample("D").mean()  # [m s-1]

    _rh = collection_to_series(epw.relative_humidity)
    rhmean_day = _rh.resample("D").mean()
    rhmax_day = _rh.resample("D").max()
    rhmin_day = _rh.resample("D").min()

    _atm = collection_to_series(epw.atmospheric_station_pressure)
    atm_day = _atm.resample("D").mean()

    rad = collection_to_series(epw.global_horizontal_radiation)
    rad_day = rad.resample("D").sum()

    sunpath = Sunpath.from_location(epw.location)
    sunshine_duration_hours = pd.Series(
        [
            (i["sunset"] - i["sunrise"]).total_seconds() / (60 * 60)
            for i in [
                sunpath.calculate_sunrise_sunset(month, day)
                for month, day in list(
                    zip(*[tmean_day.index.month, tmean_day.index.day])
                )
            ]
        ],
        index=tmean_day.index,
    )

    # Estimate daily open-water evaporation using Penman
    wind_coefficient_a = 1.313  # from
    wind_coefficient_b = 1.381
    penman = pyet.penman(
        tmean=tmean_day,  # [deg C]
        wind=wind_day,
        rs=rad_day * 0.0036,  # [MJ m-2 d-1]
        tmax=tmax_day,  # [deg C]
        tmin=tmin_day,  # [deg C]
        rhmin=rhmin_day,  # [%]
        rhmax=rhmax_day,
        rh=rhmean_day,
        pressure=atm_day / 1000,  # [kPa]
        n=sunshine_duration_hours,
        lat=np.deg2rad(epw.location.latitude),
        elevation=epw.location.elevation,
        aw=wind_coefficient_a,
        bw=wind_coefficient_b,
        albedo=0.08,  # value used for water
        clip_zero=True,
    )

    # We have to divide the coefficient from McMahon et al. 2013 as PyEts Penman uses an unit conversion factor Ku
    lambda0 = 2.45
    lambda1 = pyet.calc_lambda(tmean_day)
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
    overall_profile = np.average([rh_profile, dbt_profile], axis=0, weights=[1, 1])
    penman_openwater_hourly = pd.Series(
        (pd.DataFrame(overall_profile) * penman_openwater_daily.values)
        .unstack()
        .values,
        index=_rh.index,
        name="Evaporation Rate (l/m2/hour)",
    )

    return penman_openwater_hourly


def evaporation_gain_bensmallwood(surface_area: float, epw: EPW) -> float:
    """Using the method defined in the calculator created by Ben Smallwood,
    where enthalpy of vaporisation is used to calculate the energy required to
    evaporate a given volume of water, which equates to energy lost from the
    body of water.

    Args:
        surface_area (float): Surface area of the pool in m2.
        epw (EPW): EPW object.

    Returns:
        float:
            Energy lost from the body of water in W.
    """

    evaporation_rate = evaporation_rate_bensmallwood(epw)  # l/m2/hour
    evaporation_rate_l_hour = evaporation_rate * surface_area  # l/hour
    water_loss_m3_hour = evaporation_rate_l_hour / 1000  # m3/hour

    enthalpy_vaporisation_water = (
        2257.000  # kJ/kg  # TODO - shouldnt this be J/kg??????
    )
    water_density = 1000  # kg/m3
    water_mass = water_loss_m3_hour * water_density  # kg
    water_loss_kg_s = water_mass / (60 * 60)  # kg/s
    return -(water_loss_kg_s * enthalpy_vaporisation_water)  # kW


def evaporation_gain_woolley(
    surface_area: float, epw: EPW, water_temperature: float
) -> pd.Series:
    """Calculate the evaporation gain from a body of water.

    Source:
        Woolley J, et al., Swimming pools as heat sinks for air conditioners:
        Model design and experimental..., Building and Environment (2010),
        doi:10.1016/j.buildenv.2010.07.014

    Args:
        surface_area (float): Surface area of the pool in m2.
        epw (EPW): An EPW object.
        water_temperature (float): Temperature of the water in C.

    Returns:
        pd.Series: A pandas series of evaporation gain in W.
    """

    wind_speed = wind_speed_at_height(
        collection_to_series(epw.wind_speed), 10, WIND_HEIGHT_ABOVE_WATER
    )
    air_temperature = collection_to_series(epw.dry_bulb_temperature)

    vapor_pressure = vapor_pressure_antoine(
        air_temperature + 273.15
    )  # vapor pressure in ambient air, Pa
    sat_vapor_pressure = np.vectorize(saturated_vapor_pressure)(
        water_temperature + 273.15
    )  # saturation vapor pressure of air at the pool temperature, Pa
    h_evap = (
        0.0360 + 0.0250 * wind_speed
    )  # wind speed function for evaporation, W/m2.Pa
    q_evap = h_evap * (sat_vapor_pressure - vapor_pressure) * surface_area  # W
    return q_evap


def evaporation_gain_mancic(surface_area: float, epw: EPW) -> float:
    """Calculate the evaporation gain from a body of water based on the volume
    of water being lost from it.

    Args:
        surface_area (float): Surface area of the pool in m2.
        epw (EPW): EPW object.

    Source:
        Mančić Marko V., Živković Dragoljub S., Milosavljević Peđa M.,
        Todorović Milena N. (2014) Mathematical modelling and simulation of
        the thermal performance of a solar heated indoor swimming pool.
        Thermal Science 2014 Volume 18, Issue 3, Pages: 999-1010,
        https://doi.org/10.2298/TSCI1403999M

    Returns:
        float:
            Evaporation gain in W.
    """

    evaporation_rate = evaporation_rate_penman(epw)  # l/m2/hour or mm/hour
    evaporation_rate_l_hour = evaporation_rate * surface_area  # l/hour
    water_loss_m3_hour = evaporation_rate_l_hour / 1000  # m3/hour

    latent_heat_of_evaporation = 2260.000  # kJ/kg
    water_density = 1000  # kg/m3
    water_mass = water_loss_m3_hour * water_density  # kg/hour
    water_loss_kg_s = water_mass / (60 * 60)  # kg/s
    return -water_loss_kg_s * latent_heat_of_evaporation * 1000  # W


def evaporation_gain_jamesramsden(surface_area: float, epw: EPW) -> float:
    """Approximate the evaporation gain from a body of water based on the volume
    of water being lost from it.

    Args:
        surface_area (float): Surface area of the pool in m2.
        epw (EPW): EPW object.

    Source:
        James Ramsden, 2023 + EngineeringToolbox, https://www.engineeringtoolbox.com/water-properties-d_1573.html

    Returns:
        float:
            Evaporation gain in W.
    """

    evaporation_rate = evaporation_rate_jamesramsden(epw)  # l/m2/hour
    evaporation_rate_l_hour = evaporation_rate * surface_area  # l/hour
    water_loss_m3_hour = evaporation_rate_l_hour / 1000  # m3/hour

    heat_of_vaporisation = 2454.000  # J/kg, assuming room temperature water # TODO - shouldnt this be J/kg??????
    water_density = 1000  # kg/m3
    water_mass = water_loss_m3_hour * water_density  # kg
    water_loss_kg_s = water_mass / (60 * 60)  # kg/s
    return -water_loss_kg_s * heat_of_vaporisation  # kW
