import pandas as pd


def supply_water_heating(
    water_volume_m3_hour: float,
    supply_temperature: float,
    target_temperature: float,
) -> float:
    """The heat rate required to heat up the makeup water to the target temperature.

    Args:
        water_volume_m3_hour (float): Volume of the water in m3/hour.
        supply_temperature (float): Temperature of the water in C.
        target_temperature (float): Target temperature of the water in C.
        pressure (float, optional): Pressure of the water in Pa. Defaults to 101325.

    Returns:
        float: Heat rate required to heat up the makeup water to the target temperature in Wh.
    """

    water_specific_heat_capacity = 4186  # J/kg.K
    water_density = 1000  # kg/m3
    water_mass = water_volume_m3_hour * water_density  # kg
    energy = (
        water_mass
        * water_specific_heat_capacity
        * (target_temperature - supply_temperature)
        / 3600
    )  # Wh

    if isinstance(energy, pd.Series):
        energy.rename("Supply Water Heating (Wh)", inplace=True)

    return energy
